import os
import sys
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from HarmoMeshNet_config import load_config
from HarmoMeshNet_data import make_splits
from HarmoMeshNet_architecture import HarmoMeshNet
from HarmoMeshNet_losses import compute_losses, chamfer_distance, sample_points_from_meshes
from utils import compute_normal, save_mesh_vedo
from dataclasses import asdict

if __name__ == "__main__":
    config = load_config()
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.optional_id is None:
        exp_dir = os.path.join("./HarmoMeshNet_trainings", f"training_{timestamp}")
    else:
        exp_dir = os.path.join("./HarmoMeshNet_trainings", f"training_{config.optional_id}")
        if os.path.exists(exp_dir) and config.pretrain_file is None:
            print(f'The experiment {config.optional_id} already exist please check it -> {exp_dir}')
            sys.exit(1)

    if config.save_mesh_eval:
        folder_C = ["logs", "ckpts/model", "ckpts/mesh"]
    else:
        folder_C = ["logs", "ckpts/model"]

    for sub in folder_C:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    
    inference_args_str = (
        f"--n_start_filters {config.n_start_filters} "
        f"--latent_dim {config.latent_dim} "
        f"--max_sh_degree {config.max_sh_degree} "
        f"--sphere_subdivisions {config.sphere_subdivisions} "
        f"--refiner_steps {config.refiner_steps} "
        f"--refiner_layers {config.refiner_layers} "
        f"--refiner_hidden {config.refiner_hidden} "
        f"--n_smooth {config.n_smooth} "
        f"--lambd {config.lambd}"
    )
    
    with open(os.path.join(exp_dir, "inference_args.txt"), "w") as fh:
        fh.write(inference_args_str)
    
    with open(os.path.join(exp_dir, "training_execution_config.txt"), "w") as f:
        config_str = json.dumps(asdict(config), indent=4)
        f.write(config_str)

    log_step = os.path.join(exp_dir, "logs", "train_step.txt")
    log_epoch = os.path.join(exp_dir, "logs", "train_epoch.txt")
    log_val = os.path.join(exp_dir, "logs", "val_epoch.txt")
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    
    train_dir = os.path.join(config.data_path, "train")
    train_set, valid_set = make_splits(train_dir, val_split=config.val_split, seed=seed)
    trainloader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=config.num_workers)
    validloader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=config.num_workers)
    
    model = HarmoMeshNet(
        in_ch=1,
        base_f=config.n_start_filters,
        latent_dim=config.latent_dim,
        max_sh_degree=config.max_sh_degree,
        sphere_subdivisions=config.sphere_subdivisions,
        refiner_steps=config.refiner_steps,
        refiner_layers=config.refiner_layers,
        refiner_hidden=config.refiner_hidden,
    ).to(device)
    
    if config.pretrain_file is not None:
        if not os.path.exists(config.pretrain_file):
            sys.exit(1)
        model.load_state_dict(torch.load(config.pretrain_file, map_location=device))
        
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    warmup_ep = max(1, int(config.warmup_ratio * config.n_epoch))
    warmup_sch = LinearLR(optimizer, start_factor=config.warmup_lr_start / config.lr, end_factor=1.0, total_iters=warmup_ep)
    cosine_sch = CosineAnnealingLR(optimizer, T_max=config.n_epoch - warmup_ep, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup_sch, cosine_sch], milestones=[warmup_ep])
    
    loss_scales = {
        "chamfer": config.chamfer_scale,
        "normal": config.normal_consistency_scale,
        "edge": config.edge_length_scale,
        "laplacian": config.laplacian_smoothing_scale,
        "willmore": config.willmore_scale,
        "mode_energy": config.mode_energy_scale,
    }
    
    accum_steps = getattr(config, "accumulate_grad_batches", 1)
    accum_steps = max(1, int(accum_steps))
    n_train = len(trainloader)
    best_val = float("inf")
    no_improve = 0
    epoch_bar = tqdm(range(config.n_epoch), desc="Epoch")
    
    for epoch in epoch_bar:
        model.train()
        epoch_losses = []
        optimizer.zero_grad()
        
        for step, (volume, v_gt, f_gt, _) in enumerate(tqdm(trainloader, desc="Run step", leave=False)):
            volume = volume.to(device)
            v_gt = v_gt.to(device)
            f_gt = f_gt.to(device)
            t0 = time.time()
            v_out, f_out = model(volume, n_smooth=config.n_smooth, lambd=config.lambd)
            loss, breakdown = compute_losses(
                v_pred=v_out, f_pred=f_out,
                v_gt=v_gt, f_gt=f_gt,
                sh_coeffs=model.last_sh_coeffs,
                max_sh_degree=config.max_sh_degree,
                n_pts=v_out.shape[1],
                scales=loss_scales,
            )
            
            is_last_batch = (step + 1) == n_train
            is_accum_ready = (step + 1) % accum_steps == 0
            (loss / accum_steps).backward()
            epoch_losses.append(loss.item())
            
            if is_accum_ready or is_last_batch:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                
            if step % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                parts = " + ".join(f"{k} {v.item():.5f}" for k, v in breakdown.items())
                with open(log_step, "a") as fh:
                    fh.write(f"E{epoch+1}-{step}  total={loss.item():.6f}  {parts}  dt={time.time()-t0:.2f}s  lr={lr:.2e}\n")
                    
        scheduler.step()
        
        if config.report_training_loss:
            avg = np.mean(epoch_losses)
            with open(log_epoch, "a") as fh:
                fh.write(f"epoch_{epoch+1}_{avg:.6f}\n")
                
        if (epoch + 1) % config.ckpts_interval == 0:
            model.eval()
            val_errors = []
            with torch.no_grad():
                for vi, (volume, v_gt, f_gt, _) in enumerate(tqdm(validloader, desc="Val", leave=False)):
                    volume = volume.to(device)
                    v_gt = v_gt.to(device)
                    f_gt = f_gt.to(device)
                    v_out, f_out = model(volume, n_smooth=config.n_smooth, lambd=config.lambd)
                    pts_p = sample_points_from_meshes(v_out, f_out, v_out.shape[1])
                    pts_g = sample_points_from_meshes(v_gt, f_gt, v_out.shape[1])
                    e = 1e3 * chamfer_distance(pts_p, pts_g)[0]
                    val_errors.append(e.item())
                    
                    if config.save_mesh_eval and vi % config.mesh_interval == 0:
                        normal = compute_normal(v_out, f_out)
                        save_mesh_vedo(
                            v_out[0].cpu().numpy(), f_out[0].cpu().numpy(), normal[0].cpu().numpy(),
                            os.path.join(exp_dir, "ckpts", "mesh", f"pred_e{epoch+1}_s{vi}.obj"),
                        )
                        normal_gt = compute_normal(v_gt, f_gt)
                        save_mesh_vedo(
                            v_gt[0].cpu().numpy(), f_gt[0].cpu().numpy(), normal_gt[0].cpu().numpy(),
                            os.path.join(exp_dir, "ckpts", "mesh", f"gt_e{epoch+1}_s{vi}.obj"),
                        )
                        
            avg_val = np.mean(val_errors)
            with open(log_val, "a") as fh:
                fh.write(f"epoch_{epoch+1}_{avg_val:.6f}\n")
                
            if avg_val < best_val:
                best_val = avg_val
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(exp_dir, "ckpts", "model", "best_model.ckpt"))
            else:
                no_improve += 1
                
            if config.save_model:
                torch.save(model.state_dict(), os.path.join(exp_dir, "ckpts", "model", f"model_epoch{epoch+1}.ckpt"))
                
            if no_improve >= config.patience:
                break