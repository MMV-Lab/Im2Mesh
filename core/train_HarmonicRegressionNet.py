import os
import sys
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from HarmonicRegressionNet_config import load_config
from HarmonicRegressionNet_data import make_splits
from HarmonicRegressionNet_architecture import HarmonicRegressionNet
from HarmonicRegressionNet_losses import compute_losses
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
        exp_dir = os.path.join("./HarmonicRegressionNet_trainings", f"training_{timestamp}")
    else:
        exp_dir = os.path.join("./HarmonicRegressionNet_trainings", f"training_{config.optional_id}")
        if os.path.exists(exp_dir) and config.pretrain_file is None:
            print(f'The experiment {config.optional_id} already exist please check it -> {exp_dir}')
            sys.exit(1)

    folder_C = ["logs", "ckpts/model"]
    for sub in folder_C:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    
    with open(os.path.join(exp_dir, "training_execution_config.txt"), "w") as f:
        config_str = json.dumps(asdict(config), indent=4)
        f.write(config_str)

    inference_args_str = (
        f"--n_start_filters {config.n_start_filters} "
        f"--latent_dim {config.latent_dim} "
        f"--target_dim {config.target_dim} "
    )
    with open(os.path.join(exp_dir, "inference_args.txt"), "w") as fh:
        fh.write(inference_args_str)    

    log_step = os.path.join(exp_dir, "logs", "train_step.txt")
    log_epoch = os.path.join(exp_dir, "logs", "train_epoch.txt")
    log_val = os.path.join(exp_dir, "logs", "val_epoch.txt")
    
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    
    train_set, valid_set = make_splits(config.data_path, val_split=config.val_split, seed=seed)
    trainloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    validloader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    model = HarmonicRegressionNet(
        in_ch=1,
        base_f=config.n_start_filters,
        latent_dim=config.latent_dim,
        target_dim=config.target_dim
    ).to(device)
    
    if config.pretrain_file is not None:
        if os.path.exists(config.pretrain_file):
            model.load_state_dict(torch.load(config.pretrain_file, map_location=device))
        else:
            sys.exit(1)
        
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    warmup_ep = max(1, int(config.warmup_ratio * config.n_epoch))
    warmup_sch = LinearLR(optimizer, start_factor=config.warmup_lr_start / config.lr, end_factor=1.0, total_iters=warmup_ep)
    cosine_sch = CosineAnnealingLR(optimizer, T_max=config.n_epoch - warmup_ep, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup_sch, cosine_sch], milestones=[warmup_ep])
    
    accum_steps = max(1, int(config.accumulate_grad_batches))
    n_train = len(trainloader)
    best_val_loss = float("inf")
    no_improve = 0
    epoch_bar = tqdm(range(config.n_epoch), desc="Epoch")
    
    for epoch in epoch_bar:
        model.train()
        epoch_losses = []
        optimizer.zero_grad()
        
        for step, (volume, target_vector, _) in enumerate(tqdm(trainloader, desc="Run step", leave=False)):
            volume = volume.to(device)
            target_vector = target_vector.to(device)
            t0 = time.time()
            
            pred_vector = model(volume)
            loss, breakdown = compute_losses(pred_vector, target_vector, loss_type=config.loss_type)
            
            is_last_batch = (step + 1) == n_train
            is_accum_ready = (step + 1) % accum_steps == 0
            (loss / accum_steps).backward()
            epoch_losses.append(loss.item())
            
            if is_accum_ready or is_last_batch:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            if step % 20 == 0:
                lr = optimizer.param_groups[0]["lr"]
                parts = "  ".join(f"{k}={v.item():.5f}" for k, v in breakdown.items())
                with open(log_step, "a") as fh:
                    fh.write(f"E{epoch+1}-{step} total_loss={loss.item():.6f} | {parts} | dt={time.time()-t0:.2f}s | lr={lr:.2e}\n")
                    
        scheduler.step()
        
        if config.report_training_loss:
            avg_train_loss = np.mean(epoch_losses)
            with open(log_epoch, "a") as fh:
                fh.write(f"epoch_{epoch+1}_{avg_train_loss:.6f}\n")
                
        if (epoch + 1) % config.ckpts_interval == 0:
            model.eval()
            val_losses = []
            val_maes = []
            val_r2s = []
            
            with torch.no_grad():
                for volume, target_vector, _ in tqdm(validloader, desc="Val", leave=False):
                    volume = volume.to(device)
                    target_vector = target_vector.to(device)
                    
                    pred_vector = model(volume)
                    loss, breakdown = compute_losses(pred_vector, target_vector, loss_type=config.loss_type)
                    
                    val_losses.append(loss.item())
                    val_maes.append(breakdown["mae"].item())
                    val_r2s.append(breakdown["r2"].item())
                    
            avg_val_loss = np.mean(val_losses)
            avg_val_mae = np.mean(val_maes)
            avg_val_r2 = np.mean(val_r2s)
            
            with open(log_val, "a") as fh:
                fh.write(f"epoch_{epoch+1}_loss={avg_val_loss:.6f}_mae={avg_val_mae:.6f}_r2={avg_val_r2:.6f}\n")
                
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(exp_dir, "ckpts", "model", "best_model.ckpt"))
            else:
                no_improve += 1
                
            if config.save_model:
                torch.save(model.state_dict(), os.path.join(exp_dir, "ckpts", "model", f"model_epoch{epoch+1}.ckpt"))
                
            if no_improve >= config.patience:
                print(f"Early stopping activado en la época {epoch+1}")
                break