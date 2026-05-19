from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    optional_id: Optional[str]
    data_path: str
    val_split: float
    num_workers: int
    n_start_filters: int
    latent_dim: int
    max_sh_degree: int
    sphere_subdivisions: int
    refiner_steps: int
    refiner_layers: int
    refiner_hidden: int
    pretrain_file: Optional[str]
    lr: float
    warmup_ratio: float
    warmup_lr_start: float
    n_epoch: int
    n_smooth: int
    lambd: float
    accumulate_grad_batches: int
    chamfer_scale: float
    normal_consistency_scale: float
    edge_length_scale: float
    laplacian_smoothing_scale: float
    willmore_scale: float
    mode_energy_scale: float
    report_training_loss: bool
    ckpts_interval: int
    save_model: bool
    save_mesh_eval: bool
    mesh_interval: int
    patience: int

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--optional_id", type=str, default=None)
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_start_filters", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=512)
    p.add_argument("--max_sh_degree", type=int, default=8)
    p.add_argument("--sphere_subdivisions", type=int, default=4)
    p.add_argument("--refiner_steps", type=int, default=3)
    p.add_argument("--refiner_layers", type=int, default=3)
    p.add_argument("--refiner_hidden", type=int, default=64)
    p.add_argument("--pretrain_file", type=str, default=None)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--warmup_lr_start", type=float, default=1e-6)
    p.add_argument("--n_epoch", type=int, default=3000)
    p.add_argument("--n_smooth", type=int, default=1)
    p.add_argument("--lambd", type=float, default=0.5)
    p.add_argument("--accumulate_grad_batches", type=int, default=6)
    p.add_argument("--chamfer_scale", type=float, default=1.0)
    p.add_argument("--normal_consistency_scale", type=float, default=0.05)
    p.add_argument("--edge_length_scale", type=float, default=0.3)
    p.add_argument("--laplacian_smoothing_scale", type=float, default=0.3)
    p.add_argument("--willmore_scale", type=float, default=5e-05)
    p.add_argument("--mode_energy_scale", type=float, default=0.0001)
    p.add_argument("--report_training_loss", action="store_true", default=True)
    p.add_argument("--no_report_training_loss", dest="report_training_loss", action="store_false")
    p.add_argument("--ckpts_interval", type=int, default=10)
    p.add_argument("--save_model", action="store_true", default=True)
    p.add_argument("--no_save_model", dest="save_model", action="store_false")
    p.add_argument("--save_mesh_eval", action="store_true", default=False)
    p.add_argument("--no_save_mesh_eval", dest="save_mesh_eval", action="store_false")
    p.add_argument("--mesh_interval", type=int, default=5)
    p.add_argument("--patience", type=int, default=90)
    return p

def load_config() -> Config:
    parser = _build_parser()
    args = parser.parse_args()
    if not (0.0 < args.val_split < 1.0):
        parser.error(f"--val_split must be in (0, 1), got {args.val_split}")
    if args.lr <= 0:
        parser.error(f"--lr must be positive, got {args.lr}")
    if args.warmup_lr_start >= args.lr:
        parser.error(f"--warmup_lr_start must be smaller than --lr")
    if not (0.0 < args.lambd < 1.0):
        parser.error(f"--lambd must be in (0, 1), got {args.lambd}")
    if args.accumulate_grad_batches < 1:
        parser.error(f"--accumulate_grad_batches must be >= 1")
    return Config(
        optional_id=args.optional_id,
        data_path=args.data_path,
        val_split=args.val_split,
        num_workers=args.num_workers,
        n_start_filters=args.n_start_filters,
        latent_dim=args.latent_dim,
        max_sh_degree=args.max_sh_degree,
        sphere_subdivisions=args.sphere_subdivisions,
        refiner_steps=args.refiner_steps,
        refiner_layers=args.refiner_layers,
        refiner_hidden=args.refiner_hidden,
        pretrain_file=args.pretrain_file,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        warmup_lr_start=args.warmup_lr_start,
        n_epoch=args.n_epoch,
        n_smooth=args.n_smooth,
        lambd=args.lambd,
        accumulate_grad_batches=args.accumulate_grad_batches,
        chamfer_scale=args.chamfer_scale,
        normal_consistency_scale=args.normal_consistency_scale,
        edge_length_scale=args.edge_length_scale,
        laplacian_smoothing_scale=args.laplacian_smoothing_scale,
        willmore_scale=args.willmore_scale,
        mode_energy_scale=args.mode_energy_scale,
        report_training_loss=args.report_training_loss,
        ckpts_interval=args.ckpts_interval,
        save_model=args.save_model,
        save_mesh_eval=args.save_mesh_eval,
        mesh_interval=args.mesh_interval,
        patience=args.patience,
    )