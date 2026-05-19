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
    batch_size: int
    n_start_filters: int
    latent_dim: int
    target_dim: int
    pretrain_file: Optional[str]
    lr: float
    warmup_ratio: float
    warmup_lr_start: float
    n_epoch: int
    accumulate_grad_batches: int
    loss_type: str  # 'mse', 'mae', o 'weighted_mse'
    report_training_loss: bool
    ckpts_interval: int
    save_model: bool
    patience: int

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--optional_id", type=str, default=None)
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_start_filters", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=512)
    p.add_argument("--target_dim", type=int, default=2503)
    p.add_argument("--pretrain_file", type=str, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--warmup_lr_start", type=float, default=1e-6)
    p.add_argument("--n_epoch", type=int, default=3000)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--loss_type", type=str, default="weighted_mse", choices=["mse", "mae", "weighted_mse"])
    p.add_argument("--report_training_loss", action="store_true", default=True)
    p.add_argument("--no_report_training_loss", dest="report_training_loss", action="store_false")
    p.add_argument("--ckpts_interval", type=int, default=10)
    p.add_argument("--save_model", action="store_true", default=True)
    p.add_argument("--no_save_model", dest="save_model", action="store_false")
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
    if args.accumulate_grad_batches < 1:
        parser.error(f"--accumulate_grad_batches must be >= 1")
    return Config(
        optional_id=args.optional_id,
        data_path=args.data_path,
        val_split=args.val_split,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        n_start_filters=args.n_start_filters,
        latent_dim=args.latent_dim,
        target_dim=args.target_dim,
        pretrain_file=args.pretrain_file,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        warmup_lr_start=args.warmup_lr_start,
        n_epoch=args.n_epoch,
        accumulate_grad_batches=args.accumulate_grad_batches,
        loss_type=args.loss_type,
        report_training_loss=args.report_training_loss,
        ckpts_interval=args.ckpts_interval,
        save_model=args.save_model,
        patience=args.patience,
    )