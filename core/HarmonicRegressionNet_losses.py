import torch
import torch.nn as nn
from typing import Dict, Tuple

def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, decay_rate: float = 0.02) -> torch.Tensor:
    n_dim = pred.shape[-1]
    indices = torch.arange(n_dim, dtype=torch.float32, device=pred.device)
    weights = torch.exp(-decay_rate * indices)
    weights = weights / weights.sum() * n_dim 
    
    sq_error = (pred - target).pow(2)
    weighted_error = sq_error * weights.unsqueeze(0)
    return weighted_error.mean()

def compute_regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:

    with torch.no_grad():
        mse = nn.functional.mse_loss(pred, target).item()
        mae = nn.functional.l1_loss(pred, target).item()
        
        cos_sim = nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
        
        target_mean = torch.mean(target, dim=0, keepdim=True)
        ss_tot = torch.sum((target - target_mean).pow(2))
        ss_res = torch.sum((target - pred).pow(2))
        r2_score = (1.0 - (ss_res / (ss_tot + 1e-8))).item()
        
    return {
        "mse": mse,
        "mae": mae,
        "cosine_similarity": cos_sim,
        "r2": r2_score
    }

def compute_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "weighted_mse"
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    if loss_type == "mse":
        main_loss = nn.functional.mse_loss(pred, target)
    elif loss_type == "mae":
        main_loss = nn.functional.l1_loss(pred, target)
    elif loss_type == "weighted_mse":
        main_loss = weighted_mse_loss(pred, target)
    else:
        main_loss = nn.functional.mse_loss(pred, target)
        
    metrics = compute_regression_metrics(pred, target)
    
    breakdown = {k: torch.tensor(v, device=pred.device) for k, v in metrics.items()}
    
    return main_loss, breakdown