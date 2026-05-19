import os
import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

def parse_epoch_data(filepath):
    """Parsea el archivo train_epoch.txt"""
    epochs = []
    losses = []
    if not os.path.exists(filepath):
        return epochs, losses
    with open(filepath, "r") as f:
        for line in f:
            match = re.search(r"epoch_(\d+)_([\d.-]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return epochs, losses

def parse_val_epoch_data(filepath):

    epochs = []
    losses = []
    maes = []
    r2s = []
    if not os.path.exists(filepath):
        return epochs, losses, maes, r2s
    with open(filepath, "r") as f:
        for line in f:
            match = re.search(r"epoch_(\d+)_loss=([\d.-]+)_mae=([\d.-]+)_r2=([\d.-]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                maes.append(float(match.group(3)))
                r2s.append(float(match.group(4)))
    return epochs, losses, maes, r2s

def parse_step_data(filepath):
    """Parsea el archivo train_step.txt paso a paso de forma robusta"""
    steps = []
    total_losses = []
    mses = []
    maes = []
    cosine_sims = []
    r2s = []
    lrs = []
    
    if not os.path.exists(filepath):
        return steps, total_losses, mses, maes, cosine_sims, r2s, lrs
        
    with open(filepath, "r") as f:
        for line in f:
            # Expresión regular exacta para capturar el formato con pipes '|' y espacios intermedios
            pattern = r"E(\d+)-(\d+)\s+total_loss=([\d.-]+)\s*\|\s*mse=([\d.-]+)\s+mae=([\d.-]+)\s+cosine_similarity=([\d.-]+)\s+r2=([\d.-]+)\s*\|\s*dt=([\d.-]+)s\s*\|\s*lr=([\d.eE+-]+)"
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                step_val = int(match.group(2))
                steps.append(f"E{epoch}-{step_val}")
                total_losses.append(float(match.group(3)))
                mses.append(float(match.group(4)))
                maes.append(float(match.group(5)))
                cosine_sims.append(float(match.group(6)))
                r2s.append(float(match.group(7)))
                lrs.append(float(match.group(9)))
                
    return steps, total_losses, mses, maes, cosine_sims, r2s, lrs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation metrics from text files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the log txt files.")
    parser.add_argument("--dpis", type=int,default=300 , help="dpi for plot generation.")
    parser.add_argument('--output_dir', type=str, required=False, default=None, help="Directory to save the generated PNG plots.")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Logs folder not found in -> {args.input_dir}")
        exit(1)
    
    out = Path(args.output_dir) if args.output_dir is len else Path(args.input_dir).parent / 'Generated_plots'
    os.makedirs(out, exist_ok=True)

    train_epoch_file = os.path.join(args.input_dir, "train_epoch.txt")
    val_epoch_file = os.path.join(args.input_dir, "val_epoch.txt")
    train_step_file = os.path.join(args.input_dir, "train_step.txt")


    train_ep_x, train_ep_y = parse_epoch_data(train_epoch_file)
    if train_ep_x:
        plt.figure(figsize=(8, 5))
        plt.plot(train_ep_x, train_ep_y, color='teal', label='Train Loss (Epoch)')
        plt.title("Training Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Total Loss")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out, "train_epoch_loss.png"), dpi=args.dpis)
        plt.close()


    val_ep, val_loss, val_mae, val_r2 = parse_val_epoch_data(val_epoch_file)
    if val_ep:
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        

        axs[0].plot(val_ep, val_loss, color='crimson', marker='o', label='Val Loss')
        axs[0].set_ylabel("Weighted MSE Loss")
        axs[0].grid(True, linestyle='--', alpha=0.5)
        axs[0].legend()
        axs[0].set_title("Validation Metrics over Epochs")
        

        axs[1].plot(val_ep, val_mae, color='orange', marker='s', label='Val MAE')
        axs[1].set_ylabel("MAE")
        axs[1].grid(True, linestyle='--', alpha=0.5)
        axs[1].legend()
        
        
        axs[2].plot(val_ep, val_r2, color='purple', marker='^', label='Val R² Score')
        axs[2].set_ylabel("R² (Symlog Scale)")
        axs[2].set_xlabel("Epoch")
        axs[2].set_yscale('symlog') 
        axs[2].grid(True, linestyle='--', alpha=0.5)
        axs[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(out, "validation_epoch_metrics.png"), dpi=args.dpis)
        plt.close()


    steps, st_loss, st_mse, st_mae, st_cos, st_r2, st_lr = parse_step_data(train_step_file)
    if steps:

        x_indices = list(range(len(steps)))
        
        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        

        axs[0].plot(x_indices, st_loss, color='blue', alpha=0.7, label='Total Loss')
        axs[0].plot(x_indices, st_mse, color='green', alpha=0.5, label='Raw MSE')
        axs[0].set_ylabel("Loss / MSE")
        axs[0].grid(True, linestyle=':', alpha=0.5)
        axs[0].legend()
        axs[0].set_title("Step-by-Step Training Progress")
        
 
        axs[1].plot(x_indices, st_mae, color='darkorange', label='Step MAE')
        axs[1].set_ylabel("MAE")
        axs[1].grid(True, linestyle=':', alpha=0.5)
        axs[1].legend()
        

        axs[2].plot(x_indices, st_cos, color='darkcyan', label='Cosine Similarity')
        axs[2].set_ylabel("Cosine Sim")
        axs[2].set_ylim(-1.1, 1.1)
        axs[2].grid(True, linestyle=':', alpha=0.5)
        axs[2].legend()
        

        axs[3].plot(x_indices, st_lr, color='magenta', label='Learning Rate')
        axs[3].set_ylabel("LR")
        axs[3].set_xlabel("Training Step Entry")
        axs[3].set_yscale('log')
        axs[3].grid(True, linestyle=':', alpha=0.5)
        axs[3].legend()
        

        step_cadence = max(1, len(steps) // 10)
        plt.xticks(x_indices[::step_cadence], steps[::step_cadence], rotation=25, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out, "train_step_metrics.png"), dpi=args.dpis)
        plt.close()

    print(f"Plot generated and saved in ->  {out}")