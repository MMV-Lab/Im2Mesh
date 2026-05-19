import os
import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

def parse_epoch_data(filepath):
    epochs = []
    losses = []
    if not os.path.exists(filepath):
        return epochs, losses
    with open(filepath, "r") as f:
        for line in f:
            match = re.search(r"epoch_(\d+)_([\d.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return epochs, losses

def parse_step_data(filepath):
    steps = []
    totals = []
    components = {}
    if not os.path.exists(filepath):
        return steps, totals, components
    
    with open(filepath, "r") as f:
        content = f.read()
        
    content = re.sub(r"/", "", content)
    content = content.replace("\n", " ")
    
    logs = re.findall(r"E(\d+)-(\d+)\s+total=([\d.]+)(.*?)(?=E\d+-\d+\s+total=|$)", content)
    for log in logs:
        epoch = int(log[0])
        step = int(log[1])
        total = float(log[2])
        steps.append(f"E{epoch}-{step}")
        totals.append(total)
        
        comps_str = log[3]
        pairs = re.findall(r"([a-zA-Z_]+)\s+([\d.eE+-]+)", comps_str)
        for name, val in pairs:
            if name not in ["dt", "s", "lr"]:
                if name not in components:
                    components[name] = []
                components[name].append(float(val))
                
    return steps, totals, components

def plot_simple(x, y, title, xlabel, ylabel, output_path,dpis):
    if not x or not y:
        return
    plt.figure(figsize=(14, 9))
    plt.plot(x, y, color="blue", linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpis)
    plt.close()

def plot_steps(steps, totals, components, output_folder,dpis):
    if not steps or not totals:
        return
        
    x = list(range(len(steps)))
    all_values = totals + [val for sublist in components.values() for val in sublist]
    max_y_total = max(all_values) * 1.05 if all_values else 1.0
    colors = ["red", "green", "purple", "orange", "brown", "cyan", "magenta", "lime", "pink", "teal"]
    
    plt.figure(figsize=(14, 9))
    plt.plot(x, totals, label="Total Loss", color="blue", linewidth=2)
    for i, (label, values) in enumerate(components.items()):
        color = colors[i % len(colors)]
        plt.plot(x, values, label=label, color=color, linewidth=1, linestyle="--")
    
    plt.xlabel("Step Index")
    plt.ylabel("Loss Value")
    plt.title("Training Total Loss and Components over Steps")
    plt.ylim(0, max_y_total)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "train_step_total_and_components.png"), dpi=dpis)
    plt.close()

    for i, (label, values) in enumerate(components.items()):
        plt.figure(figsize=(14, 9))
        color = colors[i % len(colors)]
        plt.plot(x, values, label=label, color=color, linewidth=2)
        plt.xlabel("Step Index")
        plt.ylabel("Loss Value")
        plt.title(f"Training {label} over Steps")
        
        comp_max_y = max(values) * 1.05 if values else 1.0
        plt.ylim(0, comp_max_y)
        
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"train_step_{label}.png"), dpi=dpis)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot training and validation losses from text files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the log txt files.")
    parser.add_argument("--dpis", type=int,default=300 , help="dpi for plot generation.")
    parser.add_argument('--output_dir', type=str, required=False, default=None, help="Directory to save the generated PNG plots.")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Logs folder not found in -> {args.input_dir}")
    
    if args.output_dir is None:
        out = Path(args.input_dir).parent / 'Generated_plots'
    else:
        out = args.output_dir

    
    os.makedirs(out, exist_ok=True)

    train_epoch_file = os.path.join(args.input_dir, "train_epoch.txt")
    val_epoch_file = os.path.join(args.input_dir, "val_epoch.txt")
    train_step_file = os.path.join(args.input_dir, "train_step.txt")

    train_ep_x, train_ep_y = parse_epoch_data(train_epoch_file)
    plot_simple(train_ep_x, train_ep_y, "Training Loss over Epochs", "Epoch", "Total Loss", os.path.join(out, "train_epoch_loss.png"),args.dpis)

    val_ep_x, val_ep_y = parse_epoch_data(val_epoch_file)
    plot_simple(val_ep_x, val_ep_y, "Validation Loss over Epochs", "Epoch", "Total Loss", os.path.join(out, "val_epoch_loss.png"),args.dpis)

    steps, totals, components = parse_step_data(train_step_file)
    plot_steps(steps, totals, components, out,args.dpis)
    
    print(f"Plot generated and saved in -> { out }")

if __name__ == "__main__":
    main()