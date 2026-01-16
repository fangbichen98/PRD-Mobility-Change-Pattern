"""
Plot training curves from the completed training
"""
import matplotlib.pyplot as plt
import re

def parse_training_log(log_file):
    """Parse training log and extract metrics"""

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    epochs = []
    train_losses = []
    train_accs = []
    train_f1s = []
    val_losses = []
    val_accs = []
    val_f1s = []

    lines = content.split('\n')
    current_epoch = None

    for line in lines:
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        train_match = re.search(r'Train - Loss: ([\d.]+), Acc: ([\d.]+), F1: ([\d.]+)', line)
        if train_match and current_epoch:
            train_losses.append(float(train_match.group(1)))
            train_accs.append(float(train_match.group(2)))
            train_f1s.append(float(train_match.group(3)))

        val_match = re.search(r'Val   - Loss: ([\d.]+), Acc: ([\d.]+), F1: ([\d.]+)', line)
        if val_match and current_epoch:
            epochs.append(current_epoch)
            val_losses.append(float(val_match.group(1)))
            val_accs.append(float(val_match.group(2)))
            val_f1s.append(float(val_match.group(3)))

    return epochs, train_losses, train_accs, train_f1s, val_losses, val_accs, val_f1s

def plot_training_curves(log_file, output_file):
    """Plot training curves"""

    epochs, train_losses, train_accs, train_f1s, val_losses, val_accs, val_f1s = parse_training_log(log_file)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Mark best epoch
    best_epoch_idx = val_accs.index(max(val_accs))
    axes[0].axvline(x=epochs[best_epoch_idx], color='g', linestyle='--', alpha=0.5, label=f'Best (Epoch {epochs[best_epoch_idx]})')
    axes[0].legend(fontsize=10)

    # Plot 2: Accuracy
    axes[1].plot(epochs, [acc*100 for acc in train_accs], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, [acc*100 for acc in val_accs], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=epochs[best_epoch_idx], color='g', linestyle='--', alpha=0.5, label=f'Best (Epoch {epochs[best_epoch_idx]})')
    axes[1].legend(fontsize=10)

    # Plot 3: F1 Score
    axes[2].plot(epochs, train_f1s, 'b-', label='Train F1', linewidth=2)
    axes[2].plot(epochs, val_f1s, 'r-', label='Val F1', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].axvline(x=epochs[best_epoch_idx], color='g', linestyle='--', alpha=0.5, label=f'Best (Epoch {epochs[best_epoch_idx]})')
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Total Epochs: {len(epochs)}")
    print(f"Best Epoch: {epochs[best_epoch_idx]}")
    print(f"Best Val Accuracy: {val_accs[best_epoch_idx]:.4f} ({val_accs[best_epoch_idx]*100:.2f}%)")
    print(f"Best Val F1 Score: {val_f1s[best_epoch_idx]:.4f}")
    print(f"Final Train Accuracy: {train_accs[-1]:.4f} ({train_accs[-1]*100:.2f}%)")
    print(f"Final Val Accuracy: {val_accs[-1]:.4f} ({val_accs[-1]*100:.2f}%)")
    print("="*80)

if __name__ == "__main__":
    log_file = "outputs/training_full_50epochs.log"
    output_file = "outputs/training_curves.jpg"

    plot_training_curves(log_file, output_file)
