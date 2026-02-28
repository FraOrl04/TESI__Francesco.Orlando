import matplotlib.pyplot as plt
from history import TrainingHistory

def plot_metrics(history_path="training_history.npz", out_prefix="plot"):
    hist = TrainingHistory.load(history_path)

    # Loss
    plt.figure(figsize=(8,5))
    plt.plot(hist.train_loss, label="Train Loss")
    plt.plot(hist.val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_prefix}_loss.png", dpi=200)

    # Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(hist.val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_prefix}_accuracy.png", dpi=200)
