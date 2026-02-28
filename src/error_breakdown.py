# src/error_breakdown.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from .config import DEVICE

def compute_predictions(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_labels.append(labels.cpu())
            all_preds.append(preds)

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    return all_labels, all_preds


def plot_error_breakdown(model, dataloader, class_names=None, out_path="error_breakdown.png"):
    y_true, y_pred = compute_predictions(model, dataloader)
    num_classes = len(np.unique(y_true))

    errors_per_class = []
    counts_per_class = []

    for c in range(num_classes):
        mask = y_true == c
        counts = mask.sum()
        errors = (y_pred[mask] != y_true[mask]).sum()
        counts_per_class.append(counts)
        errors_per_class.append(errors / counts if counts > 0 else 0.0)

    x = np.arange(num_classes)
    plt.figure(figsize=(8, 5))
    plt.bar(x, errors_per_class)
    if class_names is not None:
        plt.xticks(x, class_names, rotation=45, ha="right")
    else:
        plt.xticks(x, [f"C{c}" for c in x])
    plt.ylabel("Error rate")
    plt.title("Error breakdown per classe")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
