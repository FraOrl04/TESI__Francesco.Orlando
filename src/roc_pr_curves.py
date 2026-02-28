# src/roc_pr_curves.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from .config import DEVICE

def get_logits_and_labels(model, dataloader, num_classes):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            outputs = model(images)  # (B, C)
            all_logits.append(outputs.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return logits, labels


def plot_roc_curves(model, dataloader, num_classes, class_names=None, out_path="roc_curves.png"):
    logits, labels = get_logits_and_labels(model, dataloader, num_classes)
    y_true = np.eye(num_classes)[labels]  # one-hot
    y_score = logits  # se hai softmax, puoi applicarlo, ma per ROC non è obbligatorio

    plt.figure(figsize=(8, 8))
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, c], y_score[:, c])
        roc_auc = auc(fpr, tpr)
        name = class_names[c] if class_names is not None else f"Class {c}"
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve per classe")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pr_curves(model, dataloader, num_classes, class_names=None, out_path="pr_curves.png"):
    logits, labels = get_logits_and_labels(model, dataloader, num_classes)
    y_true = np.eye(num_classes)[labels]
    y_score = logits

    plt.figure(figsize=(8, 8))
    for c in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, c], y_score[:, c])
        ap = average_precision_score(y_true[:, c], y_score[:, c])
        name = class_names[c] if class_names is not None else f"Class {c}"
        plt.plot(recall, precision, label=f"{name} (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curve per classe")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
