import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from .config import DEVICE, SELECTED_CLASSES

def compute_confusion_matrix(model, loader, out_path="confusion_matrix.png"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(data)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_labels = [str(c) for c in SELECTED_CLASSES]
    plt.xticks(range(len(tick_labels)), tick_labels)
    plt.yticks(range(len(tick_labels)), tick_labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
