# src/tsne_plot.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from .config import DEVICE

def extract_features(model, dataloader, feature_layer):
    model.eval()
    feats = []
    labels_list = []

    def hook(module, inp, out):
        feats.append(out.detach().cpu())

    handle = feature_layer.register_forward_hook(hook)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            _ = model(images)
            labels_list.append(labels.cpu())

    handle.remove()

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels_list, dim=0)

    feats = feats.view(feats.size(0), -1)  # flatten
    return feats.numpy(), labels.numpy()


def plot_tsne(model, dataloader, feature_layer, out_path="tsne_features.png", perplexity=30, n_samples=2000):
    feats, labels = extract_features(model, dataloader, feature_layer)

    if feats.shape[0] > n_samples:
        idx = np.random.choice(feats.shape[0], n_samples, replace=False)
        feats = feats[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, perplexity=perplexity, init="random", learning_rate="auto")
    emb = tsne.fit_transform(feats)

    plt.figure(figsize=(8, 8))
    num_classes = len(np.unique(labels))
    for c in range(num_classes):
        mask = labels == c
        plt.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.7, label=f"Class {c}")

    plt.legend(markerscale=3, fontsize=8)
    plt.title("t-SNE delle feature (penultimo layer)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
