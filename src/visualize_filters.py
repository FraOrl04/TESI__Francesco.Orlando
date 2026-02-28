import torch
import matplotlib.pyplot as plt
import numpy as np
from .config import DEVICE
from torchvision.utils import make_grid

def visualize_filters(model, out_prefix="filters"):
    """
    Salva i filtri (kernel) dei layer convoluzionali come immagini PNG.
    """
    conv_layers = [
        module for module in model.modules()
        if isinstance(module, torch.nn.Conv2d)
    ]

    for idx, layer in enumerate(conv_layers):
        weights = layer.weight.data.clone().cpu()

        # Se i filtri hanno 3 canali (RGB), prendiamo solo il primo
        if weights.shape[1] > 1:
            weights = weights[:, 0:1, :, :]

        # Normalizzazione globale
        w_min, w_max = weights.min(), weights.max()
        weights = (weights - w_min) / (w_max - w_min + 1e-8)

        grid = make_grid(weights, nrow=8, padding=1)
        npimg = grid.numpy().transpose((1, 2, 0))

        plt.figure(figsize=(10, 10))
        plt.imshow(npimg.squeeze(), cmap="viridis")
        plt.axis("off")
        plt.title(f"Filtri Layer Conv {idx}")
        plt.savefig(f"{out_prefix}_layer{idx}.png", dpi=200)
        plt.close()
def visualize_feature_maps(model, image, out_prefix="featuremap"):
    """
    Passa un'immagine attraverso la rete e salva le feature maps di ogni layer conv.
    """
    model.eval()
    conv_layers = []

    # Estrai i layer conv in ordine
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)

    x = image.unsqueeze(0).to(DEVICE)

    for idx, layer in enumerate(conv_layers):
        x = layer(x)
        x = torch.relu(x)

        fmap = x.detach().cpu()

        # Prendi solo le prime 16 feature map per non esagerare
        fmap = fmap[0, :16, :, :]

        grid = make_grid(fmap.unsqueeze(1), nrow=4, normalize=True, padding=1)
        npimg = grid.numpy().transpose((1, 2, 0))

        plt.figure(figsize=(8, 8))
        plt.imshow(npimg, cmap="viridis")
        plt.axis("off")
        plt.title(f"Feature Maps Layer {idx}")
        plt.savefig(f"{out_prefix}_layer{idx}.png", dpi=200)
        plt.close()
