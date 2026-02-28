import torch
import matplotlib.pyplot as plt
from .config import DEVICE, SELECTED_CLASSES

def show_examples(model, loader, out_prefix="examples"):
    model.eval()

    correct_imgs = []
    wrong_imgs = []

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(data)
            _, preds = outputs.max(1)

            for img, pred, true in zip(data, preds, labels):
                if pred == true and len(correct_imgs) < 6:
                    correct_imgs.append((img.cpu(), pred.item(), true.item()))
                elif pred != true and len(wrong_imgs) < 6:
                    wrong_imgs.append((img.cpu(), pred.item(), true.item()))

            if len(correct_imgs) >= 6 and len(wrong_imgs) >= 6:
                break

    # Salva immagini corrette
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, (img, pred, true) in zip(axes.flatten(), correct_imgs):
        img = img.permute(1, 2, 0)
        img = (img * 0.5 + 0.5).clamp(0, 1)
        ax.imshow(img)
        ax.set_title(f"Pred: {SELECTED_CLASSES[pred]} | True: {SELECTED_CLASSES[true]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_correct.png", dpi=200)
    plt.close()

    # Salva immagini sbagliate
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, (img, pred, true) in zip(axes.flatten(), wrong_imgs):
        img = img.permute(1, 2, 0)
        img = (img * 0.5 + 0.5).clamp(0, 1)
        ax.imshow(img)
        ax.set_title(f"Pred: {SELECTED_CLASSES[pred]} | True: {SELECTED_CLASSES[true]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_wrong.png", dpi=200)
    plt.close()
