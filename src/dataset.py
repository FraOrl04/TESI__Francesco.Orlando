import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from astroNN.datasets import load_galaxy10
from PIL import Image
from .config import SELECTED_CLASSES, IMG_SIZE
class GalaxyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.uint8)  # (H, W, 3)
        img = Image.fromarray(img)
        label = int(self.labels[idx])

        if self.transform:
            img = self.transform(img)

        return img, label


def load_filtered_dataset():
    # Carica Galaxy10 DECals
    images, labels = load_galaxy10()

    # Filtra solo le classi desiderate
    mask = np.isin(labels, SELECTED_CLASSES)
    images = images[mask]
    labels = labels[mask]

    # Rimappa le etichette a 0,1,2
    class_to_idx = {c: i for i, c in enumerate(SELECTED_CLASSES)}
    labels = np.array([class_to_idx[int(l)] for l in labels])

    # Trasformazioni: resize + tensor + normalizzazione
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    return GalaxyDataset(images, labels, transform)
