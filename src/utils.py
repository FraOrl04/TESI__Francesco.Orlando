from torch.utils.data import random_split, DataLoader
from .config import BATCH_SIZE, NUM_WORKERS, DEVICE

def split_dataset(dataset):
    total = len(dataset)
    train_len = int(0.7 * total)
    val_len = int(0.15 * total)
    test_len = total - train_len - val_len

    return random_split(dataset, [train_len, val_len, test_len])

def make_loader(ds, shuffle=True):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False
    )
