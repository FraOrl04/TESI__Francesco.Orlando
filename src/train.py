import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from .config import DEVICE

scaler = GradScaler()

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for data, targets in tqdm(loader, desc="Training", leave=False):
        data = data.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision solo se siamo su GPU
        with autocast(enabled=(DEVICE == "cuda")):
            outputs = model(data)
            loss = criterion(outputs, targets)

        if DEVICE == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * data.size(0)

    return total_loss / len(loader.dataset)
