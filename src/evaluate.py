import torch
from tqdm import tqdm
from .config import DEVICE

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Evaluating", leave=False):
            data = data.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            outputs = model(data)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * data.size(0)

            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(loader.dataset)
    acc = 100.0 * correct / total
    return avg_loss, acc
