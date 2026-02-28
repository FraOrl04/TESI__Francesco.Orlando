import torch
from sklearn.metrics import classification_report
from .config import DEVICE, SELECTED_CLASSES

def generate_classification_report(model, loader, out_path="classification_report.txt"):
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

    report = classification_report(
        all_labels,
        all_preds,
        target_names=[f"class_{c}" for c in SELECTED_CLASSES]
    )

    with open(out_path, "w") as f:
        f.write(report)

    print(report)
