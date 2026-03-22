import torch
from torch import nn, optim

from src.dataset import load_filtered_dataset
from src.model import GalaxyCNN
from src.utils import split_dataset, make_loader
from src.train import train_one_epoch
from src.evaluate import evaluate
from src.config import DEVICE, LR, EPOCHS, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_DELTA
from src.config import USE_ADAPTIVE_EARLY_STOPPING, ADAPTIVE_MODE, ADAPTIVE_MIN_PATIENCE, ADAPTIVE_MAX_PATIENCE

# Moduli
from src.history import TrainingHistory
from src.early_stopping import EarlyStopping
from src.early_stopping_adaptive import AdaptiveEarlyStopping
from src.plot_metrics import plot_metrics
from src.confusion import compute_confusion_matrix
from src.classification_report_gen import generate_classification_report
from src.visualize_filters import visualize_filters, visualize_feature_maps
from src.show_examples import show_examples
from src.gradcam import generate_gradcam_examples
from src.tsne_plot import plot_tsne
from src.error_breakdown import plot_error_breakdown
from src.roc_pr_curves import plot_roc_curves, plot_pr_curves
from src.dataset_analysis import plot_class_distribution
from src.config import SELECTED_CLASSES


def main():
    print(f"Using device: {DEVICE}")

    num_classes = len(SELECTED_CLASSES)

    # Dataset
    dataset = load_filtered_dataset()
    train_ds, val_ds, test_ds = split_dataset(dataset)

    print(f"Train set: {len(train_ds)} immagini")
    print(f"Validation set: {len(val_ds)} immagini")
    print(f"Test set: {len(test_ds)} immagini")

    train_loader = make_loader(train_ds)
    val_loader = make_loader(val_ds)
    test_loader = make_loader(test_ds, shuffle=False)

    # Modello
    model = GalaxyCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Early stopping
    if USE_ADAPTIVE_EARLY_STOPPING:
        early_stopping = AdaptiveEarlyStopping(
            mode=ADAPTIVE_MODE,
            verbose=True,
            delta=EARLY_STOPPING_DELTA,
            path="best_model_decals.pth",
            min_patience=ADAPTIVE_MIN_PATIENCE,
            max_patience=ADAPTIVE_MAX_PATIENCE
        )
        print(f" Early Stopping ADATTIVO (mode={ADAPTIVE_MODE})")
    else:
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            delta=EARLY_STOPPING_DELTA,
            verbose=True,
            path="best_model_decals.pth"
        )
        print(f" Early Stopping CLASSICO (patience={EARLY_STOPPING_PATIENCE})")

    history = TrainingHistory()
    stopped_early = False

    # ================= TRAIN =================
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_acc:.2f}%")

        history.update(train_loss, val_loss, val_acc)

        # Early stopping step
        early_stopping(val_loss, model, epoch + 1)

        if early_stopping.early_stop:
            stopped_early = True
            print(f"\n Allenamento interrotto all'epoca {epoch + 1}")

            history.save("training_history.npz")

            print("\nGenerazione grafici (early stopping)...")
            plot_metrics("training_history.npz", out_prefix="galaxy10")

            break

    # ================= POST TRAIN =================

    if not stopped_early:
        history.save("training_history.npz")

        print("\nGenerazione grafici...")
        plot_metrics("training_history.npz", out_prefix="galaxy10")

    # Carica miglior modello
    early_stopping.load_best_model(model)

    # Test
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    num_classes = model.fc2.out_features

    # ================= ANALISI =================

    print("\nGenerazione confusion matrix...")
    compute_confusion_matrix(model, test_loader, out_path="confusion_matrix.png")

    print("Generazione classification report...")
    generate_classification_report(model, test_loader, out_path="classification_report.txt")

    print("Generazione filtri...")
    visualize_filters(model, out_prefix="filters")

    print("Generazione feature maps...")
    example_img, _ = next(iter(test_loader))
    example_img = example_img[0].to(DEVICE)
    visualize_feature_maps(model, example_img, out_prefix="featuremap")

    print("Generazione esempi...")
    show_examples(model, test_loader, out_prefix="examples")

    print("Generazione GradCAM...")
    generate_gradcam_examples(
        model,
        test_loader,
        target_layer=model.conv4,
        out_prefix="gradcam",
        num_examples=8
    )

    print("Generazione t-SNE...")
    plot_tsne(model, test_loader, feature_layer=model.fc1, out_path="tsne_plot.png")

    print("Generazione error breakdown...")
    plot_error_breakdown(model, test_loader, out_path="error_breakdown.png")

    print("Generazione ROC curves...")
    plot_roc_curves(model, test_loader, num_classes=num_classes, out_path="roc_curves.png")

    print("Generazione PR curves...")
    plot_pr_curves(model, test_loader, num_classes=num_classes, out_path="pr_curves.png")

    print("Distribuzione classi...")
    plot_class_distribution(test_ds, out_path="class_distribution.png")

    print(f"\n Miglior epoca: {early_stopping.best_epoch}")
    print("\nTutto completato!")


if __name__ == "__main__":
    main()
