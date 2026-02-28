import torch
from torch import nn, optim

from src.dataset import load_filtered_dataset
from src.model import GalaxyCNN
from src.utils import split_dataset, make_loader
from src.train import train_one_epoch
from src.evaluate import evaluate
from src.config import DEVICE, LR, EPOCHS

# Import dei nuovi moduli
from src.history import TrainingHistory
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


def main():
    print(f"Using device: {DEVICE}")

    # Carica dataset filtrato (Galaxy10 DECals, classi scelte)
    dataset = load_filtered_dataset()
    train_ds, val_ds, test_ds = split_dataset(dataset)

    print(f"Train set: {len(train_ds)} immagini")
    print(f"Validation set: {len(val_ds)} immagini")
    print(f"Test set: {len(test_ds)} immagini")

    train_loader = make_loader(train_ds)
    val_loader = make_loader(val_ds)
    test_loader = make_loader(test_ds, shuffle=False)

    # Modello, loss, optimizer
    model = GalaxyCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    history = TrainingHistory()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_acc:.2f}%")

        # aggiorna history
        history.update(train_loss, val_loss, val_acc)

        # salva modello migliore
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_decals.pth")
            print(">> Nuovo modello migliore salvato.")

    # salva la history
    history.save("training_history.npz")

    # Test finale
    model.load_state_dict(torch.load("best_model_decals.pth", map_location=DEVICE))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # Ottieni il numero di classi dal modello
    num_classes = model.fc2.out_features

    # Genera grafici
    print("\nGenerazione grafici...")
    plot_metrics("training_history.npz", out_prefix="galaxy10")

    # Confusion matrix
    print("Generazione confusion matrix...")
    compute_confusion_matrix(model, test_loader, out_path="confusion_matrix.png")

    # Classification report
    print("Generazione classification report...")
    generate_classification_report(model, test_loader, out_path="classification_report.txt")

    # Visualizzazione filtri
    print("Generazione immagini dei filtri convoluzionali...")
    visualize_filters(model, out_prefix="filters")

    # Visualizzazione feature maps
    print("Generazione feature maps...")
    example_img, _ = next(iter(test_loader))
    example_img = example_img[0]  # prima immagine del batch
    visualize_feature_maps(model, example_img, out_prefix="featuremap")

    # Esempi corretti e sbagliati
    print("Generazione esempi corretti e sbagliati...")
    show_examples(model, test_loader, out_prefix="examples")

    # GradCAM
    print("Generazione GradCAM examples...")
    generate_gradcam_examples(model, test_loader, target_layer=model.conv4, out_prefix="gradcam", num_examples=8)

    # T-SNE plot
    print("Generazione t-SNE plot...")
    plot_tsne(model, test_loader, feature_layer=model.fc1, out_path="tsne_plot.png")

    # Error breakdown
    print("Generazione error breakdown plot...")
    plot_error_breakdown(model, test_loader, out_path="error_breakdown.png")

    # ROC curves
    print("Generazione ROC curves...")
    plot_roc_curves(model, test_loader, num_classes=num_classes, out_path="roc_curves.png")

    # Precision-Recall curves
    print("Generazione Precision-Recall curves...")
    plot_pr_curves(model, test_loader, num_classes=num_classes, out_path="pr_curves.png")

    # Class distribution (train, val e test separatamente)
    print("Generazione class distribution plot...")
    plot_class_distribution(test_ds, out_path="class_distribution.png")

    print("\nTutto completato! File generati:")
    print("- galaxy10_loss.png")
    print("- galaxy10_accuracy.png")
    print("- confusion_matrix.png")
    print("- classification_report.txt")
    print("- filters_layerX.png")
    print("- featuremap_layerX.png")
    print("- examples_correct.png")
    print("- examples_wrong.png")
    print("- gradcam_*.png")
    print("- tsne_plot.png")
    print("- error_breakdown.png")
    print("- roc_curves.png")
    print("- pr_curves.png")
    print("- class_distribution.png")
    print("- best_model_decals.pth")
    print("- training_history.npz")


if __name__ == "__main__":
    main()

