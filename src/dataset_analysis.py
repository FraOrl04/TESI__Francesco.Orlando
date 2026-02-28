# src/dataset_analysis.py
import numpy as np
import matplotlib.pyplot as plt

def plot_class_distribution(dataset, class_names=None, out_path="class_distribution.png"):
    labels = [dataset[i][1] for i in range(len(dataset))]
    labels = np.array(labels)
    num_classes = len(np.unique(labels))

    counts = [(labels == c).sum() for c in range(num_classes)]
    x = np.arange(num_classes)

    plt.figure(figsize=(8, 5))
    plt.bar(x, counts)
    if class_names is not None:
        plt.xticks(x, class_names, rotation=45, ha="right")
    else:
        plt.xticks(x, [f"C{c}" for c in x])
    plt.ylabel("Numero di esempi")
    plt.title("Distribuzione delle classi nel dataset")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
