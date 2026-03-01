import torch

# Dispositivo: usa automaticamente la GPU se disponibile
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Classi del dataset Galaxy10 DECals da usare

SELECTED_CLASSES = [1, 5, 8]

# Numero di epoche
EPOCHS = 10

# Iperparametri
BATCH_SIZE = 32
LR = 1e-3

# Dimensione immagini (256 consigliato, 128 più veloce)
IMG_SIZE = 256

# Su Windows lascia 0, altrimenti può dare errori
NUM_WORKERS = 0
