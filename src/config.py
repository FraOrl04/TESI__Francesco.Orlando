import torch

# Dispositivo: usa automaticamente la GPU se disponibile
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Classi del dataset Galaxy10 DECals da usare
# Modifica questa lista per scegliere quali classi allenare
# Esempi:
# [1, 5, 8] → merging, barred spiral, edge-on
# [1, 2, 6] → merging, round smooth, unbarred tight spiral
# [0,1,2,3,4,5,6,7,8,9] → tutte le classi
SELECTED_CLASSES = [1, 5, 8]

# Numero di epoche
EPOCHS = 20   # <-- MODIFICA QUI

# Iperparametri
BATCH_SIZE = 32
LR = 1e-3

# Dimensione immagini (256 consigliato, 128 più veloce)
IMG_SIZE = 256

# Su Windows lascia 0, altrimenti può dare errori
NUM_WORKERS = 0
