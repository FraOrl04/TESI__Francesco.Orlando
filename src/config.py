import torch

# Dispositivo: usa automaticamente la GPU se disponibile
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Classi del dataset Galaxy10 DECals da usare

SELECTED_CLASSES = [1, 5, 8] #primo test con 3 classi

# Numero di epoche
#EPOCHS = 10  epoche per 3 classi
EPOCHS = 120 #epoche per 5 classi

# Iperparametri
BATCH_SIZE = 32
LR = 1e-3

# Dimensione immagini (256 consigliato, 128 più veloce)
IMG_SIZE = 128

# Su Windows lascia 0, altrimenti può dare errori
NUM_WORKERS = 0

# Early Stopping
EARLY_STOPPING_PATIENCE = 10  # Numero di epoche senza miglioramento prima di fermarsi
EARLY_STOPPING_DELTA = 0.01  # Miglioramento minimo della validation loss (0 = qualsiasi miglioramento)

# Early Stopping Adattivo (NUOVO - PIÙ OGGETTIVO!)
# Modalità: "auto", "variance", "slope", "stability", oppure None per disattivare
USE_ADAPTIVE_EARLY_STOPPING = True  # Attiva l'early stopping adattivo
ADAPTIVE_MODE = "auto"  # Quale algoritmo usare
ADAPTIVE_MIN_PATIENCE = 5   # Patience minimo
ADAPTIVE_MAX_PATIENCE =20  # Patience massimo

