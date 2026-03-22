import torch
import numpy as np


class EarlyStopping:


    def __init__(self, patience=5, verbose=True, delta=0.0, path="best_model.pth"):

        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        # Inizializzazione
        self.counter = 0  # Contatore epoche senza miglioramento
        self.best_loss = None  # Miglior loss visto finora
        self.early_stop = False  # Flag per interrompere
        self.best_epoch = 0  # Epoca del miglior modello

    def __call__(self, val_loss, model, epoch):


        # Prima volta: salva come miglior loss
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.best_epoch = epoch

        # Controlla se c'è miglioramento (loss diminuisce)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset contatore
            self.save_checkpoint(model)
            self.best_epoch = epoch

            if self.verbose:
                print(f"✓ Validation loss migliorata: {val_loss:.4f}")

        else:
            # Nessun miglioramento
            self.counter += 1

            if self.verbose:
                print(f"✗ Nessun miglioramento per {self.counter}/{self.patience} epoche")

            # Controlla se è il momento di fermarsi
            if self.counter >= self.patience:
                self.early_stop = True

                if self.verbose:
                    print("\n EARLY STOPPING ATTIVATO!")
                    print(f"   Nessun miglioramento per {self.patience} epoche consecutive")
                    print(f"   Miglior epoca: {self.best_epoch}")
                    print(f"   Miglior loss: {self.best_loss:.4f}")

    def save_checkpoint(self, model):
        """Salva lo stato del modello."""
        torch.save(model.state_dict(), self.path)

    def load_best_model(self, model):
        """Carica il miglior modello salvato."""
        device = next(model.parameters()).device
        state_dict = torch.load(self.path, map_location=device)
        model.load_state_dict(state_dict)

        if self.verbose:
            print(f"Best model caricato da '{self.path}' (epoca {self.best_epoch})")

        return model
