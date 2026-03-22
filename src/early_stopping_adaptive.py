import torch
import numpy as np


class AdaptiveEarlyStopping:
    """
    Early Stopping adattivo che calcola automaticamente la 'patience' 
    basandosi su metriche statistiche oggettive della validation loss.
    
    Questo è più scientifico rispetto a un semplice numero arbitrario.
    """
    
    def __init__(self, mode="auto", verbose=True, delta=0.0, path="best_model.pth", 
                 min_patience=3, max_patience=15):
        """
        Args:
            mode (str): Come calcolare la patience
                - "auto": Calcola automaticamente dall'entropia della loss
                - "variance": Basato sulla varianza della loss
                - "slope": Basato sulla pendenza della curva
                - "stability": Basato sulla stabilità (scarto quadratico medio)
                Default: "auto"
            verbose (bool): Se True, stampa messaggi informativi
            delta (float): Miglioramento minimo della loss
            path (str): Percorso dove salvare il modello
            min_patience (int): Patience minimo (default: 3)
            max_patience (int): Patience massimo (default: 15)
        """
        self.mode = mode
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.min_patience = min_patience
        self.max_patience = max_patience
        
        # Variabili di stato
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        
        # Per il calcolo adattivo
        self.loss_history = []
        self.patience = None  # Calcolato dinamicamente
        self.patience_calculated = False
        
    def calculate_patience(self):
        """
        Calcola la patience in modo oggettivo basandosi sui dati storici.
        
        Ritorna:
            int: Valore di patience calcolato
        """
        
        if len(self.loss_history) < 3:
            self.patience = self.min_patience
            if self.verbose:
                print(f" Dati insufficienti per calcolare patience. "
                      f"Usando valore minimo: {self.patience}")
            return self.patience
        
        loss_array = np.array(self.loss_history)
        
        if self.mode == "auto":
            patience = self._calculate_auto(loss_array)
        elif self.mode == "variance":
            patience = self._calculate_variance(loss_array)
        elif self.mode == "slope":
            patience = self._calculate_slope(loss_array)
        elif self.mode == "stability":
            patience = self._calculate_stability(loss_array)
        else:
            patience = self.min_patience
        
        # Constraina tra min e max
        self.patience = max(self.min_patience, min(int(patience), self.max_patience))
        
        if self.verbose:
            print(f"\nEARLY STOPPING ADATTIVO")
            print(f"   Mode: {self.mode}")
            print(f"   Patience calcolato: {self.patience} epoche")
            print(f"   (Range: {self.min_patience}-{self.max_patience})")
        
        self.patience_calculated = True
        return self.patience
    
    def _calculate_auto(self, loss_array):
        """
        Combina varianza e slope per una stima intelligente.
        
        Idea:
        - Se la loss è stabile (bassa varianza) → patience ALTA
        - Se la loss sta scendendo velocemente (alto slope negativo) → patience BASSA
        - Se la loss è caotica (alta varianza) → patience MEDIA
        """
        variance = np.var(loss_array)
        slope = np.polyfit(np.arange(len(loss_array)), loss_array, 1)[0]
        
        # Normalizza
        normalized_variance = min(variance / (np.mean(loss_array) ** 2 + 1e-5), 1.0)
        normalized_slope = min(abs(slope) / (np.mean(np.abs(loss_array)) + 1e-5), 1.0)
        
        # Logica:
        # - Se la curva è piatta (bassa varianza) e slope vicino a 0 → siamo nel plateau → patience ALTA
        # - Se la curva sta ancora scendendo (slope negativo) → continua a imparare → patience BASSA
        # - Se la curva è caotica (alta varianza) → incertezza → patience MEDIA
        
        if normalized_slope < 0.05 and normalized_variance < 0.2:
            # Siamo nel plateau stabile
            patience = 8 + normalized_variance * 5
        elif normalized_slope < 0.1:
            # Scende ancora un po'
            patience = 6 + normalized_variance * 3
        else:
            # Sta ancora scendendo rapidamente
            patience = 4 + normalized_variance * 2
        
        return patience
    
    def _calculate_variance(self, loss_array):
        """
        Bassa varianza = più stabile = patience più ALTA
        Alta varianza = oscillazioni = patience più BASSA
        """
        variance = np.var(loss_array)
        mean_loss = np.mean(loss_array)
        
        # Varianza normalizzata
        cv = np.sqrt(variance) / (mean_loss + 1e-5)  # Coefficient of variation
        
        # Se CV < 0.05 → molto stabile → patience = 10
        # Se CV > 0.2 → molto instabile → patience = 4
        patience = 10 - (cv / 0.2) * 6  # Scala da 10 a 4
        
        return patience
    
    def _calculate_slope(self, loss_array):
        """
        Misura come sta scendendo la loss.
        
        Slope negativo (scende) = continua ad imparare = patience BASSA
        Slope positivo (sale) = sta peggiorando = patience ALTA
        """
        coefficients = np.polyfit(np.arange(len(loss_array)), loss_array, 1)
        slope = coefficients[0]
        
        # Se slope è negativo, la loss sta scendendo (buono)
        # Se slope è positivo, la loss sta salendo (male)
        
        if slope < -0.05:
            # Sta scendendo bene
            patience = 4
        elif slope < 0:
            # Sta scendendo piano
            patience = 6
        elif slope < 0.05:
            # È piatta (plateau)
            patience = 8
        else:
            # Sta salendo
            patience = 10
        
        return patience
    
    def _calculate_stability(self, loss_array):
        """
        Usa la deviazione standard degli ultimi N valori.
        
        Bassa deviazione std = stabile = patience ALTA
        Alta deviazione std = oscillazioni = patience BASSA
        """
        # Guarda gli ultimi 5 valori (o meno se non disponibili)
        recent_losses = loss_array[-min(5, len(loss_array)):]
        
        std = np.std(recent_losses)
        mean = np.mean(recent_losses)
        
        # Normalizza
        normalized_std = std / (mean + 1e-5)
        
        # Se normalized_std è basso → stabile → patience alta
        patience = 5 + (0.2 - normalized_std) / 0.2 * 5
        
        return patience
    
    def __call__(self, val_loss, model, epoch):
        """
        Chiama l'early stopping per ogni epoca.
        """
        
        # Accumula la history della loss
        self.loss_history.append(val_loss)
        
        # Calcola la patience dopo le prime epoche (almeno 3)
        if not self.patience_calculated and epoch >= 3:
            self.calculate_patience()
        
        # Se patience non è ancora calcolato, usa uno provvisorio
        current_patience = self.patience if self.patience is not None else self.min_patience
        
        # Prima volta
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.best_epoch = epoch
            return
        
        # Controlla miglioramento
        if val_loss < self.best_loss * (1 - self.delta):
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
            self.best_epoch = epoch
            
            if self.verbose:
                print(f"✓ Validation loss migliorata: {val_loss:.4f}")
        else:
            self.counter += 1
            
            if self.verbose:
                print(f"✗ Nessun miglioramento per {self.counter}/{current_patience} epoche")
            
            if self.counter >= current_patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n EARLY STOPPING ATTIVATO!")
                    print(f"   Miglior modello era all'epoca {self.best_epoch} con loss: {self.best_loss:.4f}")
                    if self.patience_calculated:
                        print(f"   Patience calcolato (mode={self.mode}): {current_patience} epoche")
    
    def save_checkpoint(self, model):
        """Salva lo stato del modello."""
        torch.save(model.state_dict(), self.path)

    def load_best_model(self, model):
        device = next(model.parameters()).device
        state_dict = torch.load(self.path, map_location=device)
        model.load_state_dict(state_dict)
        return model
    
    def get_statistics(self):
        """
        Ritorna statistiche sulla validation loss.
        Utile per debug e analisi.
        """
        if not self.loss_history:
            return None
        
        loss_array = np.array(self.loss_history)
        
        return {
            "mean": float(np.mean(loss_array)),
            "std": float(np.std(loss_array)),
            "min": float(np.min(loss_array)),
            "max": float(np.max(loss_array)),
            "cv": float(np.std(loss_array) / np.mean(loss_array)),  # Coefficient of variation
            "trend": "descending" if loss_array[-1] < loss_array[0] else "ascending",
            "epochs_recorded": len(loss_array),
            "calculated_patience": self.patience
        }

