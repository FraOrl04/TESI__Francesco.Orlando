class TrainingHistory:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []

    def update(self, train_loss, val_loss, val_acc):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)

    def save(self, path="training_history.npz"):
        import numpy as np
        np.savez(path,
                 train_loss=self.train_loss,
                 val_loss=self.val_loss,
                 val_acc=self.val_acc)

    @staticmethod
    def load(path="training_history.npz"):
        import numpy as np
        data = np.load(path)
        hist = TrainingHistory()
        hist.train_loss = data["train_loss"].tolist()
        hist.val_loss = data["val_loss"].tolist()
        hist.val_acc = data["val_acc"].tolist()
        return hist
