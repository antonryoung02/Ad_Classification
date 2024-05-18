import matplotlib.pyplot as plt

class PerformanceTracker:
    """Stores model predictions and provides diagnostic/performance plots"""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_precisions = []
        self.val_recalls = []

    def plot_curves(self):
        """Plots and displays losses, precision/recall"""
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label="Training Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.val_precisions, label="Precision")
        plt.plot(epochs, self.val_recalls, label="Recall")
        plt.title("Precision and Recall")
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.legend()

        plt.tight_layout()
        plt.savefig("plotcurves.png")

        

