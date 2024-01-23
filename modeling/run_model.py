import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
from typing import Tuple
from torchvision import transforms
from PIL import Image


class RunModel:
    """Encapulates pytorch learning / performance functions."""

    def __init__(
        self,
        data: ImageFolder,
        model: nn.Module,
        criterion,
        optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
    ):
        """
        param data: ImageFolder object of all true/neg examples
        param model: Model from models.py
        param criterion: Model loss criteria
        param optimizer: Model optimizer
        param train_proportion: Size of training set
        """

        self.data = data
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_losses = []
        self.val_losses = []
        self.val_precisions = []
        self.val_recalls = []

    def train(self) -> float:
        """Performs 1 training step. Called at each epoch in model training"""
        self.model.train()
        train_loss = 0
        train_loader = tqdm(self.train_loader, desc="Training", leave=False)

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model(inputs)
            labels = labels.unsqueeze(1).type_as(outputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_loader.set_description(f"Training (Loss: {loss.item():.4f})")

        return train_loss / len(self.train_loader)

    def validate(self) -> Tuple[float, float, float]:
        """Calculates validation metrics. Called at each epoch in model training if validation set exists"""
        self.model.eval()
        val_loss = 0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                labels = labels.unsqueeze(1).type_as(outputs)

                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                predicted = torch.sigmoid(outputs).round()
                all_outputs.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        precision = precision_score(all_labels, all_outputs)
        recall = recall_score(all_labels, all_outputs)
        avg_val_loss = val_loss / len(self.val_loader)

        return avg_val_loss, precision, recall

    def run(self, num_epochs: int):
        """Trains the model to desired threshold(s)"""
        epoch_progress = tqdm(
            range(1, num_epochs + 1), desc="Overall Epoch Progress", unit="epoch"
        )

        for epoch in epoch_progress:
            train_loss = self.train()
            self.train_losses.append(train_loss)
            if self.val_loader:
                val_loss, precision, recall = self.validate()
                self.val_losses.append(val_loss)
                self.val_precisions.append(precision)
                self.val_recalls.append(recall)
            else:
                val_loss, precision, recall = None, None, None

            epoch_progress.set_description(f"Epoch {epoch}/{num_epochs}")
            epoch_progress.set_postfix(
                Train_Loss=train_loss,
                Val_Loss=val_loss,
                Precision=precision,
                Recall=recall,
            )

    def inference(self, image_path):
        """Performs inference on a single image and displays the result."""

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        image = Image.open(image_path).resize((320, 320)).convert("RGB")
        image = transform(image).unsqueeze(
            0
        )  # Using preprocessing function assuming inference data is not transformed
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
            probability = torch.sigmoid(outputs).item()

        predicted_class = "Advertisement" if probability > 0.5 else "Hockey"

        plt.imshow(
            image.squeeze(0).permute(1, 2, 0)
        )  # Permute to get channel last format
        plt.title(
            f"Predicted: {predicted_class}, Probability of Advertisement: {probability:.4f}"
        )
        plt.show()

    def save(self, file_path: str = "./pi_inference"):
        """Saves the model's state dictionary to output file. Default to raspberry pi directory"""
        torch.save(self.model.state_dict(), file_path)

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
        plt.show()
