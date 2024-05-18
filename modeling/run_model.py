import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc

from tqdm import tqdm
from typing import Tuple
from torchvision import transforms
from PIL import Image
import numpy as np

from performance_tracker import PerformanceTracker


class RunModel:
    """Encapulates pytorch model learning functions."""

    def __init__(
        self,
        model: nn.Module,
        criterion,
        optimizer,
        performance_tracker: PerformanceTracker,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
    ):
        """
        param data: ImageFolder object of all true/neg examples
        param model: Model from models.py
        param criterion: Model loss criteria
        param optimizer: Model optimizer
        """

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.performance_tracker = performance_tracker


    def train_epoch(self) -> float:
        """Performs 1 training step. Called at each epoch in model training"""
        self.model.train()
        train_loss = 0
        total_samples = 0
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
            total_samples += len(labels)
            train_loader.set_description(f"Training (Loss: {loss.item():.4f})")

        return train_loss / total_samples

    def validate_epoch(self) -> Tuple[np.array, np.array]:
        """Calculates validation predictions. Called at each epoch in model training if validation set exists."""
        self.model.eval()
        all_outputs = np.array([])
        all_labels = np.array([])

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                labels = labels.unsqueeze(1).type_as(outputs)

                all_outputs = np.concatenate((all_outputs, outputs.cpu().numpy().flatten()))
                all_labels = np.concatenate((all_labels, labels.cpu().numpy().flatten()))

        return all_labels, all_outputs

    def run(self, num_epochs: int):
        """Trains the model to desired threshold(s)"""
        epoch_progress = tqdm(
            range(1, num_epochs + 1),
            desc="Overall Epoch Progress",
            unit="epoch",
            leave=False,
        )

        for epoch in epoch_progress:
            train_loss = self.train_epoch()

            epoch_progress.set_description(f"Epoch {epoch}/{num_epochs}")
            epoch_progress.set_postfix(
                Train_Loss=train_loss
            )
        
        if self.val_loader:
            y_true, y_prob = self.validate_epoch()
            self.performance_tracker.record_metrics(y_true, y_prob)

        return self.model

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

