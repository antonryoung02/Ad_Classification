import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from Configurator import Configurator

class Trainer:
    """Encapulates pytorch model learning functions."""

    def __init__(
        self,
        config: dict, 
        train_loader: DataLoader,
        val_loader: DataLoader = None,
    ):
        """
        param data: ImageFolder object of all true/neg examples
        param model: Model from models.py
        param criterion: Model loss criteria
        param optimizer: Model optimizer
        """
        self.configurator = Configurator(config)
        self.model = self.configurator.get_model()
        self.criterion = self.configurator.get_criterion()
        self.optimizer = self.configurator.get_optimizer(self.model)
        self.device = self.configurator.get_device()

        self.model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_epoch(self, epoch, fold_name) -> float:
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

        wandb.log({f"{fold_name}/train_loss": train_loss / len(self.train_loader) })

        return train_loss / len(self.train_loader) 

    def validate_epoch(self, epoch, fold_name):
        """Calculates validation predictions. Called at each epoch in model training if validation set exists."""
        self.model.eval()
        val_loss = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                labels = labels.unsqueeze(1).type_as(outputs)

                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                total_samples += len(labels)
        
        wandb.log({f"{fold_name}/val_loss": val_loss / len(self.val_loader)})

        return val_loss / len(self.val_loader)
            
    def run(self, num_epochs: int, fold_name:str):
        """Trains the model to desired threshold(s)"""
        epoch_progress = tqdm(
            range(1, num_epochs + 1),
            desc="Overall Epoch Progress",
            unit="epoch",
            leave=False,
        )

        for epoch in epoch_progress:
            train_loss = self.train_epoch(epoch, fold_name)
            if self.val_loader:
                val_loss = self.validate_epoch(epoch, fold_name)
            else:
                val_loss = None

            epoch_progress.set_description(f"Epoch {epoch}/{num_epochs}")
            epoch_progress.set_postfix(
                Train_Loss=train_loss, Val_Loss=val_loss
            )
        
    def get_val_labels_and_preds(self):
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                preds = self.model(inputs)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
        return val_labels, val_preds

