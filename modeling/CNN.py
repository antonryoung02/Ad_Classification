import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from modeling.ModelInitializerFactory import ModelInitializerFactory
from typing import Tuple, Union
from torch.optim import Optimizer

class CNN(pl.LightningModule):
    input_shape = (3,224,224)
    """A Convolutional Neural Network learning class capable of initializing any configuration of Network, Criterion, Optimizer, and LR Scheduler"""
    def __init__(self, config:dict, fold_idx:int=0):
        super().__init__()
        self.config = config
        self.fold_idx = fold_idx
        self.initializer = ModelInitializerFactory()(config)
        self.network, self.criterion, self.optimizer, self.scheduler = self.initializer.initialize_model_crit_opt_sched(CNN.input_shape)
        self.train_acc = Accuracy(task="multiclass", num_classes=5)
        self.valid_acc = Accuracy(task="multiclass", num_classes=5)
        self.valid_precision = Precision(task="multiclass", num_classes=5)
        self.valid_recall = Recall(task="multiclass", num_classes=5)
        self.valid_f1 = F1Score(task="multiclass", num_classes=5)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def configure_optimizers(self) -> Union[Optimizer, dict]:
        if self.scheduler is not None:
            return {"optimizer":self.optimizer, "lr_scheduler":self.scheduler}
        return self.optimizer
    
    def common_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, labels = batch
        logits = self.forward(inputs).squeeze()
        labels = labels.long()
        
        loss = self.criterion(logits, labels)
        return loss, logits, labels
    
    def training_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx:int) -> dict:
        loss, logits, labels = self.common_step(batch, batch_idx)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        self.log(f'fold_{self.fold_idx}/Train_Loss', loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "probs":probs, "labels": labels}

    def validation_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx:int) -> dict:
        loss, logits, labels = self.common_step(batch, batch_idx)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        self.log(f'fold_{self.fold_idx}/Validation_Loss', loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "probs":probs, "labels": labels}

    def test_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx:int) -> dict:
        loss, logits, labels = self.common_step(batch, batch_idx)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {"loss": loss, "preds": preds, "probs":probs, "labels": labels}
    
    def predict_step(self, batch:torch.Tensor, batch_idx:int) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(batch)
            return torch.argmax(torch.softmax(logits, dim=1), dim=1)

