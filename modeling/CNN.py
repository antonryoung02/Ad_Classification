import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from modeling.ModelInitializer import ModelInitializerFactory
from typing import Tuple, Union, Dict
from torch.optim import Optimizer

class CNN(pl.LightningModule):
    """A Convolutional Neural Network learning class capable of initializing any configuration of Network, Criterion, Optimizer, and LR Scheduler"""
    def __init__(self, config:dict, input_shape:Tuple[int, int, int]=(3, 224,224), fold_idx:int=0):
        super().__init__()
        self.config = config
        self.fold_idx = fold_idx
        self.input_shape = input_shape
        self.initializer = ModelInitializerFactory()(config)
        self.network, self.criterion, self.optimizer, self.scheduler = self.initializer.initialize_model_crit_opt_sched(self.input_shape)
        self.train_acc = BinaryAccuracy()
        self.valid_acc = BinaryAccuracy()
        self.valid_precision = BinaryPrecision()
        self.valid_recall = BinaryRecall()
        self.valid_f1 = BinaryF1Score()
        self.valid_auroc = BinaryAUROC()

    def forward(self, x:torch.Tensor):
        return self.network(x)
    
    def configure_optimizers(self) -> Union[Optimizer, dict]:
        if self.scheduler is not None:
            return {"optimizer":self.optimizer, "lr_scheduler":self.scheduler}
        return self.optimizer
    
    def common_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, labels = batch
        logits = self.forward(inputs).squeeze()
        labels = labels.float()
        loss = self.criterion(logits, labels)
        return loss, logits, labels
    
    def training_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx:int) -> dict:
        loss, logits, labels = self.common_step(batch, batch_idx)
        probs = torch.sigmoid(logits)
        preds = torch.round(probs)
        self.log(f'fold_{self.fold_idx}/Train_Loss', loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "probs":probs, "labels": labels}

    def validation_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx:int) -> dict:
        loss, logits, labels = self.common_step(batch, batch_idx)
        probs = torch.sigmoid(logits)
        preds = torch.round(probs)
        self.log(f'fold_{self.fold_idx}/Validation_Loss', loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "probs":probs, "labels": labels}

    def test_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx:int) -> dict:
        loss, logits, labels = self.common_step(batch, batch_idx)
        probs = torch.sigmoid(logits)
        preds = torch.round(probs)
        return {"loss": loss, "preds": preds, "probs":probs, "labels": labels}
    
    def predict_step(self, batch:torch.Tensor, batch_idx:int=0) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(batch)
            return torch.round(torch.sigmoid(logits))

