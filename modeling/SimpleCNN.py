import torch
import pytorch_lightning as pl
from ModelInitializer import ModelInitializer
from Metrics import PLMetrics
import torchmetrics.classification as m

class SimpleCNN(pl.LightningModule):
    """Creates a cnn with specified network architecture. Calculates dense network shape for last layer"""
    def __init__(self, config, input_shape=(224,224), fold_idx=0):
        super().__init__()
        self.config = config
        self.fold_idx = fold_idx
        self.input_shape = input_shape
        self.network, self.criterion, self.optimizer = ModelInitializer(self.config).initialize_model_crit_opt(self.input_shape)
        
        self.metrics = PLMetrics()
        self.compute_train_acc = m.BinaryAccuracy()
        self.compute_valid_acc = m.BinaryAccuracy()
        self.compute_valid_precision = m.BinaryPrecision()
        self.compute_valid_recall = m.BinaryRecall()
        self.compute_valid_f1 = m.BinaryF1Score()
        self.compute_valid_auroc = m.BinaryAUROC()

        #self.save_hyperparameters()

    def forward(self, x):
        return self.network(x)
    
    def configure_optimizers(self):
        return self.optimizer
    
    def common_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        labels = labels.unsqueeze(1).float()
        loss = self.criterion(logits, labels)
        return loss, logits, labels
    
    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch, batch_idx)
        preds = torch.round(torch.sigmoid(logits))
        self.log(f'fold_{self.fold_idx}/train_loss', loss, on_epoch=True)
        self.compute_train_acc(preds, labels)
        self.log(f'fold_{self.fold_idx}/train_accuracy', self.compute_train_acc, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch, batch_idx)
        probs = torch.sigmoid(logits) 
        preds = torch.round(probs)

        self.log(f'fold_{self.fold_idx}/val_loss', loss, on_epoch=True)
        self.compute_valid_acc(preds, labels)
        self.log(f'fold_{self.fold_idx}/val_accuracy', self.compute_valid_acc, on_step=False, on_epoch=True)

        self.compute_valid_acc(preds, labels)
        self.compute_valid_precision(preds, labels)
        self.compute_valid_recall(preds, labels)
        self.compute_valid_f1(preds, labels) 
        self.compute_valid_auroc(probs, labels)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch, batch_idx)
        return loss
    
    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        return torch.round(torch.sigmoid(logits))

