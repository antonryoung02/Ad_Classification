import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC

class CNN(pl.LightningModule):
    def __init__(self, config, initializer, input_shape=(3, 224,224), fold_idx=0):
        super().__init__()
        self.config = config
        self.fold_idx = fold_idx
        self.input_shape = input_shape
        self.network, self.criterion, self.optimizer, self.scheduler = initializer.initialize_model_crit_opt_sched(self.input_shape)

        self.train_acc = BinaryAccuracy()
        self.valid_acc = BinaryAccuracy()
        self.valid_precision = BinaryPrecision()
        self.valid_recall = BinaryRecall()
        self.valid_f1 = BinaryF1Score()
        self.valid_auroc = BinaryAUROC()

    def forward(self, x):
        return self.network(x)
    
    def configure_optimizers(self):
        if self.scheduler is not None:
            return {"optimizer":self.optimizer, "lr_scheduler":self.scheduler}
        return self.optimizer
    
    def common_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs).squeeze()
        labels = labels.float()
        loss = self.criterion(logits, labels)
        return loss, logits, labels
    
    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch, batch_idx)
        probs = torch.sigmoid(logits)
        preds = torch.round(probs)
        self.log(f'fold_{self.fold_idx}/Train_Loss', loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "probs":probs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch, batch_idx)
        probs = torch.sigmoid(logits)
        preds = torch.round(probs)
        self.log(f'fold_{self.fold_idx}/Validation_Loss', loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "probs":probs, "labels": labels}


    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch, batch_idx)
        return loss
    
    def predict_step(self, batch, batch_idx=None):
        inputs = batch
        logits = self.forward(inputs)
        return torch.round(torch.sigmoid(logits))

