import torch
import pytorch_lightning as pl
import torchmetrics.classification as m

class CNN(pl.LightningModule):
    def __init__(self, config, initializer, input_shape=(224,224), fold_idx=0):
        super().__init__()
        self.example_input_array = torch.Tensor(16, 3, 224, 224) #Logs layer shapes
        self.config = config
        self.fold_idx = fold_idx
        self.input_shape = input_shape

        self.network, self.criterion, self.optimizer = initializer.initialize_model_crit_opt(self.input_shape)
    
        self.compute_train_acc = m.BinaryAccuracy()
        self.compute_cumu_valid_acc = m.BinaryAccuracy()
        self.compute_valid_acc = m.BinaryAccuracy()
        self.compute_valid_precision = m.BinaryPrecision()
        self.compute_valid_recall = m.BinaryRecall()
        self.compute_valid_f1 = m.BinaryF1Score()
        self.compute_valid_auroc = m.BinaryAUROC()

    def forward(self, x):
        return self.network(x)
    
    def configure_optimizers(self):
        return self.optimizer
    
    def common_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs).squeeze()
        labels = labels.float()
        loss = self.criterion(logits, labels)
        return loss, logits, labels
    
    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch, batch_idx)
        preds = torch.round(torch.sigmoid(logits))
        self.log(f'fold_{self.fold_idx}/Train_Loss', loss, on_step=False, on_epoch=True)
        self.compute_train_acc(preds, labels)
        self.log(f'fold_{self.fold_idx}/Train_Accuracy', self.compute_train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch, batch_idx)
        probs = torch.sigmoid(logits) 
        preds = torch.round(probs)

        self.compute_valid_acc.update(preds, labels)
        self.compute_cumu_valid_acc.update(preds, labels)
        self.compute_valid_precision.update(preds, labels)
        self.compute_valid_recall.update(preds, labels)
        self.compute_valid_f1.update(preds, labels)
        self.compute_valid_auroc.update(probs, labels)

        self.log(f'fold_{self.fold_idx}/Validation_Accuracy', self.compute_valid_acc, on_step=False, on_epoch=True)
        self.log(f'fold_{self.fold_idx}/Validation_Loss', loss, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch, batch_idx)
        return loss
    
    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        return torch.round(torch.sigmoid(logits))

