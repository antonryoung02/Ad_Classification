import wandb
import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC

class LogMetrics(pl.Callback):
    """"""
    def __init__(self):
        super().__init__()
        self.metrics = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        preds, probs, labels = outputs['preds'], outputs['probs'], outputs['labels']
        pl_module.train_acc(preds, labels)

        pl_module.log(f'fold_{pl_module.fold_idx}/Train_Accuracy', pl_module.train_acc, on_step=False, on_epoch=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        preds, probs, labels = outputs['preds'], outputs['probs'], outputs['labels']
        pl_module.valid_acc(preds, labels)
        pl_module.valid_precision(preds, labels)
        pl_module.valid_recall(preds, labels)
        pl_module.valid_f1(preds, labels)
        pl_module.valid_auroc(probs, labels)

        pl_module.log(f'fold_{pl_module.fold_idx}/Validation_Accuracy', pl_module.valid_acc, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        train_acc = pl_module.train_acc.compute().item()
        pl_module.log(f'fold_{pl_module.fold_idx}/Train_Accuracy', train_acc)
        pl_module.train_acc.reset()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        valid_acc = pl_module.valid_acc.compute().item()
        valid_precision = pl_module.valid_precision.compute().item()
        valid_recall = pl_module.valid_recall.compute().item()
        valid_f1 = pl_module.valid_f1.compute().item()
        valid_auroc = pl_module.valid_auroc.compute().item()

        self.metrics = {
            'Validation_Accuracy': valid_acc,
            'Validation_Precision': valid_precision,
            'Validation_Recall': valid_recall,
            'Validation_F1_Score': valid_f1,
            'Validation_AUROC': valid_auroc,
        }
        pl_module.valid_acc.reset()
        pl_module.valid_precision.reset()
        pl_module.valid_recall.reset()
        pl_module.valid_f1.reset()
        pl_module.valid_auroc.reset()

    def on_fit_end(self, trainer:pl.Trainer, pl_module:pl.LightningModule) -> None:
        trainer.logger.experiment.log({
            f'fold_{pl_module.fold_idx}/Validation_Precision': self.metrics['Validation_Precision'],
            f'fold_{pl_module.fold_idx}/Validation_Recall': self.metrics['Validation_Recall'],
            f'fold_{pl_module.fold_idx}/Validation_F1_Score': self.metrics['Validation_F1_Score'],
            f'fold_{pl_module.fold_idx}/Validation_AUROC': self.metrics['Validation_AUROC'],
            'fold': pl_module.fold_idx,
        })

    def get_metrics(self):
        return self.metrics

class GradientNormLogger(pl.Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx=None):
        grad_norms = {}
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad, p=2).item()
                grad_norms[f"grad_norm_{name}"] = grad_norm
                if grad_norm == 0.0:
                    print(f"Warning: Gradient norm is zero for layer {name}")
            else:
                grad_norms[f"grad_norm_{name}"] = 0.0
                print(f"Warning: No gradient for layer {name}")
        trainer.logger.experiment.log(grad_norms)

class KWorstPredictionsLogger(pl.Callback):
    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def on_fit_end(self, trainer, pl_module):
        device = pl_module.device
        all_imgs = []
        all_labels = []
        all_losses = []
        all_probs = []
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        pl_module.eval()
        with torch.no_grad():
            for batch_idx, (val_imgs, val_labels) in enumerate(trainer.val_dataloaders):
                val_imgs = val_imgs.to(device)
                val_labels = val_labels.to(device)

                logits = pl_module.forward(val_imgs).squeeze()
                losses = criterion(logits, val_labels.float())

                all_imgs.append(val_imgs)
                all_labels.append(val_labels)
                all_losses.append(losses)
                all_probs.append(torch.sigmoid(logits))

        all_imgs = torch.cat(all_imgs)
        all_labels = torch.cat(all_labels)
        all_losses = torch.cat(all_losses)
        all_probs = torch.cat(all_probs)

        topk_losses, topk_indices = torch.topk(all_losses.squeeze(), self.k)
        topk_imgs = all_imgs[topk_indices]
        topk_labels = all_labels[topk_indices]
        topk_probs = all_probs[topk_indices]

        trainer.logger.experiment.log({
            "Worst Predictions By Model": [
                wandb.Image(img.cpu(), caption=f"P(Advertisement):{prob.item():.3f}, Label:{label.item()}")
                for img, prob, label in zip(topk_imgs, topk_probs, topk_labels)
            ], 
            "fold":pl_module.fold_idx
        })