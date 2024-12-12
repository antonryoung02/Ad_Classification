import wandb
import pytorch_lightning as pl
import torch
from typing import Dict, Tuple
from torch.optim import Optimizer
from pytorch_lightning.loggers import WandbLogger
from modeling.CNN import CNN
from modeling.Augment import GeneralImageAugmentations

class MulticlassMetricsLogger(pl.Callback):
    """
    Logs multiclass metrics

    Records train accuracy, valid accuracy, valid precision, valid recall, valid f1, and valid confusion matrices to wandb.
    """
    def __init__(self):
        super().__init__()
        self.metrics = {}
        self.valid_labels = torch.empty(0).to("cuda")
        self.valid_predictions = torch.empty(0).to("cuda")

    def on_train_batch_end(self, trainer:pl.Trainer, pl_module:CNN, outputs:Dict[str, torch.Tensor], batch:Tuple[torch.Tensor, torch.Tensor], batch_idx:int) -> None:
        preds, probs, labels = outputs['preds'], outputs['probs'], outputs['labels']
        pl_module.train_acc(preds, labels)

        pl_module.log(f'fold_{pl_module.fold_idx}/Train_Accuracy', pl_module.train_acc, on_step=False, on_epoch=True)

    def on_validation_batch_end(self, trainer:pl.Trainer, pl_module:CNN, outputs:Dict[str, torch.Tensor], batch:Tuple[torch.Tensor, torch.Tensor], batch_idx:int, dataloader_idx:int=0) -> None:
        preds, probs, labels = outputs['preds'], outputs['probs'], outputs['labels']
        self.valid_predictions = torch.cat([self.valid_predictions, preds], dim=0)
        self.valid_labels = torch.cat([self.valid_labels, labels], dim=0)

        pl_module.valid_acc(preds, labels)
        pl_module.valid_precision(preds, labels)
        pl_module.valid_recall(preds, labels)
        pl_module.valid_f1(preds, labels)

        pl_module.log(f'fold_{pl_module.fold_idx}/Validation_Accuracy', pl_module.valid_acc, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: CNN) -> None:
        train_acc = pl_module.train_acc.compute().item()
        pl_module.log(f'fold_{pl_module.fold_idx}/Train_Accuracy', train_acc)
        pl_module.train_acc.reset()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: CNN) -> None:

        valid_acc = pl_module.valid_acc.compute().item()
        valid_precision = pl_module.valid_precision.compute().item()
        valid_recall = pl_module.valid_recall.compute().item()
        valid_f1 = pl_module.valid_f1.compute().item()

        self.metrics = {
            'Validation_Accuracy': valid_acc,
            'Validation_Precision': valid_precision,
            'Validation_Recall': valid_recall,
            'Validation_F1_Score': valid_f1,
        }
        
        # Using built-in wandb confusion matrix logger. Maybe find alternative to match other metic computations?
        labels = self.valid_labels.to(int).cpu().tolist()
        predicted_classes = self.valid_predictions.to(int).cpu().tolist()
        wandb.log({
            "conf_mat" : wandb.plot.confusion_matrix(probs=None,
                y_true=labels, 
                preds=predicted_classes,
                class_names=["Ads", "Baseball", "Football", "Basketball", "Hockey"]
                )
            })


        self.valid_labels = torch.empty(0).to("cuda")
        self.valid_predictions = torch.empty(0).to("cuda")
        pl_module.valid_acc.reset()
        pl_module.valid_precision.reset()
        pl_module.valid_recall.reset()
        pl_module.valid_f1.reset()

    def on_fit_end(self, trainer:pl.Trainer, pl_module:CNN) -> None:
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log({
                f'fold_{pl_module.fold_idx}/Validation_Precision': self.metrics['Validation_Precision'],
                f'fold_{pl_module.fold_idx}/Validation_Recall': self.metrics['Validation_Recall'],
                f'fold_{pl_module.fold_idx}/Validation_F1_Score': self.metrics['Validation_F1_Score'],
                'fold': pl_module.fold_idx,
            })

        else:
            raise ValueError("trainer does not have a wandb logger attached!")


    def get_epoch_metrics(self) -> dict:
        return self.metrics

class BinaryMetricsLogger(pl.Callback):
    """Logs metrics to module logger on every epoch and end of training.
    Records Train Accuracy, Val Accuracy, Val Precision, Val Recall, Val F1 and Val AUROC
    """
    def __init__(self):
        super().__init__()
        self.metrics = {}
        raise NotImplementedError("Binary metrics are no longer supported with architectures")

    def on_train_batch_end(self, trainer:pl.Trainer, pl_module:CNN, outputs:Dict[str, torch.Tensor], batch:Tuple[torch.Tensor, torch.Tensor], batch_idx:int):
        preds, probs, labels = outputs['preds'], outputs['probs'], outputs['labels']
        pl_module.train_acc(preds, labels)

        pl_module.log(f'fold_{pl_module.fold_idx}/Train_Accuracy', pl_module.train_acc, on_step=False, on_epoch=True)

    def on_validation_batch_end(self, trainer:pl.Trainer, pl_module:CNN, outputs:Dict[str, torch.Tensor], batch:Tuple[torch.Tensor, torch.Tensor], batch_idx:int, dataloader_idx:int=0):
        preds, probs, labels = outputs['preds'], outputs['probs'], outputs['labels']
        pl_module.valid_acc(preds, labels)
        pl_module.valid_precision(preds, labels)
        pl_module.valid_recall(preds, labels)
        pl_module.valid_f1(preds, labels)
        pl_module.valid_auroc(probs, labels)

        pl_module.log(f'fold_{pl_module.fold_idx}/Validation_Accuracy', pl_module.valid_acc, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: CNN) -> None:
        train_acc = pl_module.train_acc.compute().item()
        pl_module.log(f'fold_{pl_module.fold_idx}/Train_Accuracy', train_acc)
        pl_module.train_acc.reset()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: CNN) -> None:
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

    def on_fit_end(self, trainer:pl.Trainer, pl_module:CNN) -> None:
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log({
                f'fold_{pl_module.fold_idx}/Validation_Precision': self.metrics['Validation_Precision'],
                f'fold_{pl_module.fold_idx}/Validation_Recall': self.metrics['Validation_Recall'],
                f'fold_{pl_module.fold_idx}/Validation_F1_Score': self.metrics['Validation_F1_Score'],
                f'fold_{pl_module.fold_idx}/Validation_AUROC': self.metrics['Validation_AUROC'],
                'fold': pl_module.fold_idx,
            })
        else:
            raise ValueError("trainer does not have a wandb logger attached!")

    def get_epoch_metrics(self) -> dict:
        return self.metrics

class GradientNormLogger(pl.Callback):
    """Logs the l2 gradient norm of each layer in the network for debugging purposes"""
    def on_before_optimizer_step(self, trainer:pl.Trainer, pl_module:CNN, optimizer:Optimizer, optimizer_idx:int=0) -> None:
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
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log(grad_norms)
        else:
            raise ValueError(f"trainer logger {type(trainer.logger)} is not a wandb logger!")

class KWorstPredictionsLogger(pl.Callback):
    """Uses the wandb logger to log images of the fully trained model's top k worst validation set losses"""
    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def on_fit_end(self, trainer:pl.Trainer, pl_module:CNN) -> None:
        device = pl_module.device
        all_imgs = []
        all_labels = []
        all_losses = []
        all_probs = []
        criterion = torch.nn.CrossEntropyLoss()

        pl_module.eval()
        with torch.no_grad():
            if trainer.val_dataloaders:
                for batch_idx, (val_imgs, val_labels) in enumerate(trainer.val_dataloaders):
                    val_imgs = val_imgs.to(device)
                    val_labels = val_labels.to(device)

                    logits = pl_module.forward(val_imgs).squeeze()
                    losses = criterion(logits, val_labels.long())

                    all_imgs.append(val_imgs)
                    all_labels.append(val_labels)
                    all_losses.append(losses.unsqueeze(0))
                    all_probs.append(torch.softmax(logits, dim=1))
            else:
                raise ValueError("Can't use a KWorstPredictionsLogger with no validation set(s)!")

        all_imgs = torch.cat(all_imgs)
        all_labels = torch.cat(all_labels)
        all_losses = torch.cat(all_losses)
        all_probs = torch.cat(all_probs)

        topk_losses, topk_indices = torch.topk(all_losses.squeeze(), self.k)
        topk_imgs = all_imgs[topk_indices]
        topk_labels = all_labels[topk_indices]
        topk_probs = all_probs[topk_indices]

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log({
                "Worst Predictions By Model": [
                    wandb.Image(img.cpu(), caption=f"P(Advertisement):{prob}, Label:{label.item()}")
                    for img, prob, label in zip(topk_imgs, topk_probs, topk_labels)
                ], 
                "fold":pl_module.fold_idx
            })
        else:
            raise ValueError("The trainer does not have a wandb logger attached!")

            
class CurriculumLearningCallback(pl.Callback):
    """Callback that increases the difficulty of image augmentations over time"""
    def __init__(self, augmentation):
        if not isinstance(augmentation, GeneralImageAugmentations):
            raise ValueError("Only GeneralImageAugmentation class is supported!")
        self.augmentation = augmentation

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: CNN) -> None:
        self.augmentation.hue = min(0.2, self.augmentation.hue + 0.01)
        self.augmentation.contrast = min(0.1, self.augmentation.contrast + 0.01)
            