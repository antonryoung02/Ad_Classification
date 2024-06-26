import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb

class LogMetrics(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_fit_end(self, trainer, pl_module):
        valid_acc = pl_module.compute_cumu_valid_acc.compute().item()
        valid_precision = pl_module.compute_valid_precision.compute().item()
        valid_recall = pl_module.compute_valid_recall.compute().item()
        valid_f1 = pl_module.compute_valid_f1.compute().item()
        valid_auroc = pl_module.compute_valid_auroc.compute().item()

        metrics = {
            f'fold_{pl_module.fold_idx}/Validation_Precision': valid_precision,
            f'fold_{pl_module.fold_idx}/Validation_Recall': valid_recall,
            f'fold_{pl_module.fold_idx}/Validation_F1_Score': valid_f1,
            f'fold_{pl_module.fold_idx}/Validation_AUROC': valid_auroc,
            f'fold': pl_module.fold_idx,
        }
        
        trainer.logger.experiment.log(metrics)

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

                logits = pl_module.forward(val_imgs)
                losses = criterion(logits, val_labels.unsqueeze(1).float())

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

def find_fc_layer_shape(network, input_shape):
    """Finds the output shape of the last conv/pool layer in a Sequential model, 
    which is the required input shape for the fc layer."""
    batch_size = 1
    num_channels = 3
    dummy_input = torch.rand(batch_size, num_channels, *input_shape)

    with torch.no_grad():
        for layer in network:
            dummy_input = layer(dummy_input)
            if isinstance(layer, nn.Flatten):
                break
    return dummy_input.numel()
