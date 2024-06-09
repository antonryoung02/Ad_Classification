import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb

class StoreAveragedMetrics(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_validation_end(self, trainer, pl_module):
        pl_module.metrics.valid_acc = pl_module.compute_valid_acc.compute().item()
        pl_module.metrics.valid_precision = pl_module.compute_valid_precision.compute().item()
        pl_module.metrics.valid_recall = pl_module.compute_valid_recall.compute().item()
        pl_module.metrics.valid_f1 = pl_module.compute_valid_f1.compute().item()
        pl_module.metrics.valid_auroc = pl_module.compute_valid_auroc.compute().item()
        
        pl_module.compute_valid_acc.reset()
        pl_module.compute_valid_precision.reset()
        pl_module.compute_valid_recall.reset()
        pl_module.compute_valid_f1.reset()
        pl_module.compute_valid_auroc.reset()

class KWorstPredictionsLogger(pl.Callback):
    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def on_fit_end(self, trainer, pl_module):
        device = pl_module.device
        all_imgs = []
        all_labels = []
        all_losses = []

        pl_module.eval()
        with torch.no_grad():
            for batch in trainer.val_dataloaders:
                val_imgs, val_labels = batch
                val_imgs = val_imgs.to(device)
                val_labels = val_labels.to(device)

                loss, _, _ = pl_module.common_step((val_imgs, val_labels), None) 
                loss = loss.unsqueeze(0)

                all_imgs.append(val_imgs)
                all_labels.append(val_labels)
                all_losses.append(loss)

        all_imgs = torch.cat(all_imgs)
        all_labels = torch.cat(all_labels)
        all_losses = torch.cat(all_losses)

        topk_losses, topk_indices = torch.topk(all_losses, self.k)
        topk_imgs = all_imgs[topk_indices]
        topk_labels = all_labels[topk_indices]

        trainer.logger.experiment.log({
            "worst predictions": [
                wandb.Image(img.cpu(), caption=f"Loss:{loss.item()}, Label:{label.item()}")
                for img, loss, label in zip(topk_imgs, topk_losses, topk_labels)
            ]
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
                break  # Stop right before Flatten layer
    return dummy_input.numel()
