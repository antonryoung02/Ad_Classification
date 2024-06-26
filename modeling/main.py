import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
import wandb
import pytorch_lightning as pl
from SimpleCNN import SimpleCNN
from utils import KWorstPredictionsLogger, LogMetrics
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping
from Augment import AugmentedImageFolder, DefaultsTransformation

def k_fold_cross_validation(k=4):
    with wandb.init(tags=["15_epochs"]) as run:
        config = wandb.config
        kf = KFold(n_splits=k, shuffle=True, random_state=123)

        initial_transform = transforms.ToTensor()
        train_augmentation = DefaultsTransformation()
        train_data_folder = AugmentedImageFolder(root="data", transform=initial_transform, augmentation=train_augmentation)
        val_data_folder = AugmentedImageFolder(root="data", transform=initial_transform, augmentation=None) 
        indices = torch.randperm(len(train_data_folder)).tolist()
        

        metrics = {
            'Average_Validation_Accuracy': [],
            'Average_Validation_Precision': [],
            'Average_Validation_Recall': [],
            'Average_Validation_F1_Score': [],
            'Average_Validation_AUROC': [],
        }

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            train_subset = Subset(train_data_folder, [indices[i] for i in train_idx])
            val_subset = Subset(val_data_folder, [indices[i] for i in val_idx])
            train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)

            model = SimpleCNN(config, fold_idx=fold_idx)
            wandb_logger = WandbLogger(project='Ad_Classification')
            trainer = pl.Trainer(deterministic=True, accelerator="gpu", devices=1, 
                                min_epochs=2, max_epochs=config['num_epochs'], 
                                callbacks=[KWorstPredictionsLogger(5), LogMetrics(), EarlyStopping(f'fold_{fold_idx}/Validation_Loss')],
                                logger=wandb_logger)
            trainer.fit(model, train_loader, val_loader)

            metrics['Average_Validation_Accuracy'].append(model.compute_cumu_valid_acc.compute().item())
            metrics['Average_Validation_Precision'].append(model.compute_valid_precision.compute().item())
            metrics['Average_Validation_Recall'].append(model.compute_valid_recall.compute().item())
            metrics['Average_Validation_F1_Score'].append(model.compute_valid_f1.compute().item())
            metrics['Average_Validation_AUROC'].append(model.compute_valid_auroc.compute().item())

        avg_metrics = {key: np.mean(val) for key, val in metrics.items()}
        wandb.log(avg_metrics)
        wandb.finish()



def main():
    sweep_config = {
        "name": "Ad_Classification",
        "method": "random",
        "metric": {
            "name": "Average_Validation_Accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "fc_units": {
                "values": [16, 24, 32, 40, 48]
            },
            "dropout": {
                "distribution": "uniform",
                "min":0,
                "max":0.5
            },
            "kernels": {
                "values": [[16, 16, 16, 16], [8, 12, 16, 20], [12, 14, 16, 18], [14,16,18,24]]
            },
            "batch_size": {
                "values": [256]
            },
            "num_epochs": {
                "values": [10, 15]
            },
            "lr": {
                "distribution": "log_uniform_values",
                "min": 0.00001,
                "max": 0.01
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 0.000001,
                "max": 0.1
            }
        }
    }

    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="Ad_Classification")
    wandb.agent(sweep_id, k_fold_cross_validation, count=30)

if __name__ == "__main__":
    main()