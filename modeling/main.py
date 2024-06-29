import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
import wandb
import pytorch_lightning as pl
from CNN import CNN
from utils import KWorstPredictionsLogger, LogMetrics, GradientNormLogger
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping
from Augment import AugmentedImageFolder, DefaultsTransformation
from ModelInitializer import SimpleCNNInitializer, SqueezeNetInitializer

def k_fold_cross_validation(k=4):
    with wandb.init(tags=["15_epochs"]) as run:
        config = wandb.config
        kf = KFold(n_splits=k, shuffle=True, random_state=123)

        val_transform = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
        train_transform  = v2.ToTensor()
        train_augmentation = DefaultsTransformation()
        train_data_folder = AugmentedImageFolder(root="data", transform=train_transform, augmentation=train_augmentation)
        val_data_folder = AugmentedImageFolder(root="data", transform=val_transform, augmentation=None) 
        indices = torch.randperm(len(train_data_folder)).tolist()
        

        metrics = {
            'Average_Validation_Accuracy': [],
            'Average_Validation_Precision': [],
            'Average_Validation_Recall': [],
            'Average_Validation_F1_Score': [],
            'Average_Validation_AUROC': [],
        }

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            train_subset = Subset(val_data_folder, [indices[i] for i in train_idx])
            val_subset = Subset(val_data_folder, [indices[i] for i in val_idx])
            train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)

            initializer = SimpleCNNInitializer(config) if config['initializer'] == "simple_cnn" else SqueezeNetInitializer(config)
            model = CNN(config, initializer, fold_idx=fold_idx)
            wandb_logger = WandbLogger(project='Ad_Classification')
            wandb_logger.watch(model, log="all")

            trainer = pl.Trainer(deterministic=True, accelerator="gpu", devices=1, 
                                min_epochs=2, max_epochs=config['num_epochs'], 
                                callbacks=[
                                            KWorstPredictionsLogger(5), 
                                           LogMetrics(), 
                                           EarlyStopping(f'fold_{fold_idx}/Validation_Loss'),
                                           GradientNormLogger(),
                                           ],
                                logger=wandb_logger,
                                gradient_clip_val=0.5)
            
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
    simple_sweep_config = {
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
            "initializer": {
                "values": ["simple_cnn"]
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

    squeeze_sweep_config = {
        "name": "Ad_Classification",
        "method": "random",
        "metric": {
            "name": "Average_Validation_Accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "base_e": {
                "values": [128]
            },
            "initializer": {
                "values": ["squeezenet"]
            },
            "incr_e": {
                "values": [96, 128, 160]
            },
            "sr": {
                "values":[0.125, 0.1925, 0.25, 0.325]
            },
            "pct_3x3": {
                "values":[0.4, 0.5, 0.6]
            },
            "dropout": {
                "distribution": "uniform",
                "min":0,
                "max":0.5
            },
            "batch_size": {
                "values": [256]
            },
            "num_epochs": {
                "values": [15]
            },
            "lr": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.1
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 0.0001,
                "max": 0.1
            }
        }
    }
    wandb.login()
    # sweep_id = wandb.sweep(squeeze_sweep_config, project="Ad_Classification")
    # wandb.agent(sweep_id, k_fold_cross_validation, count=10)
    sweep_id = wandb.sweep(squeeze_sweep_config, project="Ad_Classification")
    wandb.agent(sweep_id, k_fold_cross_validation, count=15)

if __name__ == "__main__":
    main()