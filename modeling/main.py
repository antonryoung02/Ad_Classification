import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from torchvision.transforms import v2
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
from modeling.CNN import CNN
from modeling.Callbacks import KWorstPredictionsLogger, MetricsLogger, GradientNormLogger
from modeling.Augment import AugmentedImageFolder, TrainTransformation
from typing import Dict

def main():
    wandb.login()
    train_best_model(best_config_dict)
    # sweep_id = wandb.sweep(squeeze_sweep_config, project="Ad_Classification")
    # wandb.agent(sweep_id, k_fold_cross_validation, count=1)

def k_fold_cross_validation(k:int=4):
    with wandb.init():
        config = wandb.config
        kf = KFold(n_splits=k, shuffle=True, random_state=123)

        val_transform = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_transform  = v2.Compose([v2.ToTensor()])
        train_augmentation = TrainTransformation()
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

            model = CNN(config, fold_idx=fold_idx)
            wandb_logger = WandbLogger(project='Ad_Classification')
            wandb_logger.watch(model, log="all")
            metrics_logger = MetricsLogger()

            trainer = pl.Trainer(
                deterministic=True,
                accelerator="gpu",
                devices=1,
                min_epochs=2,
                max_epochs=config['num_epochs'],

                callbacks=[
                    metrics_logger,
                    KWorstPredictionsLogger(5),
                    EarlyStopping(f'fold_{fold_idx}/Validation_Loss'),
                    GradientNormLogger(),
                    ],
                logger=wandb_logger,
                gradient_clip_val=0.5,
                overfit_batches=5,
            )
            
            trainer.fit(model, train_loader, val_loader)

            fold_metrics = metrics_logger.get_epoch_metrics()
            metrics['Average_Validation_Accuracy'].append(fold_metrics["Validation_Accuracy"])
            metrics['Average_Validation_Precision'].append(fold_metrics["Validation_Precision"])
            metrics['Average_Validation_Recall'].append(fold_metrics["Validation_Recall"])
            metrics['Average_Validation_F1_Score'].append(fold_metrics["Validation_F1_Score"])
            metrics['Average_Validation_AUROC'].append(fold_metrics["Validation_AUROC"])

        avg_metrics = {key: np.mean(val) for key, val in metrics.items()}
        wandb.log(avg_metrics)
        wandb.finish()

def train_best_model(config:Dict[str, any]):
    train_transform  = v2.ToTensor()
    train_augmentation = TrainTransformation()
    data_folder = AugmentedImageFolder(root="data", transform=train_transform, augmentation=train_augmentation)
    data_loader = DataLoader(data_folder, batch_size=config['batch_size'], shuffle=True, num_workers=29)
                
    model = CNN(config, fold_idx=0)
    wandb_logger = WandbLogger(project='Ad_Classification')
    wandb_logger.watch(model, log="all")

    trainer = pl.Trainer(
        deterministic=True,
        accelerator="gpu",
        devices=1,
        min_epochs=2,
        max_epochs=config['num_epochs'],
        callbacks=[
            ],
        logger=wandb_logger,
        gradient_clip_val=0.5,
    )

    trainer.fit(model, data_loader)
    checkpoint_path = "example.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    if os.path.isfile(checkpoint_path):
        print(f"Checkpoint saved at {checkpoint_path}")
    else:
        print("Failed to save checkpoint")

best_config_dict = {
    "base_e": 128,
    "batch_size":256,
    "dropout":0.06482,
    "incr_e":96,
    "initializer":"squeezenet",
    "lr":0.01786,
    "lr_gamma":0.3295,
    "num_epochs":20,
    "pct_3x3":0.5,
    "sr":0.25,
    "weight_decay":0.001374
}
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
            "values": [[16, 16, 16, 16], [8, 12, 16, 20], [12, 14, 16, 18], [14, 16, 18, 24]]
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
            "values": [96, 128, 160]
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
            "values": [20]
        },
        "lr": {
            "distribution": "log_uniform_values",
            "min":0.001,
            "max":0.1
        },
        "lr_gamma": {
            "distribution": "uniform",
            "min":0.1,
            "max":1
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 0.0001,
            "max": 0.1
        }
    }
}
if __name__ == "__main__":
    main()