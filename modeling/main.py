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
from modeling.Augment import AugmentedImageFolder, AugmentationFactory
from typing import Dict
import yaml
from datetime import datetime

def main():
    wandb.login()
    # train_best_model(best_config_dict)
    config = load_config("./modeling/sweep_config/football_squeezenet.yaml")
    sweep_id = wandb.sweep(config, project="Ad_Classification")
    wandb.agent(sweep_id, k_fold_cross_validation, count=2)

def load_config(path:str) -> dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config
        
def k_fold_cross_validation():
    with wandb.init() as run:
        config = wandb.config

        current_date = datetime.now().strftime('%Y-%m-%d')
        file_path = f'./modeling/configs/{current_date}_{run.name}.yaml'
        
        with open(file_path, 'w') as yaml_file:
            yaml.dump(dict(config), yaml_file, default_flow_style=False)
            
        train_config = config.pop('training') 
        kf = KFold(n_splits=train_config['num_fold'], shuffle=True, random_state=123)

        val_transform = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_transform  = v2.Compose([v2.ToTensor()])
        
        augmentation_config = config.pop('augmentation')
        train_augmentation = AugmentationFactory()(augmentation_config)
        train_data_folder = AugmentedImageFolder(root="../FootballVA", transform=train_transform, augmentation=train_augmentation)
        val_data_folder = AugmentedImageFolder(root="../FootballVA", transform=val_transform, augmentation=None) 
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
            train_loader = DataLoader(train_subset, batch_size=train_config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=train_config['batch_size'], shuffle=False)

            model = CNN(config.pop('algorithm'), fold_idx=fold_idx)
            wandb_logger = WandbLogger(project=run.project)
            wandb_logger.watch(model, log="all")
            metrics_logger = MetricsLogger()

            trainer = pl.Trainer(
                deterministic=True,
                accelerator="gpu",
                devices=1,
                min_epochs=2,
                max_epochs=train_config['num_epochs'],

                callbacks=[
                    metrics_logger,
                    KWorstPredictionsLogger(5),
                    EarlyStopping(f'fold_{fold_idx}/Validation_Loss'),
                    GradientNormLogger(),
                    ],
                logger=wandb_logger,
                gradient_clip_val=0.5,
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

def train_best_model(config:dict):
    train_transform  = v2.Compose([v2.ToTensor()])
    train_augmentation = AugmentationFactory()(config['augmentation'])
    
    train_config = config.pop('training')
    data_folder = AugmentedImageFolder(root="data", transform=train_transform, augmentation=train_augmentation)
    data_loader = DataLoader(data_folder, batch_size=train_config['batch_size'], shuffle=True, num_workers=29)
                
    model = CNN(config.pop('algorithm'), fold_idx=0)
    wandb_logger = WandbLogger(project='Ad_Classification')
    wandb_logger.watch(model, log="all")

    trainer = pl.Trainer(
        deterministic=True,
        accelerator="gpu",
        devices=1,
        min_epochs=2,
        max_epochs=train_config['num_epochs'],
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

if __name__ == "__main__":
    main()