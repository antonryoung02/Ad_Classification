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
from modeling.Callbacks import KWorstPredictionsLogger, MetricsLogger, GradientNormLogger, CurriculumLearningCallback
from modeling.Augment import AugmentedImageFolder, AugmentationFactory
from modeling.utils import write_config_to_yaml, load_config, save_as_coreml, save_as_pt, add_tag_to_run

DATA_PATH = "../FootballVA"
PROJECT = "Ad_Classification_Football"

def main():
    wandb.login()
    train_best_model('./configs/2024-11-06_hearty-sweep-6.yaml')
    # config = load_config("sweep_config/ghostnet.yaml")
    # sweep_id = wandb.sweep(config, project=PROJECT)
    # wandb.agent(sweep_id, k_fold_cross_validation, count=10)

def k_fold_cross_validation():
    with wandb.init() as run:
        config = wandb.config
        write_config_to_yaml(run.name, config)
        add_tag_to_run(run, config['model_initializer'])
        
        kf = KFold(n_splits=config['num_fold'], shuffle=True, random_state=123)

        val_transform = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_transform  = v2.Compose([v2.ToTensor()])
 
        train_augmentation = AugmentationFactory()(config)
        train_data_folder = AugmentedImageFolder(root=DATA_PATH, transform=train_transform, augmentation=train_augmentation)
        val_data_folder = AugmentedImageFolder(root=DATA_PATH, transform=val_transform, augmentation=None) 
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
            train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=30)
            val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=30)

            model = CNN(config, fold_idx=fold_idx)
            wandb_logger = WandbLogger(project=run.project)
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
            break

        avg_metrics = {key: np.mean(val) for key, val in metrics.items()}
        wandb.log(avg_metrics)
        add_tag_to_run(run, "completed")
        wandb.finish()

def train_best_model(filepath:str):
    config = load_config(filepath)
    train_transform  = v2.Compose([v2.ToTensor()])
    train_augmentation = AugmentationFactory()(config)
    
    data_folder = AugmentedImageFolder(root=DATA_PATH, transform=train_transform, augmentation=train_augmentation)
    data_loader = DataLoader(data_folder, batch_size=config['batch_size'], shuffle=True, num_workers=30)
                
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
    save_as_pt(model.network, 'model')
    save_as_coreml(model.network, 'model')

if __name__ == "__main__":
    main()