import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from Trainer import Trainer
from torch.utils.data import Subset, DataLoader
from PerformanceTrackers import WandBPerformanceTracker
from sklearn.model_selection import KFold
import wandb
import random
import numpy as np

def k_fold_cross_validation(k=4):
    with wandb.init() as run:
        config = wandb.config

        run = wandb.init(project="Ad_Classification",config=config, group=str(random.random()))
        kf = KFold(n_splits=k, shuffle=True, random_state=123)
        pt = WandBPerformanceTracker()

        transform = transforms.ToTensor()
        data = ImageFolder(root="data", transform=transform)
        indices = torch.randperm(len(data)).tolist()

        agg_labels, agg_preds = [], []

        for fold,(train_idx, val_idx) in enumerate(kf.split(indices)):
            fold_name = f"fold_{fold}"
            wandb.run.tags = [fold_name]
        
            train_subset = Subset(data, [indices[i] for i in train_idx])
            val_subset = Subset(data, [indices[i] for i in val_idx])
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

            trainer = Trainer(config, train_loader, val_loader)
            trainer.run(config['num_epochs'], fold_name)

            labels, preds = trainer.get_val_labels_and_preds()
            agg_labels.extend(labels)
            agg_preds.extend(preds)

        pt.record_metrics(np.array(agg_labels), np.array(agg_preds), fold_name)
        run.finish()


def main():
    sweep_config = {
        "name": "Ad_Classification",
        "method": "random",
        "metric": {
            "name": "accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "fc_units": {
                "values": [32, 48, 64]
            },
            "dropout": {
                "values": [0, 0.1, 0.2, 0.3, 0.4]
            },
            "kernels": {
                "values": [[16, 16, 16, 16], [32, 32, 16, 16], [32,32,32,32]]
            },
            "batch_size": {
                "values": [256]
            },
            "num_epochs": {
                "values": [8]
            },
            "lr": {
                "distribution": "log_uniform_values",
                "min": 0.00001,
                "max": 0.1
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 0.00001,
                "max": 0.1
            }
        }
    }

    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="Ad_Classification")
    wandb.agent(sweep_id, k_fold_cross_validation, count=14)

if __name__ == "__main__":
    main()