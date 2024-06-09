import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
import wandb
import pytorch_lightning as pl
from SimpleCNN import SimpleCNN
from utils import KWorstPredictionsLogger, StoreAveragedMetrics
from pytorch_lightning.loggers import WandbLogger

def k_fold_cross_validation(k=4):
    with wandb.init() as run:
        config = wandb.config
        kf = KFold(n_splits=k, shuffle=True, random_state=123)

        transform = transforms.ToTensor()
        data = ImageFolder(root="data", transform=transform)
        indices = torch.randperm(len(data)).tolist()
        agg_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            print(f"num train pts: {len(train_idx)}")
            train_subset = Subset(data, [indices[i] for i in train_idx])
            val_subset = Subset(data, [indices[i] for i in val_idx])
            train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)
            model = SimpleCNN(config, fold_idx=fold_idx)
            wandb_logger = WandbLogger(project='Ad_Classification')
            trainer = pl.Trainer(deterministic=True, accelerator="gpu", devices=1, 
                                min_epochs=2, max_epochs=config['num_epochs'], 
                                callbacks=[KWorstPredictionsLogger(4), StoreAveragedMetrics()],
                                logger=wandb_logger)
            trainer.fit(model, train_loader, val_loader)

            agg_metrics.append(model.metrics.get_metrics_dict())

        average_metrics = {}
        num_folds = len(agg_metrics)
        for key in agg_metrics[0]:
            average_metrics[key] = sum(d[key] for d in agg_metrics) / num_folds
        wandb.log(average_metrics)
        wandb.finish()



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
                "values": [32, 48]
            },
            "dropout": {
                "distribution": "uniform",
                "min":0,
                "max":0.5
            },
            "kernels": {
                "values": [[16, 16, 16, 16], [32, 32, 16, 16], [32,32,32,32]]
            },
            "batch_size": {
                "values": [256]
            },
            "num_epochs": {
                "values": [10]
            },
            "lr": {
                "distribution": "log_uniform_values",
                "min": 0.000001,
                "max": 0.001
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 0.000001,
                "max": 0.01
            }
        }
    }

    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="Ad_Classification")
    wandb.agent(sweep_id, k_fold_cross_validation, count=14)

if __name__ == "__main__":
    main()