import torch
from torch import nn
from typing import Tuple
from torch.utils.data import random_split, DataLoader, Subset
from sklearn.model_selection import KFold
from run_model import RunModel
import numpy as np
import itertools
import pandas as pd


def find_fc_layer_input_shape(model: nn.Sequential, input_shape: tuple):
    """Finds the output shape of the last conv/pool layer in a Sequential model,
    which is the required input shape for the fc layer."""
    batch_size = 1
    dummy_input = torch.rand(batch_size, *input_shape)

    with torch.no_grad():
        for layer in model.network:
            dummy_input = layer(dummy_input)
            if isinstance(layer, nn.Flatten):
                break  # Stop right before Flatten layer

    return dummy_input.shape[1:]


# is this needed anymore?
# def split_dataset(
#     train_proportion: float, data, batch_size=32
# ) -> Tuple[DataLoader, DataLoader]:
#     """Helper function called during init that splits all data into train/val"""
#     total_samples = len(data)
#     train_size = int(total_samples * train_proportion)
#     val_size = total_samples - train_size
#     train_dataset, val_dataset = random_split(data, [train_size, val_size])
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     return train_loader, val_loader


# Create function that calls k_fold for each hyperparameter combination
# Save scores for each, retrain best combination on all data
def grid_search(
    data,
    k,
    hyperparameters_grid,
    model,
    criterion,
    save_path="./modeling/grid_search_results.csv",
):
    """Performs grid search over hyperparameters, saves results to csv"""
    results = []

    keys, values = zip(*hyperparameters_grid.items())
    hyperparameter_combinations = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]

    for hyperparameters in hyperparameter_combinations:
        print(f"Testing hyperparameters: {hyperparameters}")
        cv_results = k_fold_cross_validation(data, k, hyperparameters, model, criterion)
        results.append({**cv_results, **hyperparameters})

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    best_hyperparameters = results_df.loc[results_df["val_loss"].idxmin()].to_dict()

    print(f"Best hyperparameters: {best_hyperparameters}")
    return best_hyperparameters


def k_fold_cross_validation(data, k, hyperparameters, model, criterion):
    """Performs k-fold cross validation, records performance scores"""

    scores = {
        "train_loss": [],
        "val_loss": [],
        "val_precision": [],
        "val_recall": [],
    }

    indices = torch.randperm(len(data)).tolist()
    kf = KFold(n_splits=k, shuffle=False)

    for train_idx, val_idx in kf.split(indices):
        run_model = train_fold(
            indices, train_idx, val_idx, data, model, criterion, hyperparameters
        )

        scores["train_loss"].append(run_model.train_losses[-1])
        scores["val_loss"].append(run_model.val_losses[-1])
        scores["val_precision"].append(run_model.val_precisions[-1])
        scores["val_recall"].append(run_model.val_recalls[-1])

    return {
        "train_loss": np.mean(scores["train_loss"]),
        "val_loss": np.mean(scores["val_loss"]),
        "val_precision": np.mean(scores["val_precision"]),
        "val_recall": np.mean(scores["val_recall"]),
    }


def train_fold(indices, train_idx, val_idx, data, model, criterion, hyperparameters):
    """Helper function that trains a single fold of k-fold cross validation"""
    batch_size = hyperparameters["batch_size"]
    num_epochs = hyperparameters["num_epochs"]
    optimizer = hyperparameters["optimizer"]
    learning_rate = hyperparameters["learning_rate"]

    train_subset = Subset(data, [indices[i] for i in train_idx])
    val_subset = Subset(data, [indices[i] for i in val_idx])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    current_model = model()
    current_optimizer = optimizer(
        current_model.parameters(), lr=learning_rate
    )  # Have to reinitialize optimizer with new model

    run_model = RunModel(
        current_model, criterion, current_optimizer, train_loader, val_loader
    )
    run_model.run(num_epochs)
    return run_model
