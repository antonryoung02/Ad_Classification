import torch
from torch import nn
from typing import Tuple
from torch.utils.data import random_split, DataLoader, Subset
from sklearn.model_selection import KFold
from run_model import RunModel
import numpy as np
import itertools
import pandas as pd
import copy


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


# Create function that calls k_fold for each hyperparameter combination
#  scores for each, retrain best combination on all data
def grid_search(
    data,
    k,
    hyperparameters_grid,
    model,
    criterion,
):
    """Performs grid search over hyperparameters"""
    best_model = None
    best_model_auc = 0
    iteration = 0

    keys, values = zip(*hyperparameters_grid.items())
    hyperparameter_combinations = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]

    for hyperparameters in hyperparameter_combinations:
        print(f"Iteration {iteration}. Testing hyperparameters: {hyperparameters}")
        cv_results, run_model = k_fold_cross_validation(
            data, k, hyperparameters, model, criterion
        )
        print(f"Results: {cv_results}")
        current_auc = cv_results["auc"]

        if current_auc > best_model_auc:
            best_model_auc = current_auc
            best_model = run_model
            print(f"New best model found. PR AUC: {best_model_auc}")
            with open("best_hyperparameters.txt", "w") as file:
                file.write(str(hyperparameters))
        else:
            print(f"PR AUC: {best_model_auc}")
        iteration += 1

        print("------------------------")

    return best_model


def k_fold_cross_validation(data, k, hyperparameters, model, criterion):
    """Performs k-fold cross validation, records performance scores"""
    scores = {
        "auc": [],
        "train_loss": [],
        "val_loss": [],
        "val_precision": [],
        "val_recall": [],
    }

    indices = torch.randperm(len(data)).tolist()
    kf = KFold(n_splits=k, shuffle=True, random_state=123)

    for train_idx, val_idx in kf.split(indices):
        run_model = train_fold(
            indices, train_idx, val_idx, data, model, criterion, hyperparameters
        )
        run_model.get_precision_recall_auc()
        scores["auc"].append(run_model.get_precision_recall_auc())
        scores["train_loss"].append(run_model.train_losses[-1])
        scores["val_loss"].append(run_model.val_losses[-1])
        scores["val_precision"].append(run_model.val_precisions[-1])
        scores["val_recall"].append(run_model.val_recalls[-1])

    return {
        "train_loss": np.mean(scores["train_loss"]),
        "val_loss": np.mean(scores["val_loss"]),
        "val_precision": np.mean(scores["val_precision"]),
        "val_recall": np.mean(scores["val_recall"]),
        "auc": np.mean(scores["auc"]),
    }, run_model


def train_fold(indices, train_idx, val_idx, data, model, criterion, hyperparameters):
    """Helper function that trains a single fold of k-fold cross validation"""
    batch_size = hyperparameters["batch_size"]
    num_epochs = hyperparameters["num_epochs"]
    optimizer = hyperparameters["optimizer"]
    learning_rate = hyperparameters["learning_rate"]
    weight_decay = hyperparameters["weight_decay"]

    train_subset = Subset(data, [indices[i] for i in train_idx])
    val_subset = Subset(data, [indices[i] for i in val_idx])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    current_model = copy.deepcopy(model)
    # current_model = model
    current_optimizer = optimizer(
        current_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )  # Have to reinitialize optimizer with new model

    run_model = RunModel(  # could feed lambda term to model for l1
        current_model, criterion, current_optimizer, train_loader, val_loader
    )
    run_model.run(num_epochs)
    return run_model
