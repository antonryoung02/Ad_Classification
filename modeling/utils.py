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
from torch.utils.tensorboard import SummaryWriter
from performance_tracker import PerformanceTracker


# Create function that calls k_fold for each hyperparameter combination
#  scores for each, retrain best combination on all data
def grid_search(
    data,
    k: int,
    hyperparameters_grid: dict,
    model: nn.Sequential,
    criterion,
    optimizer
):
    """Performs grid search over hyperparameters"""
    best_model = None
    best_model_auc = 0

    keys, values = zip(*hyperparameters_grid.items())
    hyperparameter_combinations = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]

    for iteration, hyperparameters in enumerate(hyperparameter_combinations):
        print(f"Iteration {iteration}. Testing hyperparameters: {hyperparameters}")
        run_model = k_fold_cross_validation(
            data=data, k=k, model=model, criterion=criterion, optimizer=optimizer, hyperparameters=hyperparameters
        )
        # print(f"Results: {cv_results}")
        # current_auc = cv_results["auc"]

        # if current_auc > best_model_auc:
        #     best_model_auc = current_auc
        #     best_model = run_model
        #     print(f"New best model found. PR AUC: {best_model_auc}")
        #     with open("best_hyperparameters.txt", "w") as file:
        #         file.write(str(hyperparameters))
        # else:
        #     print(f"PR AUC: {best_model_auc}")

        # print("------------------------")

    return best_model


def k_fold_cross_validation(data, k, model, criterion, optimizer, hyperparameters):
    """Performs k-fold cross validation, records performance scores"""

    indices = torch.randperm(len(data)).tolist()
    kf = KFold(n_splits=k, shuffle=True, random_state=123)


    for train_idx, val_idx in kf.split(indices):
        run_model = train_fold(
            indices=indices, train_idx=train_idx, val_idx=val_idx, data=data, model=model, criterion=criterion, optimizer=optimizer, hyperparameters=hyperparameters
        )

    return run_model


def train_fold(indices, train_idx, val_idx, data, model, criterion, optimizer, hyperparameters):
    """Helper function that trains a single fold of k-fold cross validation"""
    model_params = get_model_params(hyperparameters)
    data_loader_params = get_data_loader_params(hyperparameters)
    training_params = get_training_params(hyperparameters)
    optimizer_params = get_optimizer_params(hyperparameters)
    criterion_params = get_criterion_params(hyperparameters)

    #writer = SummaryWriter(comment=get_cv_name(hyperparameters))

    train_subset = Subset(data, [indices[i] for i in train_idx])
    val_subset = Subset(data, [indices[i] for i in val_idx])

    train_loader = DataLoader(train_subset, **data_loader_params, shuffle=True)
    val_loader = DataLoader(val_subset, **data_loader_params, shuffle=False)

    current_model = model(**model_params)

    current_optimizer = optimizer(
        current_model.parameters(), **optimizer_params
    ) 
    current_criterion = criterion(**criterion_params)

    performance_tracker = PerformanceTracker()
    run_model = RunModel(  # could feed lambda term to model for l1
        current_model, current_criterion, current_optimizer, performance_tracker, train_loader, val_loader
    )

    run_model.run(**training_params)
    return run_model


def get_cv_name(parameter_dict):
    name = ""
    for key, value in parameter_dict.items():
        name += key + "_" + value + "-"
    return name

def get_model_params(hparams):
    keys = ["input_shape", "network"]
    if "network" not in hparams:
        raise ValueError("Required argument network missing from hparams!")
    return {key: hparams[key] for key in keys if key in hparams}

def get_data_loader_params(hparams):
    keys = ["batch_size"]
    if "batch_size" not in hparams:
        raise ValueError("Required argument batch_size missing from hparams!")
    return {key: hparams[key] for key in keys if key in hparams}

def get_training_params(hparams):
    keys = ["num_epochs"]
    if "num_epochs" not in hparams:
        raise ValueError("Required argument num_epochs missing from hparams!")
    return {key: hparams[key] for key in keys if key in hparams}

def get_optimizer_params(hparams):
    keys = ["lr", "weight_decay"]
    return {key: hparams[key] for key in keys if key in hparams}


def get_criterion_params(hparams):
    keys = []
    return {key: hparams[key] for key in keys if key in hparams}