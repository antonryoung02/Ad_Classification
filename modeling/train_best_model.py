import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from models import SimpleCNN
from run_model import RunModel
from utils import grid_search
import torch.optim as optim
from torch.utils.data import Subset
from performance_tracker import PerformanceTracker
from torch.utils.data import DataLoader
import utils

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

data = ImageFolder(root="./modeling/data", transform=transform)
# Only for testing!
# subset_size = 20
# indices = torch.randperm(len(data)).tolist()[:subset_size]
# data = Subset(data, indices)
# --------------------------

model = SimpleCNN
criterion = nn.BCEWithLogitsLoss
optimizer = optim.Adam
performance_tracker = PerformanceTracker

best_hyperparameters = {
    "batch_size": 128,
    "num_epochs": 51,
    "network":nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding="same"),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding="same"),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Conv2d(32, 32, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Flatten(),
    ),
    "lr": 0.0005,
    "weight_decay": 0.0001,
}
model_params = utils.get_model_params(best_hyperparameters)
data_loader_params = utils.get_data_loader_params(best_hyperparameters)
training_params = utils.get_training_params(best_hyperparameters)
optimizer_params = utils.get_optimizer_params(best_hyperparameters)
criterion_params = utils.get_criterion_params(best_hyperparameters)
current_model = model(**model_params)

data_loader = DataLoader(data, batch_size=data_loader_params["batch_size"], shuffle=True)

current_optimizer = optimizer(
    current_model.parameters(), **optimizer_params
) 
current_criterion = criterion(**criterion_params)

run_model = RunModel(  # could feed lambda term to model for l1
    current_model, current_criterion, current_optimizer, performance_tracker, data_loader
)

run_model.run(**training_params)
run_model.model.save_model_checkpoint(file_path=f"{utils.get_hparam_model_name(best_hyperparameters)}.pth")
