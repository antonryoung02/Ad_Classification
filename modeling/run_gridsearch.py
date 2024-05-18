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

model_one = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding="same"),
    nn.ReLU(),
    nn.Dropout2d(0.25), 
    nn.Conv2d(32, 32, kernel_size=3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Conv2d(32, 32, kernel_size=3),
    nn.ReLU(),
    nn.Dropout2d(0.25),
    nn.Conv2d(32, 32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Flatten(),
)

model_two = nn.Sequential(
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
)

hyperparameters = {
    "batch_size": [128],
    "num_epochs": [30],
    "network": [model_one, model_two],  # Compare dropout / no dropout
    "lr": [0.0001, 0.0005, 0.001, 0.01],
    "weight_decay": [0, 0.0001, 0.001, 0.01],  # L2 Reg
}
k = 5

performance_df = grid_search(data, k, hyperparameters, model, criterion, optimizer)
performance_df.to_csv("performances.csv", index=False)
print(performance_df)

