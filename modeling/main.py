import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from models import SimpleCNN
from run_model import RunModel
from utils import grid_search
import torch.optim as optim
from torch.utils.data import Subset


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

data = ImageFolder(root="./modeling/data", transform=transform)

# Only for testing!
subset_size = 100
indices = torch.randperm(len(data)).tolist()[:subset_size]
data = Subset(data, indices)
# --------------------------

model = SimpleCNN

criterion = nn.BCEWithLogitsLoss
optimizer = optim.Adam

hyperparameters = {
    "batch_size": [128],
    "num_epochs": [20],
    "network": [nn.Sequential(            
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten())],
    "lr": [0.01],
    "weight_decay": [0.01],  # L2 Reg
}
k = 5

best_model = grid_search(data, k, hyperparameters, model, criterion, optimizer)
# best_model.save("grid_best_checkpoint.pth")

# best_model.plot_curves()

