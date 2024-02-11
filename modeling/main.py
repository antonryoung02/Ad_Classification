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
subset_size = 500
indices = torch.randperm(len(data)).tolist()[:subset_size]
data = Subset(data, indices)
# --------------------------

model = SimpleCNN()
# model.load_model_checkpoint('./modeling/simple_cnn_checkpoint.pth')
# model.load_model_checkpoint('./modeling/simple_cnn_checkpoint.pth')
criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
hyperparameters = {
    "batch_size": [32],
    "num_epochs": [4],
    "optimizer": [optim.Adam],
    "learning_rate": [0.001],
    "weight_decay": [0.01],
}
best_model = grid_search(data, 3, hyperparameters, model, criterion)
best_model.save("./modeling/grid_best_checkpoint.pth")

best_model.inference(
    "./modeling/data/data_neg/add_random_filter_20240108_172033_image.png"
)
best_model.plot_curves()
# train_loader = None  # todo fix
# val_loader = None
# run_model = RunModel(data, model, criterion, optimizer, train_loader, val_loader)

# run_model.run(num_epochs=12)  # set early stopping instead!
# run_model.plot_curves()
##run_model.save("./mac_inference/larger_cnn_checkpoint.pth")
# run_model.save("./pi_inference/larger_cnn_checkpoint.pth")
