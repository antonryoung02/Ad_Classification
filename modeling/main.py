import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from models import SimpleCNN
from run_model import RunModel

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

data = ImageFolder(root="./modeling/data", transform=transform)
model = SimpleCNN()
# model.load_model_checkpoint('./modeling/simple_cnn_checkpoint.pth')
# model.load_model_checkpoint('./modeling/simple_cnn_checkpoint.pth')
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = None  # todo fix
val_loader = None
run_model = RunModel(data, model, criterion, optimizer, train_loader, val_loader)

run_model.run(num_epochs=12)  # set early stopping instead!
run_model.plot_curves()
run_model.save("./mac_inference/larger_cnn_checkpoint.pth")
run_model.save("./pi_inference/larger_cnn_checkpoint.pth")
