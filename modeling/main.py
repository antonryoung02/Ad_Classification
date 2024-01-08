import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from models import SimpleCNN
from run_model import RunModel

transform = transforms.Compose([
    transforms.ToTensor(),
])

data = ImageFolder(root='./modeling/data', transform=transform)
model = SimpleCNN()
#model.load_model_checkpoint('./modeling/simple_cnn_checkpoint.pth')
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

run_model = RunModel(data, model, criterion, optimizer)

run_model.run(num_epochs=10)
run_model.plot_curves()
run_model.save("./mac_inference/simple_cnn_checkpoint.pth")
run_model.save("./pi_inference/simple_cnn_checkpoint.pth")