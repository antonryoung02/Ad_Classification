import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from models import SimpleCNN
from run_model import RunModel

transform = transforms.Compose([
    transforms.ToTensor(),
])

data = ImageFolder(root='processed_data', transform=transform)
#model = SimpleCNN()
model = SimpleCNN()
model.load_model_checkpoint('./modeling/simple_cnn_checkpoint.pth')
batch_size = 32
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

run_model = RunModel(batch_size, data, model, criterion, optimizer)

#run_model.run(num_epochs=10)
#run_model.plot_curves()
run_model.inference("/Users/anton/Desktop/Screen Shot 2023-12-23 at 2.27.24 PM.png")
#run_model.save("./modeling/simple_cnn_checkpoint.pth")
