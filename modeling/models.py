import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(409600, 1), #utils.find_fc_layer_input_shape
        )

    def forward(self, x):
        """Performs forward propogation"""
        return self.network(x)
    
    def load_model_checkpoint(self, file_path:str) -> nn.Module:
        """Loads weights of pretrained model"""
        self.load_state_dict(torch.load(file_path))
        return self

