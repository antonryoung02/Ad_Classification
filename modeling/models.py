import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.input_shape = (320,320)
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(409600, 1),
        )

    def forward(self, x):
        """Performs forward propogation"""
        return self.network(x)
    
    def load_model_checkpoint(self, file_path:str) -> nn.Module:
        """Loads weights of pretrained model"""
        self.load_state_dict(torch.load(file_path))
        return self
    
    def find_fc_layer_input_shape(self):
        """Finds the output shape of the last conv/pool layer in a Sequential model, 
       which is the required input shape for the fc layer."""
        batch_size = 1
        dummy_input = torch.rand(batch_size, self.input_shape)

        with torch.no_grad():
            for layer in self.network:
                dummy_input = layer(dummy_input)
                if isinstance(layer, nn.Flatten):
                    break  # Stop right before Flatten layer

        return dummy_input.shape[1:]

