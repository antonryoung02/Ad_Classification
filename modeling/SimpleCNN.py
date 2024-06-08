import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    """Creates a cnn with specified network architecture. Calculates dense network shape for last layer"""
    def __init__(self, kernels, fc_units, dropout_rate, input_shape=(224,224)):
        super(SimpleCNN, self).__init__()
        self.input_shape = input_shape
        self.network = nn.Sequential(
            nn.Conv2d(3, kernels[0], kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate), 
            nn.Conv2d(kernels[0], kernels[1], kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(kernels[1], kernels[2], kernel_size=3),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(kernels[2], kernels[3], kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten()
        )
        self.network.add_module("fc", nn.Linear(self._find_fc_layer_shape(), fc_units))
        self.network.add_module("fc2", nn.Linear(fc_units,1))

    def forward(self, x):
        """Performs forward propogation"""
        return self.network(x)
    
    def _find_fc_layer_shape(self):
        """Finds the output shape of the last conv/pool layer in a Sequential model, 
       which is the required input shape for the fc layer."""
        batch_size = 1
        num_channels = 3
        dummy_input = torch.rand(batch_size, num_channels, *self.input_shape)

        with torch.no_grad():
            for layer in self.network:
                dummy_input = layer(dummy_input)
                if isinstance(layer, nn.Flatten):
                    break  # Stop right before Flatten layer
        return dummy_input.numel()
