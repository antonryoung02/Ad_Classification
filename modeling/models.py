import torch.nn as nn
import torch
import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    """Creates a cnn with specified network architecture. Calculates dense network shape for last layer"""
    def __init__(self, network, input_shape=(320,320)):
        super(SimpleCNN, self).__init__()
        self.input_shape = input_shape
        self.network = network
        #Adds fc layer of correct dimensions
        self.network.add_module("fc", nn.Linear(self.find_fc_layer_shape(), 1))

    def forward(self, x):
        """Performs forward propogation"""
        return self.network(x)
    
    def load_model_checkpoint(self, file_path:str) -> nn.Module:
        """Loads weights of pretrained model"""
        self.load_state_dict(torch.load(file_path))
        return self

    def save_model_checkpoint(self, file_path: str = "./pi_inference"):
        """Saves the model's state dictionary to output file. Default to raspberry pi directory"""
        torch.save(self.state_dict(), file_path)
    
    def find_fc_layer_shape(self):
        """Finds the output shape of the last conv/pool layer in a Sequential model, 
       which is the required input shape for the fc layer."""
        batch_size = 1
        num_channels = 3
        dummy_input = torch.rand(batch_size, num_channels, *self.input_shape)
        print(dummy_input.shape)

        with torch.no_grad():
            for layer in self.network:
                dummy_input = layer(dummy_input)
                if isinstance(layer, nn.Flatten):
                    break  # Stop right before Flatten layer
        return dummy_input.numel()
