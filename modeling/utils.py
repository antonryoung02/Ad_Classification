import torch
from torch import nn

def find_fc_layer_input_shape(model: nn.Sequential, input_shape: tuple):
    """Finds the output shape of the last conv/pool layer in a Sequential model, 
       which is the required input shape for the first fc layer."""
    batch_size = 1
    dummy_input = torch.rand(batch_size, *input_shape)

    with torch.no_grad():
        for layer in model.network:
            dummy_input = layer(dummy_input)
            if isinstance(layer, nn.Flatten):
                break  # Stop right before Flatten layer

    return dummy_input.shape[1:]  # Return the shape excluding the batch dimension