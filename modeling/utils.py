import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
    
class FireModule(torch.nn.Module):
    def __init__(self, in_channels:int, num_squeeze_1x1:int, num_expand_1x1:int, num_expand_3x3:int):
        """
        Creates a Fire Module from the squeezenet paper with batch normalization and ReLU activations.
        Number of output feature maps = num_expand_1x1 + num_expand 3x3

        Args:
            in_channels (int): The number of feature maps from the previous layer
            num_squeeze_1x1 (int): The total number of filters in the 'squeeze layer'
            num_expand_1x1 (int): The number of 1x1 filters in the 'expand layer'
            num_expand_3x3 (int): The number of 3x3 filters in the 'expand layer'
        """
        super(FireModule, self).__init__()
        self.squeeze_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, num_squeeze_1x1, kernel_size=1),
            torch.nn.BatchNorm2d(num_squeeze_1x1),
            torch.nn.ReLU(inplace=True)
        )
        self.expand_1x1_layer = torch.nn.Sequential(
            torch.nn.Conv2d(num_squeeze_1x1, num_expand_1x1, kernel_size=1),
            torch.nn.BatchNorm2d(num_expand_1x1),
            torch.nn.ReLU(inplace=True)
        )
        self.expand_3x3_layer = torch.nn.Sequential(
            torch.nn.Conv2d(num_squeeze_1x1, num_expand_3x3, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_expand_3x3),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x:torch.Tensor):
        x = self.squeeze_layer(x)
        x_1x1 = self.expand_1x1_layer(x)
        x_3x3 = self.expand_3x3_layer(x)
        x = torch.cat((x_1x1, x_3x3), dim=1)
        return x

class SqueezeNetWithSkipConnections(nn.Module):
    """Squeezenet architecture with simple residual connections between fire modules"""
    def __init__(self, config:dict, input_shape:Tuple[int, int, int]):
        """
        Initializes the model layers      

        Args:
            config (dict): Contains the hparam keys 'dropout', 'incr_e', 'base_e', 'pct_3x3', 'sr', 
            input_shape (tuple): Input shape of one image (num channels, image height, image width)
        """
        super().__init__()
        dropout_rate = config['dropout']
        self.config = config

        self.stem = nn.Sequential(
            nn.Conv2d(input_shape[0], 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.dropout = nn.Dropout(dropout_rate)
        squeeze_1x1, expand_1x1, expand_3x3, ei0 = self._get_ei_for_layer(0)
        self.fire2 = FireModule(96, squeeze_1x1, expand_1x1, expand_3x3)
        squeeze_1x1, expand_1x1, expand_3x3, ei1  = self._get_ei_for_layer(1)
        self.fire3 = FireModule(ei0, squeeze_1x1, expand_1x1, expand_3x3)
        squeeze_1x1, expand_1x1, expand_3x3, ei2  = self._get_ei_for_layer(2)
        self.fire4 = FireModule(ei1, squeeze_1x1, expand_1x1, expand_3x3)
        squeeze_1x1, expand_1x1, expand_3x3, ei3  = self._get_ei_for_layer(3)
        self.fire5 = FireModule(ei2, squeeze_1x1, expand_1x1, expand_3x3)
        squeeze_1x1, expand_1x1, expand_3x3, ei4  = self._get_ei_for_layer(4)
        self.fire6 = FireModule(ei3, squeeze_1x1, expand_1x1, expand_3x3)
        squeeze_1x1, expand_1x1, expand_3x3, ei5  = self._get_ei_for_layer(5)
        self.fire7 = FireModule(ei4, squeeze_1x1, expand_1x1, expand_3x3)
        squeeze_1x1, expand_1x1, expand_3x3, ei6  = self._get_ei_for_layer(6)
        self.fire8 = FireModule(ei5, squeeze_1x1, expand_1x1, expand_3x3)
        squeeze_1x1, expand_1x1, expand_3x3, ei7  = self._get_ei_for_layer(7)
        self.fire9 = FireModule(ei6, squeeze_1x1, expand_1x1, expand_3x3)
        self.final_conv = nn.Conv2d(ei7, 1, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def _get_ei_for_layer(self, i:int):
        incr_e = self.config['incr_e']
        base_e = self.config['base_e']
        freq = 2 #Residual connections structure expects freq=2
        pct_3x3 = self.config['pct_3x3']
        sr = self.config['sr']
        ei = base_e + (incr_e * np.floor((i / freq)))

        squeeze_1x1 = int(ei * sr)
        expand_1x1 = int(ei) - int(ei * pct_3x3)
        expand_3x3 = int(ei * pct_3x3)
        return squeeze_1x1, expand_1x1, expand_3x3, int(ei)
    
    def forward(self, x:torch.Tensor):
        x = self.stem(x)
        x = self.fire2(x)
        res1 = x
        x = self.fire3(x)
        x = self.fire4(res1 + x)
        x = self.maxpool(x)
        res2 = x
        x = self.fire5(x)
        x = self.fire6(res2 + x)
        res3 = x
        x = self.fire7(x)
        x = self.fire8(res3 + x)
        x = self.maxpool(x)
        res4 = x
        x = self.fire9(x)
        x = self.dropout(res4 + x)
        x = self.final_conv(x)
        x = self.avg_pool(x)
        return x.view(x.size(0), -1)  # output shape is (batch_size, 1)

class SimpleCNN(nn.Module):
    def __init__(self, config:dict, input_shape:Tuple[int, int, int]):
        super().__init__()
        self.config = config
        kernels = self.config['kernels']
        dropout_rate = self.config['dropout']
        fc_units = self.config['fc_units']
        self.network =  nn.Sequential(
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
        self.network.add_module("fc", nn.Linear(self._find_fc_layer_shape(self.network, input_shape), fc_units))
        self.network.add_module("fc2", nn.Linear(fc_units,1))

    def forward(self, x:torch.Tensor):
        return self.network(x)
    
    def _find_fc_layer_shape(self, network:nn.Sequential, input_shape:Tuple[int, int, int]) -> int:
        """Finds the number of neurons that feed into a model's fully dense layers

        Args:
            network (nn.Module): A network that contains a flatten layer where the Linear layers are to be added
            input_shape (tuple): Input shape of one image (num channels, image height, image width)

        Returns:
            int: The number of units that the model's Flatten() layer outputs
        """
        batch_size = 1
        dummy_input = torch.rand(batch_size, *input_shape)

        with torch.no_grad():
            for layer in network:
                dummy_input = layer(dummy_input)
                if isinstance(layer, nn.Flatten):
                    break
        return dummy_input.numel()