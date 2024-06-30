import torch
import torch.nn as nn
import numpy as np

def find_fc_layer_shape(network, input_shape):
    """Finds the output shape of the last conv/pool layer in a Sequential model, 
    which is the required input shape for the fc layer."""
    batch_size = 1
    num_channels = 3
    dummy_input = torch.rand(batch_size, num_channels, *input_shape)

    with torch.no_grad():
        for layer in network:
            dummy_input = layer(dummy_input)
            if isinstance(layer, nn.Flatten):
                break
    return dummy_input.numel()

class FireModule(torch.nn.Module):
    def __init__(self, in_channels, num_squeeze_1x1, num_expand_1x1, num_expand_3x3):
        """
        Kwarg defaults are as specified in SqueezeNet paper

        in_channels: Given from previous layer

        ei: Total number of filters in the expand layer

        pct3x3: Percentage of 3x3 filters in the expand layer

        SR: Ratio of (1x1) filters used in the squeeze layer
        (si = ei * SR) etc
        
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

    def forward(self, x):
        x = self.squeeze_layer(x)
        x_1x1 = self.expand_1x1_layer(x)
        x_3x3 = self.expand_3x3_layer(x)
        x = torch.cat((x_1x1, x_3x3), dim=1)
        return x

class SqueezeNetWithSkipConnections(nn.Module):
    def __init__(self, config, input_shape):
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

    def _get_ei_for_layer(self, i):
        incr_e = self.config['incr_e']
        base_e = self.config['base_e']
        freq = 2 #Residual structure expects freq=2
        pct_3x3 = self.config['pct_3x3']
        sr = self.config['sr']
        ei = base_e + (incr_e * np.floor((i / freq)))

        squeeze_1x1 = int(ei * sr)
        expand_1x1 = int(ei) - int(ei * pct_3x3)
        expand_3x3 = int(ei * pct_3x3)
        return squeeze_1x1, expand_1x1, expand_3x3, int(ei)
    
    def forward(self, x):
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
