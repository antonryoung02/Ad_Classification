import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from typing import Tuple, Optional
from torch.optim.lr_scheduler import StepLR
from modeling.ModelInitializer import BaseModelInitializer
import numpy as np
from modeling.architectures.utils import he_initialization

class SqueezeNetInitializer(BaseModelInitializer):
    """
    Initializer for Squeezenet with hparams

    Expected keys in config:
        model (SqueezeNetWithSkipConnections):
            model_dropout: Dropout2d layer dropout rate
            model_pct_3x3: The proportion of 3x3 filters in the expand layer. Value should be around 0.5
            model_sr: The 'squeeze ratio', the ratio of squeeze filters to expand filters. Authors reccomend 0.125-0.75
            model_base_e: The number of expand filters in the first fire module
            model_incr_e: The number of expand filters added every 2 fire modules
        optimizer (SGD):
            optimizer_lr
            optimizer_momentum
            optimizer_weight_decay
        scheduler (StepLR):
            scheduler_gamma
            scheduler_step_size
        criterion (CrossEntropyLoss):
            criterion_class_weights 
    """
    
    expected_keys = {'scheduler_gamma', 'scheduler_step_size',
                     'criterion_class_weights', 'optimizer_lr',
                     'optimizer_momentum', 'optimizer_weight_decay', 
                     'model_dropout', 'model_incr_e', 'model_base_e',
                     'model_pct_3x3', 'model_sr'}
    
    def __init__(self, config:dict):
        super().__init__(config, SqueezeNetInitializer.expected_keys)

    def get_model(self, input_shape:Tuple[int, int, int]) -> nn.Module:
        model = SqueezeNetWithSkipConnections(self.config, input_shape, num_classes=5)
        model.apply(he_initialization)
        return model
    
    def get_scheduler(self, optimizer:Optimizer) -> StepLR:
        gamma = self.config['scheduler_gamma']
        step_size = self.config['scheduler_step_size']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return scheduler

    def get_criterion(self) -> _Loss:
        weight = torch.tensor(self.config['criterion_class_weights'], dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)
    
    def get_optimizer(self, model:nn.Module) -> Optimizer:
        #Adam would not converge. But haven't tried with batch norm in the fire module
        lr = self.config['optimizer_lr']
        momentum = self.config['optimizer_momentum']
        weight_decay = self.config['optimizer_weight_decay']
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    
class SqueezeNetWithSkipConnections(nn.Module):
    """Squeezenet architecture with simple residual connections between fire modules"""
    def __init__(self, config:dict, input_shape:Tuple[int, int, int], num_classes:int):
        """
        Initializes the model layers      

        Args:
            config (dict): Contains the hparam keys 'dropout', 'incr_e', 'base_e', 'pct_3x3', 'sr', 
            input_shape (tuple): Input shape of one image (num channels, image height, image width)
        """
        super().__init__()
        self.config = config
        self.dropout_rate = config['model_dropout']
        self.incr_e = self.config['model_incr_e']
        self.base_e = self.config['model_base_e']
        self.pct_3x3 = self.config['model_pct_3x3']
        self.sr = self.config['model_sr']

        self.stem = nn.Sequential(
            nn.Conv2d(input_shape[0], 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.dropout = nn.Dropout(self.dropout_rate)
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
        self.final_conv = nn.Conv2d(ei7, num_classes, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def _get_ei_for_layer(self, i:int):
        freq = 2 #Residual connections structure expects freq=2
        ei = self.base_e + (self.incr_e * np.floor((i / freq)))

        squeeze_1x1 = int(ei * self.sr)
        expand_1x1 = int(ei) - int(ei * self.pct_3x3)
        expand_3x3 = int(ei * self.pct_3x3)
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
        return x.view(x.size(0), -1)  # output shape is (batch_size, num_classes)
    
class FireModule(nn.Module):
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

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.squeeze_layer(x)
        x_1x1 = self.expand_1x1_layer(x)
        x_3x3 = self.expand_3x3_layer(x)
        x = torch.cat((x_1x1, x_3x3), dim=1)
        return x

