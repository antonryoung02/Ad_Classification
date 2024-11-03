import torch
from typing import Tuple
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR 
from modeling.ModelInitializer import BaseModelInitializer
from modeling.architectures.utils import he_initialization

class ShuffleNetInitializer(BaseModelInitializer):
    """Initializer for ShuffleNet with hparams

    Expected keys in config:
        model (Module):
            model_groups: (1,2,3,4,5) only. The number of groups in the pointwise group convolutions. Inversly proportional to model complexity
            model_scale_factor: Scales all channels by this factor. Channels need to remain being divisible by model_groups
        optimizer (SGD):
            optimizer_lr: (0.5-0.1->0 with linear scheduler in paper)
            optimizer_momentum: (0.9 in paper)
            optimizer_weight_decay: Should be near zero (4e-5 in paper)
        scheduler (LinearLR):
            scheduler_start_factor
            scheduler_end_factor
        criterion (BCEWithLogitsLoss):
            criterion_pos_weight: positive class weight
    """
    expected_keys = {
        'model_groups', 'model_scale_factor', 'criterion_pos_weight', 
        'scheduler_start_factor', 'scheduler_end_factor', 'optimizer_lr',
        'optimizer_momentum', 'optimizer_weight_decay', 'num_epochs'
        }
    def __init__(self, config:dict):
        super().__init__(config, ShuffleNetInitializer.expected_keys)
        
    def get_scheduler(self, optimizer:Optimizer) -> LinearLR:
        """Paper decays LR linearly to zero over training"""
        start_factor = self.config['scheduler_start_factor']
        end_factor = self.config['scheduler_end_factor']
        total_iters = self.config['num_epochs'] + 1 # avoids last epoch lr=0
        scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
        return scheduler
    
    def get_model(self, input_shape: Tuple[int, int, int]) -> Module:
        model = ShuffleNet(self.config, input_shape)
        model.apply(he_initialization)
        return model
    
    def get_optimizer(self, model:nn.Module) -> Optimizer:
        lr = self.config['optimizer_lr']
        momentum = self.config['optimizer_momentum']
        weight_decay = self.config['optimizer_weight_decay']
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
    def get_criterion(self) -> _Loss:
        weight = [self.config['criterion_pos_weight']]
        pos_weight = torch.tensor(weight, dtype=torch.float).to(self.get_device())
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        
        
class ShuffleNet(Module):
    channels_for_group = {
        1: 144, 2:200, 3:240, 4:272, 5:384
    }

    def __init__(self, config:dict, input_shape:Tuple[int, int, int]):
        super(ShuffleNet, self).__init__()
        g = config['model_groups']
        s = config['model_scale_factor']
        
        c = ShuffleNet.channels_for_group[g]
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=int(24*s), kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.stage2 = nn.Sequential(
            ShuffleUnit(in_channels=int(24*s), out_channels=int(c*s), stride=2, groups=g),
            *[ShuffleUnit(in_channels=int(c*s), out_channels=int(c*s), stride=1, groups=g) for _ in range(3)]
        )
        self.stage3 = nn.Sequential(
            ShuffleUnit(in_channels=int(c*s), out_channels=int(2*c*s), stride=2, groups=g),
            *[ShuffleUnit(in_channels=int(2*c*s), out_channels=int(2*c*s), stride=1, groups=g) for _ in range(7)]
        )
        self.stage4 = nn.Sequential(
            ShuffleUnit(in_channels=int(2*c*s), out_channels=int(4*c*s), stride=2, groups=g),
            *[ShuffleUnit(in_channels=int(4*c*s), out_channels=int(4*c*s), stride=1, groups=g) for _ in range(3)]
        )
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1) 
        self.fc = nn.Linear(in_features=int(4*c*s), out_features=1)
        
    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
        
class ShuffleUnit(Module):
    """A ShuffleNet unit. Either stride=1 with residual add or stride=2 with residual concat"""
    def __init__(self, in_channels:int, out_channels:int, stride:int, groups:int):
        super().__init__()

        self.stride = stride
        if self.stride == 1:
            bottleneck_channels = in_channels
        else:
            bottleneck_channels = out_channels - in_channels

        self.residual_avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        self.group_pointwise_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=bottleneck_channels, kernel_size=1, padding=1, groups=groups)
        self.batchnorm1 = nn.BatchNorm2d(num_features=bottleneck_channels)
        self.relu1 = nn.ReLU()
        
        self.shuffle = ChannelShuffle(groups)
        self.depthwise_conv = nn.Conv2d(in_channels=bottleneck_channels, out_channels=bottleneck_channels, kernel_size=3, groups=bottleneck_channels, stride=self.stride)
        self.batchnorm2 = nn.BatchNorm2d(num_features=bottleneck_channels)
        
        self.group_pointwise_conv2 = nn.Conv2d(in_channels=bottleneck_channels, out_channels=bottleneck_channels, kernel_size=1, groups=groups)
        self.batchnorm3 = nn.BatchNorm2d(num_features=bottleneck_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            res = x
        else:
            res = self.residual_avgpool(x)
            
        x = self.group_pointwise_conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.shuffle(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.group_pointwise_conv2(x)
        x = self.batchnorm3(x)
        
        if self.stride == 1:
            x = x + res
        else:
            x = torch.cat([x, res], 1)

        x = self.relu2(x)
            
        return x
    
class ChannelShuffle(Module):
    """Shuffles channels by reshaping channels to (groups, n), then transposing, then flattening"""
    def __init__(self, groups):
        super().__init__()
        self.g = groups
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        n, c_out, h_out, w_out = x.shape
        x_reshaped = x.view(n, self.g, c_out // self.g, h_out, w_out)
        x_transposed = x_reshaped.transpose(1,2)
        return torch.flatten(x_transposed, start_dim=1, end_dim=2)
        
    
    
    
    
    
    
    
        