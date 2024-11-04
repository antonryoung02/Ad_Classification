import torch
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from typing import Tuple, Optional
from torch.optim.lr_scheduler import StepLR
from modeling.ModelInitializer import BaseModelInitializer
from torchvision.transforms import v2
from utils import SEModule

class GhostNetInitializer(BaseModelInitializer):
    expected_keys = set()
    
    def __init__(self, config:dict):
        super().__init__(config, GhostNetInitializer.expected_keys)

    def get_model(self, input_shape:tuple) -> nn.Module:
        return GhostNet(self.config, input_shape)
    
    def get_scheduler(self, optimizer:Optimizer) -> Optional[StepLR]:
        raise NotImplementedError()
    
    def get_optimizer(self, model:nn.Module) -> Optimizer:
        raise NotImplementedError()
    
    def get_criterion(self) -> _Loss:
        raise NotImplementedError()
    
    
class GhostNet(nn.Module):
    def __init__(self, config:dict, input_shape:tuple):
        super().__init__()
        self.config = config
        self.input_shape = input_shape
        s = config['model_ghost_ratio']
        d = config['model_kernel_size']
        p = config['model_width_multipler']
        
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2),
        
            GhostBottleneck(in_channels=16, exp_channels=16, out_channels=16, stride=1, s=s, d=d),
            GhostBottleneck(in_channels=16, exp_channels=48, out_channels=24, stride=2, s=s, d=d),

            GhostBottleneck(in_channels=24, exp_channels=72, out_channels=24, stride=1, s=s, d=d),
            GhostBottleneck(in_channels=24, exp_channels=72, out_channels=40, stride=2, s=s, d=d),
            SEModule(),
        
            GhostBottleneck(in_channels=40, exp_channels=120, out_channels=40, stride=1, s=s, d=d),
            SEModule(),
            GhostBottleneck(in_channels=40, exp_channels=240, out_channels=80, stride=2, s=s, d=d),
        
            GhostBottleneck(in_channels=80, exp_channels=200, out_channels=80, stride=1, s=s, d=d),
            GhostBottleneck(in_channels=80, exp_channels=184, out_channels=80, stride=1, s=s, d=d),
            GhostBottleneck(in_channels=80, exp_channels=184, out_channels=80, stride=1, s=s, d=d),
            GhostBottleneck(in_channels=80, exp_channels=480, out_channels=112, stride=1, s=s, d=d),
            SEModule(),
            GhostBottleneck(in_channels=112, exp_channels=672, out_channels=112, stride=1, s=s, d=d),
            SEModule(),
            GhostBottleneck(in_channels=112, exp_channels=672, out_channels=160, stride=2, s=s, d=d),
            SEModule(),
        
            GhostBottleneck(in_channels=160, exp_channels=960, out_channels=160, stride=1, s=s, d=d),
            GhostBottleneck(in_channels=160, exp_channels=960, out_channels=160, stride=1, s=s, d=d),
            SEModule(),
            GhostBottleneck(in_channels=160, exp_channels=960, out_channels=160, stride=1, s=s, d=d),
            GhostBottleneck(in_channels=160, exp_channels=960, out_channels=160, stride=1, s=s, d=d),
            SEModule(),
        
            nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1),
            nn.Linear(in_features=1280, out_features=1)
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.module(x)
        
    
class GhostBottleneck(nn.Module):
    def __init__(self, in_channels:int, exp_channels:int, out_channels:int, stride:int, s:int, d:int):
        super().__init__()
        
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=2,
                    groups=in_channels
                    ),
                nn.BatchNorm2d(num_features=in_channels),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,
                    kernel_size=1
                    ),
                nn.BatchNorm2d(num_features=out_channels),                    
                )
            
        self.ghost1 = GhostModule(in_channels=in_channels, out_channels=exp_channels, s=s, d=d)
        self.batchnorm1 = nn.BatchNorm2d(num_features=exp_channels)
        self.relu1 = nn.ReLU()
        
        if stride == 1:
            self.dw_conv = nn.Identity()
        else:
            self.dw_conv = nn.Conv2d(in_channels=exp_channels, out_channels=exp_channels, kernel_size=3, stride=stride, groups=exp_channels)

        self.ghost2 = GhostModule(in_channels=exp_channels, out_channels=out_channels, s=s, d=d)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_shortcut = self.shortcut(x)
        x = self.ghost1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.dw_conv(x)
        x = self.ghost2(x)
        x = self.batchnorm2(x)
        return x + x_shortcut
    
class GhostModule(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, s:float, d:int):
        super().__init__()
        m = int(out_channels / s)
        self.primary_conv = nn.Conv2d(in_channels=in_channels, out_channels=m, kernel_size=3, padding=1)
        self.ghost_operation = nn.Conv2d(in_channels=m, out_channels=out_channels-m, kernel_size=d, padding=int((d-1)/2), groups=m)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.primary_conv(x)
        x_ghost = self.ghost_operation(x)
        return torch.cat([x, x_ghost], dim=1)
    
    
    
        