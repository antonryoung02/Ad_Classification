from typing import Optional
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR
from modeling.ModelInitializer import BaseModelInitializer
from modeling.architectures.utils import SEModule, he_initialization

class GhostNetInitializer(BaseModelInitializer): 
    """
    Initializer for GhostNet with hparams

    Expected keys in config:
        model (GhostNet):
            model_width_multiplier: (0,1] scales in_channels and out_channels of all layers
            model_kernel_size: [3,5] sets size of kernel in ghost map layers
            model_ghost_ratio: [2,4] sets ratio of non-ghost maps to ghost maps 1:ratio
            model_se_ratio: [~16] sets compression ratio in SE FC layers
        optimizer (RMSProp):
            optimizer_lr
            optimizer_momentum
            optimizer_weight_decay
        criterion (BCEWithLogitsLoss):
            criterion_pos_weight: positive class weight
    """
    expected_keys = {"optimizer_lr", "optimizer_momentum", "optimizer_weight_decay", 
                     "criterion_pos_weight", "model_ghost_ratio", "model_kernel_size", 
                     "model_width_multiplier", "model_se_ratio"} 
     
    def __init__(self, config:dict):
        super().__init__(config, GhostNetInitializer.expected_keys)

    def get_model(self, input_shape:tuple) -> nn.Module:
        model = GhostNet(self.config, input_shape)
        model.apply(he_initialization)
        return model
        
    def get_scheduler(self, optimizer:Optimizer) -> Optional[LinearLR]:
        start_factor = self.config['scheduler_start_factor']
        end_factor = self.config['scheduler_end_factor']
        total_iters = self.config['num_epochs'] + 1 # avoids last epoch having lr=0
        scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
        return scheduler
    
    def get_optimizer(self, model:nn.Module) -> Optimizer:
        lr = self.config['optimizer_lr']
        momentum = self.config['optimizer_momentum']
        weight_decay = self.config['optimizer_weight_decay']
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    def get_criterion(self) -> _Loss:
        weight = [self.config['criterion_pos_weight']]
        pos_weight = torch.tensor(weight, dtype=torch.float).to(self.get_device())
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    
class GhostNet(nn.Module):
    def __init__(self, config:dict, input_shape:tuple):
        super().__init__()
        self.config = config
        self.input_shape = input_shape
        s = config['model_ghost_ratio']
        d = config['model_kernel_size']
        p = config['model_width_multiplier']
        r = config['model_se_ratio']
        
        # 9.6 million parameters at p = 1
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=int(16 * p), kernel_size=3, stride=2),  #224
            nn.BatchNorm2d(num_features=int(16 * p)),
            nn.ReLU(),
            
            GhostBottleneck(in_channels=int(16 * p), exp_channels=int(16 * p), out_channels=int(16 * p), stride=1, s=s, d=d, se_module=False, se_ratio=r),  #112
            GhostBottleneck(in_channels=int(16 * p), exp_channels=int(48 * p), out_channels=int(24 * p), stride=2, s=s, d=d, se_module=True, se_ratio=r),

            GhostBottleneck(in_channels=int(24 * p), exp_channels=int(72 * p), out_channels=int(24 * p), stride=1, s=s, d=d, se_module=False, se_ratio=r),  #56
            GhostBottleneck(in_channels=int(24 * p), exp_channels=int(72 * p), out_channels=int(40 * p), stride=2, s=s, d=d, se_module=True, se_ratio=r),
            
            GhostBottleneck(in_channels=int(40 * p), exp_channels=int(120 * p), out_channels=int(40 * p), stride=1, s=s, d=d, se_module=True, se_ratio=r),  #28
            GhostBottleneck(in_channels=int(40 * p), exp_channels=int(240 * p), out_channels=int(80 * p), stride=2, s=s, d=d, se_module=False, se_ratio=r),
            
            GhostBottleneck(in_channels=int(80 * p), exp_channels=int(200 * p), out_channels=int(80 * p), stride=1, s=s, d=d, se_module=False, se_ratio=r),  #14
            GhostBottleneck(in_channels=int(80 * p), exp_channels=int(184 * p), out_channels=int(80 * p), stride=1, s=s, d=d, se_module=False, se_ratio=r),
            GhostBottleneck(in_channels=int(80 * p), exp_channels=int(184 * p), out_channels=int(80 * p), stride=1, s=s, d=d, se_module=False, se_ratio=r),
            GhostBottleneck(in_channels=int(80 * p), exp_channels=int(480 * p), out_channels=int(112 * p), stride=1, s=s, d=d, se_module=True, se_ratio=r),
            GhostBottleneck(in_channels=int(112 * p), exp_channels=int(672 * p), out_channels=int(112 * p), stride=1, s=s, d=d, se_module=True, se_ratio=r),
            GhostBottleneck(in_channels=int(112 * p), exp_channels=int(672 * p), out_channels=int(160 * p), stride=2, s=s, d=d, se_module=True, se_ratio=r),
            
            GhostBottleneck(in_channels=int(160 * p), exp_channels=int(960 * p), out_channels=int(160 * p), stride=1, s=s, d=d, se_module=False, se_ratio=r),  #7
            GhostBottleneck(in_channels=int(160 * p), exp_channels=int(960 * p), out_channels=int(160 * p), stride=1, s=s, d=d, se_module=True, se_ratio=r),
            GhostBottleneck(in_channels=int(160 * p), exp_channels=int(960 * p), out_channels=int(160 * p), stride=1, s=s, d=d, se_module=False, se_ratio=r),
            GhostBottleneck(in_channels=int(160 * p), exp_channels=int(960 * p), out_channels=int(160 * p), stride=1, s=s, d=d, se_module=True, se_ratio=r),
            
            nn.Conv2d(in_channels=int(160 * p), out_channels=int(960 * p), kernel_size=1),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=int(960 * p), out_channels=int(1280 * p), kernel_size=1),  #1
            nn.Flatten(),
            nn.Linear(in_features=int(1280 * p), out_features=1)
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.module(x)
        
    
class GhostBottleneck(nn.Module):
    def __init__(self, in_channels:int, exp_channels:int, out_channels:int, stride:int, s:int, d:int, se_module:bool, se_ratio:int):
        """Implementation of a ghost bottleneck layer

        Args:
            in_channels (int): Channels entering the  layer
            exp_channels (int): Channels created by the inverse bottleneck
            out_channels (int): Channels exiting the layer
            stride (int): If stride != 1, adds a downsampling depthwise conv to reduce spatial dimension
            s (int): Sets ghost ratio for ghost modules
            d (int): Sets ghost filter size for ghost modules
            se_module (bool): If true, adds a squeeze-excite block in the bottleneck path
            se_ratio (int): Sets the ratio of input:bottleneck nodes in the module as se_ratio:1
        """
        super().__init__()
        
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential( # Depthwise separable convolution without nonlinearities
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=stride,
                    groups=in_channels,
                    padding=1
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
            
        self.ghost1 = GhostModule(in_channels=in_channels, out_channels=exp_channels, s=s, d=d, has_relu=True)
        
        if stride == 1:
            self.depthwise_conv = nn.Identity()
        else: # If stride == 2, add depthwise conv to reduce spatial dimension
            self.depthwise_conv = nn.Sequential(
                nn.Conv2d(in_channels=exp_channels, out_channels=exp_channels, kernel_size=3, stride=stride, groups=exp_channels, padding=1),
                nn.BatchNorm2d(num_features=exp_channels)
            )
        if se_module: # Some bottleneck layers add se module between the ghost modules
            self.se = SEModule(in_channels=exp_channels, bottleneck_ratio=se_ratio)
        else:
            self.se = nn.Identity()
                                         
        self.ghost2 = GhostModule(in_channels=exp_channels, out_channels=out_channels, s=s, d=d, has_relu=False)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_shortcut = self.shortcut(x)
        x = self.ghost1(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.ghost2(x)
        return x + x_shortcut
    
class GhostModule(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, s:int, d:int, has_relu:bool):
        """Implementation of a ghost module

        Args:
            in_channels (int): channels entering ghost module
            out_channels (int): channels exiting ghost module
            s (int): the ratio of primary channels:ghost channels is 1:s
            d (int): kernel size of the ghost convolutions
            has_relu (bool): 1st ghost module in bottleneck uses relu activations, 2nd module does not
        """
        super().__init__()
        m = int(out_channels / s)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=m, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=m),
            nn.ReLU() if has_relu else nn.Identity()
        )
        self.ghost_operation = nn.Sequential(
            nn.Conv2d(in_channels=m, out_channels=out_channels-m, kernel_size=d, padding=int((d-1)/2), groups=m),
            nn.BatchNorm2d(num_features=out_channels-m),
            nn.ReLU() if has_relu else nn.Identity()
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.primary_conv(x)
        x_ghost = self.ghost_operation(x)
        return torch.cat([x, x_ghost], dim=1) # x is the identity ghost mapping
    
    
    
        