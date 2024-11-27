import torch
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from typing import Optional
from torch.optim.lr_scheduler import StepLR
from modeling.ModelInitializer import BaseModelInitializer
from torchvision.transforms import v2

#TODO update to V3
class MobileNetInitializer(BaseModelInitializer):
    """
    Initializer for MobileNet with hparams

    Expected keys in config:
        model (MobileNet):
            model_width_multiplier: (0,1] scales in_channels and out_channels of all layers
            model_resolution_multiplier: (0,1] resizes input image by (p*width, p*height)
        optimizer (RMSProp):
            optimizer_lr
            optimizer_momentum
            optimizer_weight_decay
            optimizer_alpha
        criterion (CrossEntropyLoss):
            criterion_class_weights
    """
    
    expected_keys = {'optimizer_lr', 'optimizer_alpha', 
                     'optimizer_momentum', 'optimizer_weight_decay', 
                     'criterion_class_weights', 'model_width_multiplier', 
                     'model_resolution_multiplier'}
    
    def __init__(self, config:dict):
        super().__init__(config, MobileNetInitializer.expected_keys)

    def get_model(self, input_shape:tuple) -> nn.Module:
        return MobileNet(self.config, input_shape, num_classes=5)
    
    def get_scheduler(self, optimizer:Optimizer) -> Optional[StepLR]:
        return None
    
    def get_optimizer(self, model:nn.Module) -> Optimizer:
        """
        Typical optimizer intitialization, but only applies weight decay to non-depthwise layers
        as recommended in paper.
        """
        lr = self.config['optimizer_lr']
        alpha = self.config['optimizer_alpha']
        momentum = self.config['optimizer_momentum']
        weight_decay = self.config['optimizer_weight_decay']
       
        # https://pytorch.org/docs/stable/optim.html 
        depthwise_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'depthwise_conv' in name:
                depthwise_params.append(param)
            else:
                other_params.append(param)

        return optim.RMSprop(
            [
                {'params': depthwise_params, 'weight_decay': 0.0},
                {'params': other_params, 'weight_decay': weight_decay}
            ],
            lr=lr, alpha=alpha, momentum=momentum
        )
    
    def get_criterion(self) -> _Loss:
        weight = torch.tensor(self.config['criterion_class_weights'], dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)

        
class MobileNet(nn.Module):
    def __init__(self, config:dict, input_shape:tuple, num_classes:int):
        super().__init__()

        a = config['model_width_multiplier']
        p = config['model_resolution_multiplier']
    
        self.scale_by_resolution = v2.Resize((int(p*input_shape[1]), int(p*input_shape[2])))
        
        self.standard_conv = nn.Conv2d(in_channels=3, out_channels=int(32 * a), kernel_size=3)
        self.dws_conv1 = DepthwiseSeparableConvolution(in_channels=int(32*a), out_channels=int(64*a), depthwise_stride=1)
        self.dws_conv2 = DepthwiseSeparableConvolution(in_channels=int(64*a), out_channels=int(128*a), depthwise_stride=2)
        self.dws_conv3 = DepthwiseSeparableConvolution(in_channels=int(128*a), out_channels=int(128*a), depthwise_stride=1)
        self.dws_conv4 = DepthwiseSeparableConvolution(in_channels=int(128*a), out_channels=int(256*a), depthwise_stride=2)
        self.dws_conv5 = DepthwiseSeparableConvolution(in_channels=int(256*a), out_channels=int(256*a), depthwise_stride=1)
        self.dws_conv6 = DepthwiseSeparableConvolution(in_channels=int(256*a), out_channels=int(512*a), depthwise_stride=2)
        
        self.dws_conv7_11 = nn.Sequential(
            *[DepthwiseSeparableConvolution(in_channels=int(512*a), out_channels=int(512*a), depthwise_stride=1) for _ in range(5)]
        ) 
        
        self.dws_conv12 = DepthwiseSeparableConvolution(in_channels=int(512*a), out_channels=int(1024*a), depthwise_stride=2)
        self.dws_conv13 = DepthwiseSeparableConvolution(in_channels=int(1024*a), out_channels=int(1024*a), depthwise_stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, ceil_mode=True)
        self.fc = nn.Linear(in_features=int(1024*a), out_features=num_classes)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.scale_by_resolution(x)
        x = self.standard_conv(x)
        x = self.dws_conv1(x)
        x = self.dws_conv2(x)
        x = self.dws_conv3(x)
        x = self.dws_conv4(x)
        x = self.dws_conv5(x)
        x = self.dws_conv6(x)
        x = self.dws_conv7_11(x)
        x = self.dws_conv12(x)
        x = self.dws_conv13(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    
class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, depthwise_stride:int):
        """Creates a depthwise separable convolution as used in the MobileNet architectures.

        Args:
            in_channels (int): number of input channels (adjusted for width multiplier a)
            out_channels (int): number of output channels (adjusted for width multiplier a)
            depthwise_stride (int): stride for the depthwise convolution (1 or 2)
        """
        super(DepthwiseSeparableConvolution, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=in_channels, 
                kernel_size=3, 
                stride=depthwise_stride, 
                groups=in_channels
                ),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
                kernel_size=1
                ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.module(x)