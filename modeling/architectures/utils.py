import torch
from torch import nn
  
def he_initialization(module:nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

            
class SEModule(nn.Module):
    """https://arxiv.org/pdf/1709.01507

        bottleneck_ratio r is reccomended to be 16, but not necessarily constant across layers
    """
    def __init__(self, in_channels:int, bottleneck_ratio:int):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(output_size=1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=int(in_channels / bottleneck_ratio)),
            nn.ReLU(),
            nn.Linear(in_features=int(in_channels / bottleneck_ratio), out_features=in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_squeeze = self.squeeze(x)
        x_squeeze = x_squeeze.view(x_squeeze.size(0), x_squeeze.size(1)) 
        # Linear layers reshape to (batch_size, features) and x is (batch_size, channels, h, w)
        x_excitation = self.excitation(x_squeeze).view(x.size(0), x.size(1), 1, 1)
        return x_excitation * x
    