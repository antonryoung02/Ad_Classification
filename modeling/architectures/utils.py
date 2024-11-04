import torch
from torch import nn
  
def he_initialization(module:nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

            
class SEModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x
    