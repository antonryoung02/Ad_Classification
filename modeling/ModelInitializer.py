import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from typing import Tuple, Optional
from torch.optim.lr_scheduler import StepLR

class BaseModelInitializer:
    """Base Class for initializing a CNN-compatible model and its criterion, optimizer, and learning rate scheduler"""
    def __init__(self, config:dict, expected_keys:set):
        self._validate_config(config, expected_keys)
        self.config = config
        
    def initialize_model_crit_opt_sched(self, input_shape: Tuple[int, int, int]) -> Tuple[nn.Module, _Loss, Optimizer, StepLR | None]:
        criterion = self.get_criterion()
        model = self.get_model(input_shape)
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer)
        return model, criterion, optimizer, scheduler
    
    def get_criterion(self) -> _Loss:
        raise NotImplementedError()
         
    def get_optimizer(self, model:nn.Module) -> Optimizer:
        raise NotImplementedError()
    
    def get_model(self, input_shape) -> nn.Module:
        raise NotImplementedError()
    
    def get_scheduler(self, optimizer:Optimizer) -> Optional[StepLR]:
        raise NotImplementedError()
    
    def get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _validate_config(self, config:dict, expected_keys:set):
        for key in expected_keys:
            if key not in config:
                raise ValueError(f"required key {key} does not exist in config")
