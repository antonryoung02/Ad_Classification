import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from typing import Tuple, Optional
from torch.optim.lr_scheduler import StepLR
from modeling.architectures.MobileNet import MobileNetInitializer
from modeling.architectures.SqueezeNet import SqueezeNetInitializer

class BaseModelInitializer:
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

class ModelInitializerFactory:
    def __call__(self, config:dict) -> BaseModelInitializer:
        """Returns a ModelInitializer depending on the value of config['model_initializer']

        Args:
            config (dict): Must contain 'model_initializer' key. Supported values: 'squeezenet', 'mobilenet'

        Returns:
            BaseModelInitializer: Allows CNN class to initialize different architectures
        """
        if 'model_initializer' not in config.keys():
            raise KeyError('model_initializer not found in config')
        
        match config['model_initializer']:
            case "squeezenet":
                return SqueezeNetInitializer(config)
            case "mobilenet":
                return MobileNetInitializer(config)
            case _:
                raise ValueError(f"model type {config['model_initializer']} does not exist!")