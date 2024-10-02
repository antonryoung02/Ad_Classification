import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from modeling.utils import SqueezeNetWithSkipConnections, SimpleCNN
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict
from torch.optim.lr_scheduler import StepLR
class AbstractModelInitializer(ABC):
    """Interface needed to add a new Model Initializer"""
    @abstractmethod
    def initialize_model_crit_opt_sched(self, input_shape:Tuple[int, int, int]) -> None:
        pass

class SqueezeNetInitializer(AbstractModelInitializer):
    """
    Initializer for Squeezenet with hparams

    Expected keys in config:
        dropout: Dropout2d layer dropout rate
        pct_3x3: The proportion of 3x3 filters in the expand layer. Value should be around 0.5
        sr: The 'squeeze ratio', the ratio of squeeze filters to expand filters. Authors reccomend 0.125-0.75
        base_e: The number of expand filters in the first fire module
        incr_e: The Number of expand filters added every 2 fire modules
    """
    def __init__(self, config:dict):
        self.config = config

    def initialize_model_crit_opt_sched(self, input_shape:Tuple[int, int, int]) -> Tuple[nn.Module, _Loss, Optimizer, Optional[StepLR]]:
        model = self.get_model(input_shape)
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer)
        return model, self.get_criterion(), optimizer, scheduler
    
    def _initialize_weights(self, module:nn.Module) -> None:
        """He initialization"""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_model(self, input_shape:Tuple[int, int, int]) -> nn.Module:
        model = SqueezeNetWithSkipConnections(self.config, input_shape)
        model.apply(self._initialize_weights)
        return model
    
    def get_scheduler(self, optimizer:Optimizer) -> StepLR:
        lr_gamma = self.config['lr_gamma']
        scheduler = StepLR(optimizer, step_size=5, gamma=lr_gamma)
        return scheduler

    def get_criterion(self) -> _Loss:
        # Squeezenet seems to be sensitive to class imbalances
        class_weights = torch.tensor([0.7], dtype=torch.float).to(self.get_device())
        return nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    def get_optimizer(self, model:nn.Module) -> Optimizer:
        #Adam would not converge. But haven't tried with batch norm in the fire module
        return optim.SGD(model.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=self.config['weight_decay'])

    def get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SimpleCNNInitializer(AbstractModelInitializer):
    """
    Initializer for SimpleCNN with hparams

    Expected keys in config:
        dropout: Dropout2d layer dropout rate
        pct_3x3: The proportion of 3x3 filters in the expand layer. Value should be around 0.5
        sr: The 'squeeze ratio', the ratio of squeeze filters to expand filters. Authors reccomend 0.125-0.75
        base_e: The number of expand filters in the first fire module
        incr_e: The Number of expand filters added every 2 fire modules
    """
    def __init__(self, config:dict):
        self.config = config

    def initialize_model_crit_opt_sched(self, input_shape:Tuple[int, int, int]) -> Tuple[nn.Module, _Loss, Optimizer, Optional[_LRScheduler]]:
        model = self.get_model(input_shape)
        return model, self.get_criterion(), self.get_optimizer(model), None
    
    def get_model(self, input_shape:Tuple[int, int, int]) -> nn.Module:
        model = SimpleCNN(self.config, input_shape)
        return model
    
    def get_criterion(self) -> _Loss: 
        return nn.BCEWithLogitsLoss()

    def get_optimizer(self, model:nn.Module) -> Optimizer:
        return optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])

class ModelInitializerFactory:
    def get_initializer(self, config:dict) -> Optional[AbstractModelInitializer]:
        """Returns a ModelInitializer depending on the value of config['initializer']

        Args:
            config (dict): Must contain 'initializer' key. Supported values: 'simple_cnn', 'squeezenet'

        Returns:
            BaseModelInitializer: Allows CNN class to initialize different architectures
        """
        if 'initializer' not in config.keys():
            raise KeyError('initializer not found in config')
        
        if config['initializer'] == "simple_cnn":
            return SimpleCNNInitializer(config)
        elif config['initializer'] == "squeezenet":
            return SqueezeNetInitializer(config)
        else:
            return None