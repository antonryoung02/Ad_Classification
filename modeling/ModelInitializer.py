import torch
import torch.nn as nn
import torch.optim as optim
from modeling.utils import find_fc_layer_shape, SqueezeNetWithSkipConnections

class DefaultModelInitializer:
    def __init__(self, config):
        self.config = config

    def initialize_model_crit_opt_sched(self, input_shape):
        pass

    def get_model(self, input_shape):
        pass

    def get_criterion(self):
        pass

    def get_optimizer(self, model):
        pass

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SqueezeNetInitializer(DefaultModelInitializer):
    """
    Initializes the SqueezeNet architecture

    expected keys in self.config (defaults from SqueezeNet paper):
    dropout: 0.5
    pct_3x3: 0.5
    sr: 0.125
    base_e: 128 (probably need to be hardcoded to work)
    incr_e: 128 (probably need to be hardcoded to work)
    freq: 2 (probably need to be hardcoded to work)
    """
    def initialize_model_crit_opt_sched(self, input_shape):
        model = self.get_model(input_shape)
        model.apply(self._initialize_weights)
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer)

        return model, self.get_criterion(), optimizer, scheduler
    
    def _initialize_weights(self, module):
        """He initialization"""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_model(self, input_shape):
        network =  SqueezeNetWithSkipConnections(self.config, input_shape)
        return network
    
    def get_scheduler(self, optimizer):
        lr_gamma = self.config['lr_gamma'] if 'lr_gamma' in self.config.keys() else 0.1
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=lr_gamma)
        return scheduler

    def get_criterion(self):
        # Squeezenet seems to be sensitive to class imbalances
        class_weights = torch.tensor([1.5], dtype=torch.float).to(self.get_device())
        return nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    def get_optimizer(self, model):
        #return optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        return optim.SGD(model.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=self.config['weight_decay'])

class SimpleCNNInitializer:
    def __init__(self, config):
        self.config = config

    def initialize_model_crit_opt_sched(self, input_shape):
        model = self.get_model(input_shape)
        return model, self.get_criterion(), self.get_optimizer(model), None
    
    def get_model(self, input_shape):
        kernels = self.config['kernels']
        dropout_rate = self.config['dropout']
        fc_units = self.config['fc_units']
        network =  nn.Sequential(
            nn.Conv2d(3, kernels[0], kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate), 
            nn.Conv2d(kernels[0], kernels[1], kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(kernels[1], kernels[2], kernel_size=3),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(kernels[2], kernels[3], kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten()
        )
        network.add_module("fc", nn.Linear(find_fc_layer_shape(network, input_shape), fc_units))
        network.add_module("fc2", nn.Linear(fc_units,1))
        return network
    
    def get_criterion(self):
        return nn.BCEWithLogitsLoss()
    
    def get_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
    
    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")