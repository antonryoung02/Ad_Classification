import torch
import torch.nn as nn
import torch.optim as optim
from utils import find_fc_layer_shape

class ModelInitializer:
    def __init__(self, config):
        self.config = config

    def initialize_model_crit_opt(self, input_shape):
        model = self.get_model(input_shape)
        return model, self.get_criterion(), self.get_optimizer(model)
    
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

    def get_callbacks(self):
        pass
    
    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")