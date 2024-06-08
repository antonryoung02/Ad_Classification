from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from SimpleCNN import SimpleCNN

class Configurator:
    def __init__(self, config: Dict):
        self.config = config
    
    def get_model(self):
        return SimpleCNN(self.config['kernels'], self.config['fc_units'], self.config['dropout'])
    
    def get_criterion(self):
        return nn.BCEWithLogitsLoss()
    
    def get_optimizer(self, model):
        return optim.Adam(model.network.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
    
    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")