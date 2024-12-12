import torch
from torch import nn
from datetime import datetime
import yaml
import coremltools as ct
from typing import Any

def write_config_to_yaml(name:str, config:dict) -> None:
    """Writes the wandb run's config dictionary to a yaml file

    Args:
        name (str): The run's name accessed by wandb.init.run.name
        config (dict): The wandb.config dictionary
    """
    current_date = datetime.now().strftime('%Y-%m-%d')
    file_path = f'modeling/configs/{current_date}_{name}.yaml'
    with open(file_path, 'w') as yaml_file:
        yaml.dump(dict(config), yaml_file, default_flow_style=False)
        
def load_config(path:str) -> dict:
    """Loads the wandb-compatible yaml file to a dictionary

    Args:
        path (str): Path to the file

    Returns:
        dict: The converted dictionary
    """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config
    
def save_as_coreml(model:nn.Module, filename:str):
    model.eval()
    example_input = torch.rand(1, 3, 224, 224) 
    traced_model = torch.jit.trace(model, example_input)

    scale = 1.0 / (255.0 * 0.226)
    red_bias = -0.485 / 0.226
    green_bias = -0.456 / 0.226
    blue_bias = -0.406 / 0.226
    input = ct.ImageType(name='input', bias=[red_bias, green_bias, blue_bias], scale=scale, shape=(1, 3, 224, 224), color_layout='RGB')
    
    core_model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[input]
    )
    core_model.save(f"{filename}.mlpackage")
        
def save_as_pt(model:nn.Module, filename:str):
    state_dict = model.state_dict()
    torch.save(state_dict, f'{filename}.pt')

def add_tag_to_run(run:Any, tag:str):
    if run.tags:
        run.tags = run.tags + (tag,)


class OrdinalLogLoss(nn.Module):
    """https://paperswithcode.com/paper/a-simple-log-based-loss-function-for-ordinal-1"""
    def __init__(self, loss_matrix:torch.Tensor, alpha:torch.float=1.0): # d(i,j)for d(Ci, Cj). hardcoded in init for now
        self.loss_matrix = torch.tensor([
            [0.0, 1.0, 1.0, 1.0, 1.0],  # Row for ads
            [1.0, 0.0, 0.25, 0.5, 0.5],  # Row for baseball
            [1.0, 0.25, 0.0, 0.5, 0.5],  # Row for football
            [1.0, 0.5, 0.5, 0.0, 0.5],  # Row for basketball
            [1.0, 0.5, 0.5, 0.5, 0.0]   # Row for hockey
        ], dtype=torch.float32) ** alpha
        self.softmax = nn.Softmax();

    def forward(self, predicted:torch.Tensor, target:torch.Tensor): 
        """
        predicted: logits of output. Shape (N, C)
        target: true class. Shape (N,)
        self.loss_matrix: (C,C). Custom misclassification weight scaling. (loss_matrix[yi, yi] = 0)
        """
        p = self.softmax(predicted)  # (N, C)
        log_p = -torch.log(1 - p + 1e-9)  # prevents log(0). For numerical stability
        per_class_losses = (log_p) * torch.index_select(self.loss_matrix, 0, torch.flatten(target)) # (N, C) * (N,C) = (N, C)
        losses = torch.sum(per_class_losses, dim=1) # sum(N,C) along rows = (N)
        return losses.reshape((-1,1)) #(N,)


        