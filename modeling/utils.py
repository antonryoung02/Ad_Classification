import torch
from torch import nn
from datetime import datetime
import yaml
import coremltools as ct

def write_config_to_yaml(name:str, config:dict) -> None:
    """Writes the wandb run's config dictionary to a yaml file

    Args:
        name (str): The run's name accessed by wandb.init.run.name
        config (dict): The wandb.config dictionary
    """
    current_date = datetime.now().strftime('%Y-%m-%d')
    file_path = f'configs/{current_date}_{name}.yaml'
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
    out = traced_model(example_input)

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
    current_date = datetime.now().strftime('%Y-%m-%d')
    core_model.save(f"{filename}.mlpackage")
        
def save_as_pt(model:nn.Module, filename:str):
    state_dict = model.state_dict()
    torch.save(state_dict, f'{filename}.pt')