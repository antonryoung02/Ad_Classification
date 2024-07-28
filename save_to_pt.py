import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import time
import shutil
import torch
from torchvision.transforms import v2
from PIL import Image
from torch import nn
from modeling.utils import SqueezeNetWithSkipConnections
from modeling.CNN import CNN
from modeling.ModelInitializer import SqueezeNetInitializer
from typing import Tuple
import pytorch_lightning as pl


MODEL_CHECKPOINT_PATH="./mac_inference/example.ckpt"
CONFIG = {
    "base_e": 128,
    "batch_size":256,
    "dropout":0.06482,
    "incr_e":96,
    "initializer":"squeezenet",
    "lr":0.01786,
    "lr_gamma":0.3295,
    "num_epochs":5,
    "pct_3x3":0.5,
    "sr":0.25,
    "weight_decay":0.001374
}

model = CNN(config=CONFIG).load_from_checkpoint(checkpoint_path=MODEL_CHECKPOINT_PATH)
state_dict = model.network.state_dict()
torch.save(state_dict, 'model.pt')