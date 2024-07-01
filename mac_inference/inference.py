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

MODEL_CHECKPOINT_PATH="./example.ckpt"
IMAGE_PATH="./images/image.png"
AD_SIGNAL_PATH="./ad_signal.txt"
POS_DIR="../raw_data/collected_data_pos"
NEG_DIR="../raw_data/collected_data_neg"
RECORD_DATA=False
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

def preprocess_image(image_path:str, dimensions:Tuple[int, int]) -> torch.Tensor:
    """Transforms to fit model input expectations"""
    width, height = dimensions
    transform = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = Image.open(image_path).resize((width, height)).convert("RGB")
    transformed_image = transform(image)
    transformed_image = transformed_image.unsqueeze(0)
    return transformed_image


def run_inference(model:pl.LightningModule, record_data:bool, image_path:str) -> float:
    """>0.5 predicts advertisement, <0.5 predicts hockey

    param model: torch.nn.Module pytorch model
    param image_path: Path to image taken in classify_script.sh
    param record: Toggles data collection. True saves image to data directory
    """
    image = preprocess_image(image_path, (224, 224))
    prediction = model.predict_step(image)

    if record_data:
        unique_filename = f"{time.strftime('%Y%m%d_%H%M%S')}_image.png"
        new_dir = POS_DIR if prediction > 0.5 else NEG_DIR
        new_path = os.path.join(new_dir, unique_filename)
        shutil.move(image_path, new_path)
    else:
        os.remove(image_path)
    print(prediction)
    return prediction


def main():
    model = CNN.load_from_checkpoint(checkpoint_path=MODEL_CHECKPOINT_PATH, config=CONFIG)
    model.eval()

    while os.path.exists("./classify_script_running"):
        if os.path.exists(IMAGE_PATH):
            time.sleep(1)
            output = run_inference(model, RECORD_DATA, IMAGE_PATH)
            if output > 0.5:
                prediction = "True"
            else:
                prediction = "False"

            with open(AD_SIGNAL_PATH, "w") as file:
                file.write(prediction)

if __name__ == "__main__":
    main()
