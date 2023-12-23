import numpy as np
import pandas as pd  
from PIL import Image
import os

        
def preprocess_image(image: Image, dimensions: tuple) -> Image:
    """
    Converts an Image object to desired size / #channels
    """

    width, height = dimensions
    resized_image = image.resize((width, height))
    return resized_image.convert("RGB")

def preprocess_data(input_dir:str, output_dir:str, dimensions:tuple=(320,180)) -> None:
    """
    Transforms all .png files to desired specifications using preprocess_image

    param input_dir: the directory of untransformed images
    param output_dir: where transformed images are stored
    param dimensions: (width, height) in pixels of transformed image size
    """

    if not os.path.exists(output_dir):
        return
    if not os.path.exists(input_dir):
        return

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path)
            processed_image = preprocess_image(image, dimensions)
            
            output_path = os.path.join(output_dir, filename)
            processed_image.save(output_path)

def process_training_data():
    """
    Function that processes both my pos and neg data
    """
    pos_input = "./data_pos"
    pos_output = "./processed_data/data_pos"
    neg_input = "./data_neg"
    neg_output = "./processed_data/data_neg"

    preprocess_data(pos_input, pos_output)
    preprocess_data(neg_input, neg_output)

process_training_data()


