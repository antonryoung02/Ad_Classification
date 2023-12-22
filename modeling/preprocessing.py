import numpy as np
import pandas as pd  
from PIL import Image
import os

        
def preprocess_image(image:Image, width:int, height:int) -> Image:
    image = image.resize(width, height)
    return image.convert("RGB")

def preprocess_data(input_dir:str, output_dir:str, dimensions:tuple):
    return None



#Load all files


