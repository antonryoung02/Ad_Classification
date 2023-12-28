import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models import SimpleCNN
import sys
import os
import time
import shutil

def preprocess_image(image_path, dimensions):
    """Transforms to fit model input expectations"""
    width, height = dimensions
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).resize((width, height)).convert('RGB')
    transformed_image = transform(image).unsqueeze(0)
    return transformed_image

def run_inference(model, image_path):
    """>0.5 predicts advertisement, <0.5 predicts hockey"""
    pos_dir = "/home/antonryoung02/raspberry_pi_advertisement/data/data_pos"
    neg_dir = "/home/antonryoung02/raspberry_pi_advertisement/data/data_neg"

    image = preprocess_image(image_path, (320, 320))
    model.eval()
    with torch.no_grad():
        model_output = model(image)
        probability = torch.sigmoid(model_output).item()
        unique_filename = f"{time.strftime('%Y%m%d_%H%M%S')}_image.png"
        new_dir = pos_dir if probability > 0.5 else neg_dir
        new_path = os.path.join(new_dir, unique_filename)
        shutil.move(image_path, new_path)

    return probability
        
def main():
    print("main function called!")    
    model = SimpleCNN().load_model_checkpoint('/home/antonryoung02/raspberry_pi_advertisement/simple_cnn_checkpoint.pth')
    print("model checkpoint loaded")
    image_path = "/home/antonryoung02/raspberry_pi_advertisement/image.png"

    while os.path.exists("/tmp/classify_script_running"):

        if os.path.exists(image_path):
            time.sleep(1)
            output = run_inference(model, image_path)
            print(output)



if __name__ == "__main__":
    main()
