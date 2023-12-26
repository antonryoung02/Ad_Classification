import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models import SimpleCNN

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
    image = preprocess_image(image_path, (320, 180))
    model.eval()
    with torch.no_grad():
        model_output = model(image)
        probability = torch.sigmoid(model_output).item()
    return probability

model = SimpleCNN().load_model_checkpoint('/home/antonryoung02/raspberry_pi_advertisement/simple_cnn_checkpoint.pth')

image_path = '/home/antonryoung02/raspberry_pi_advertisement/screenshot_1.png'
output = run_inference(model, image_path)
print(output)

image_path = '/home/antonryoung02/raspberry_pi_advertisement/screenshot_2.png'
output = run_inference(model, image_path)
print(output)
