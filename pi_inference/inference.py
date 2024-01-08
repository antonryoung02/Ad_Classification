import os
import time
import shutil
import torch
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

def run_inference(model, record_data, image_path):
    """>0.5 predicts advertisement, <0.5 predicts hockey

    param model: torch.nn.Module pytorch model
    param image_path: Path to image taken in classify_script.sh
    param record: Toggles data collection. True saves image to data directory
    """
    image = preprocess_image(image_path, (320, 320))
    model.eval()
    with torch.no_grad():
        model_output = model(image)
        probability = torch.sigmoid(model_output).item()

        if record_data:
            pos_dir = os.environ["POS_DIR"]
            neg_dir = os.environ["NEG_DIR"]
            unique_filename = f"{time.strftime('%Y%m%d_%H%M%S')}_image.png"
            new_dir = pos_dir if probability > 0.5 else neg_dir
            new_path = os.path.join(new_dir, unique_filename)
            shutil.move(image_path, new_path)
        else:
            os.remove(image_path)

    return probability
        
def main(): 
    model = SimpleCNN().load_model_checkpoint(os.environ["MODEL_CHECKPOINT_PATH"])
    image_path = os.environ["IMAGE_PATH"]
    record_data = False # Set to True for data collection

    while os.path.exists("/tmp/classify_script_running"):
        if os.path.exists(image_path):
            time.sleep(1)
            output = run_inference(model, record_data, image_path)
            print(output)

if __name__ == "__main__":
    main()
