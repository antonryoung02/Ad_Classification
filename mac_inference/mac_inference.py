import os
import time
import shutil
import torch
from torchvision import transforms
from PIL import Image
from models import SimpleCNN

from torch import nn
def preprocess_image(image_path, dimensions):
    """Transforms to fit model input expectations"""
    width, height = dimensions
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).resize((width, height)).convert("RGB")
    transformed_image = transform(image).unsqueeze(0)
    return transformed_image


def run_inference(model, device, record_data, image_path):
    """>0.5 predicts advertisement, <0.5 predicts hockey

    param model: torch.nn.Module pytorch model
    param image_path: Path to image taken in classify_script.sh
    param record: Toggles data collection. True saves image to data directory
    """
    image = preprocess_image(image_path, (224, 224))
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
    print(probability)
    return probability


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(os.environ["MODEL_CHECKPOINT_PATH"])
    print(model)
    image_path = os.environ["IMAGE_PATH"]
    record_data = False  # Set to True for data collection

    while os.path.exists("./classify_script_running"):
        if os.path.exists(image_path):
            time.sleep(1)
            output = run_inference(model, device, record_data, image_path)
            if output > 0.4:
                prediction = "True"
            else:
                prediction = "False"

            with open("./ad_signal.txt", "w") as file:
                file.write(prediction)

if __name__ == "__main__":
    main()
