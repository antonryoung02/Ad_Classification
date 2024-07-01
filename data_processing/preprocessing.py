import os
import PIL
from PIL import Image
from typing import Tuple

def preprocess_data(
    input_dir: str,
    output_dir: str,
    dimensions: Tuple[int, int] = (224, 224),
) -> None:
    """Applies preprocess_image on all images in input_dir and saves them to output_dir

    Args:
        input_dir (str): Path to unprocessed images
        output_dir (str): Path to save images
        dimensions (Tuple[int, int], optional): _description_. Defaults to (224, 224).

    Raises:
        FileNotFoundError: If input_dir or output_dir directories do not exist
    """
    if not os.path.exists(input_dir) or not os.path.exists(output_dir):
        raise FileNotFoundError("Input or output directories do not exist!")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                    processed_image = preprocess_image(image, dimensions)
                    original_output_path = os.path.join(output_dir, filename)
                    processed_image.save(original_output_path)
                except (PIL.UnidentifiedImageError, OSError) as e:
                    print(f"Error processing image {image_path}: {e}")
            else:
                print(f"File not found: {image_path}")


def preprocess_image(image: Image, dimensions: Tuple[int, int]) -> Image:
    """Applies resizing to an image

    Args:
        image (Image): The unprocessed PIL.Image object
        dimensions (Tuple[int, int]): The width, height dimensions to resize to

    Returns:
        Image: The processed image
    """
    width, height = dimensions
    resized_image = image.resize((width, height))
    return resized_image.convert("RGB")


def clear_directory(directory_path:str):
    """Clears a directory

    Args:
        directory_path (str):
    """
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error: {e}")
