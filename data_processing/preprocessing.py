import os
from PIL import Image
from data_augmentation import apply_augmentations

def preprocess_data(input_dir:str, output_dir:str, augment_input:bool=True, dimensions:tuple=(320,320)) -> None:
    """
    Transforms all .png files to desired specifications using preprocess_image
    and apply_augmentations

    param input_dir: the directory of untransformed images
    param output_dir: where transformed images are stored
    param augment_input: Whether or not to augment the data in input_dir
    param dimensions: (width, height) in pixels of transformed image size
    """
    if not os.path.exists(input_dir) or not os.path.exists(output_dir):
        return

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path)
            processed_image = preprocess_image(image, dimensions)
            if augment_input:
                apply_augmentations(processed_image, output_dir, filename)
            else:
                original_output_path = os.path.join(output_dir, 'original_' + filename)
                processed_image.save(original_output_path)

def preprocess_image(image:Image, dimensions:tuple) -> Image:
    """
    Converts an Image object to size / #channels compatible with the CNN
    """
    width, height = dimensions
    resized_image = image.resize((width, height))
    return resized_image.convert("RGB")

def clear_directory(directory):
    """Clears output directory for test reusability"""

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Error: {e}')
