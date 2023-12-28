import os
import random
from PIL import Image
from data_augmentation import add_border, crop_image, add_random_filter, darken_image

def preprocess_image(image: Image, dimensions: tuple=(320,320)) -> Image:
    """
    Converts an Image object to size / #channels compatible with the CNN
    """
    width, height = dimensions
    resized_image = image.resize((width, height))
    return resized_image.convert("RGB")

def preprocess_data(input_dir:str, output_dir:str, dimensions:tuple=(320,320)) -> None:
    """
    Transforms all .png files to desired specifications using preprocess_image
    and functions in data_augmentation

    param input_dir: the directory of untransformed images
    param output_dir: where transformed images are stored
    param dimensions: (width, height) in pixels of transformed image size
    """
    if not os.path.exists(input_dir) or not os.path.exists(output_dir):
        return

    augmentation_functions = [crop_image, add_random_filter, darken_image]
    #augmentation_functions = [add_border, add_random_filter]

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path)

            # Apply initial preprocessing
            processed_image = preprocess_image(image, dimensions)

            # Randomly decide whether to add a border
            if random.random() > 0.5:
                processed_image = add_border(processed_image)
                # Reapply preprocessing if necessary after adding the border
                processed_image = preprocess_image(processed_image, dimensions)

            original_output_path = os.path.join(output_dir, 'original_' + filename)
            processed_image.save(original_output_path)

            for augmentation in augmentation_functions:
                augmented_image = augmentation(processed_image.copy())
                processed_augmented_image = preprocess_image(augmented_image, dimensions)
                augmented_output_path = os.path.join(output_dir, augmentation.__name__ + '_' + filename)
                processed_augmented_image.save(augmented_output_path)


def clear_directory(directory):
    """Clears output directory for test reusability"""

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Error: {e}')
