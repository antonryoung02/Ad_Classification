import random
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageEnhance
import numpy as np
import os 

def apply_augmentations(image, output_dir, filename) -> None:
    """Adds variations to the data by calling a set of random image-augmenting functions."""
    if random.random() > 0.5: # Add noise borders to half of the images
        image = add_border(image)

    original_output_path = os.path.join(output_dir, 'original_' + filename)
    image.save(original_output_path)

    augmentation_functions = [crop_image, add_random_filter, darken_image]
    for augmentation in augmentation_functions:
        augmented_image = augmentation(image.copy())
        augmented_output_path = os.path.join(output_dir, augmentation.__name__ + '_' + filename)
        augmented_image.save(augmented_output_path)

def add_border(image: Image, intensity=0.3) -> Image:
    """Adds a border with noise to the image."""
    width, height = image.size
    left_border = random.randint(30, 70)
    top_border = random.randint(30, 70)
    right_border = random.randint(30, 70)
    bottom_border = random.randint(30, 70)

    border_color = 'black'
    bordered_image = ImageOps.expand(image, border=(left_border, top_border, right_border, bottom_border), fill=border_color)
    noise = np.random.normal(0, 255 * intensity, bordered_image.size + (3,)).astype(np.uint8)
    noise_image = Image.fromarray(noise)

    mask = Image.new("L", bordered_image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((left_border, top_border, bordered_image.width - right_border, bordered_image.height - bottom_border), fill=255)

    noisy_bordered_image = Image.composite(bordered_image, noise_image, mask)
    return noisy_bordered_image.resize((width, height))

def crop_image(image: Image) -> Image:
    """Crops the image by a random amount"""
    width, height = image.size

    left = random.randint(0,width // 8)
    right = width - random.randint(0, width // 8)
    top = random.randint(0,height // 8)
    bottom = height - random.randint(0, height // 8)
    crop_box = (left, top, right, bottom)
    cropped_image = image.crop(crop_box)
    return cropped_image.resize((width, height))

def darken_image(image: Image) -> Image:
    """Darkens image pixels by a random amount"""
    factor = random.uniform(0.4, 0.8)
    enhancer = ImageEnhance.Brightness(image)
    darkened_image = enhancer.enhance(factor)
    return darkened_image

def add_random_filter(image:Image) -> Image:
    """Applies a random filter to the image."""
    def add_gaussian_blur(image:Image) -> Image:
        return image.filter(ImageFilter.GaussianBlur(1))

    def add_unsharp_mask(image:Image) -> Image:
        return image.filter(ImageFilter.UnsharpMask())

    def add_edge_enhance(image:Image) -> Image:
        return image.filter(ImageFilter.EDGE_ENHANCE)
    
    filters = [add_gaussian_blur, add_unsharp_mask, add_edge_enhance]
    chosen_filter = random.choice(filters)
    return chosen_filter(image)




