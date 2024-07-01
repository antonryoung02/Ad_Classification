import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from data_processing.preprocessing import clear_directory
from modeling.Augment import TrainTransformation, AbstractTransformation, AugmentedImageFolder
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image

def simulate_data_augmentation_method(augmentation:AbstractTransformation, invert_normalization:bool=False):
    """Helps test quality of the transformations in test_output directory"""
    input_dir = "./tests/test_input"
    output_dir = "./tests/test_output"
    clear_directory(output_dir)

    train_transform  = v2.ToTensor()
    invert_train_transform = v2.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], #Inverts imagenet normalization constants
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    train_data_folder = AugmentedImageFolder(root=input_dir, transform=train_transform, augmentation=augmentation)

    for i, augmented_tensor in enumerate(train_data_folder):
        if invert_normalization:
            augmented_tensor = invert_train_transform(augmented_tensor)
        augmented_image = to_pil_image(augmented_tensor[0])
        augmented_image.save(f"{output_dir}/image{i}.png")

def main():
    dt = TrainTransformation()
    simulate_data_augmentation_method(dt, invert_normalization=False)

if __name__ == "__main__":
    main()




    



