from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import random
from abc import ABC, abstractmethod
import torch
from typing import Tuple

class AbstractTransformation(ABC):
    """Defines an interface for Transformation classes used in the AugmentedImageFolder"""
    @abstractmethod
    def transform(self, image:torch.Tensor) -> None:
        pass

class TrainTransformation(AbstractTransformation):
    """An instance of AbstractTransformation.

        Applies v2.ColorJitter, v2.RandomGrayscale, v2.RandomAdjustSharpness, v2.RandomHorizontalFlip,
        v2.RandomResizedCrop, and v2.Normalize transforms
    """
    def transform(self, image:torch.Tensor) -> torch.Tensor:
        """Method that defines the augmentation steps

        Args:
            image (torch.Tensor): Tensor before augmentation

        Returns:
            torch.Tensor: Tensor after augmentation
        """
        transforms = v2.Compose([
            v2.ColorJitter(brightness=(0.7,1), hue=.1, contrast=.1),
            v2.RandomGrayscale(),
            v2.RandomAdjustSharpness(sharpness_factor=2*random.random()),
            v2.RandomHorizontalFlip(),
            v2.RandomResizedCrop((224,224), scale=(0.8, 1), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Imagenet values
        ])
        return transforms(image)
    
class AugmentedImageFolder(ImageFolder):
    """ Dataset that performs image augmentation when data is accessed. Inherits from torchvision.datasets.ImageFolder"""
    def __init__(self, root:str, transform:v2.Compose, augmentation:AbstractTransformation=None):
        """
        Args:
            root (str): Path to data folders
            transform (torchvision.transforms.Compose): Initial transformations on load. Include a v2.ToTensor()
            augmentation (AbstractTransformation, optional): A derived class with a .transorm method for image augmentations. Defaults to None.
        """
        super().__init__(root, transform=transform)
        self.augmentation = augmentation

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        """Overridden indexing function from torchvision.datasets object. Applies augmentation if defined in self.init()
        Args:
            index (int): Index of image

        Returns:
            tuple: A tuple containing:
                - image (torch.tensor): The image from the dataset, transformed if augmentation is enabled.
                - label (int): The label corresponding to the image.
        """
        image, label = super().__getitem__(index)
        if self.augmentation:
            image = self.augmentation.transform(image)

        return image, label
