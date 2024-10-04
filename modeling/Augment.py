from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import random
from abc import ABC, abstractmethod
import torch
from typing import Tuple
from itertools import accumulate

class AbstractTransformation(ABC):
    """Defines an interface for Transformation classes used in the AugmentedImageFolder"""
    @abstractmethod
    def transform(self, image:torch.Tensor, label:int) -> torch.Tensor:
        pass

class TransformationSampler(AbstractTransformation):
    """Class for applying a random transformation with label-specific probability."""
    def __init__(self, transformations:list[AbstractTransformation], pos_probs:list[float], neg_probs:list[float] | None=None):
        """
        Args:
            transformations (list[AbstractTransformation]): List of transformations to sample from
            pos_probs (list[float)): List of transformation sampling probabilities for the positive class
            neg_probs (list[float] | None): List of transformation sampling probabilities for the negative class. Defaults to neg_prob=pos_prob
        """
        self._ensure_valid_init(transformations, pos_probs, neg_probs)
        self.transformations = transformations
        self.pos_probabilities = pos_probs
        self.neg_probabilities = neg_probs
        
    
    def transform(self, image:torch.Tensor, label:int) -> torch.Tensor:
        """Method that performs the random augmentation

        Args:
            image (torch.Tensor): Tensor before augmentation

        Returns:
            torch.Tensor: Tensor after augmentation
        """
        prob = random.random()
        transformation = self._choose_transformation(prob, label)
        if transformation:
            return transformation.transform(image, label)
        return image
        
    def _choose_transformation(self, prob:float, label:int) -> AbstractTransformation | None:
        """Helper method that finds the sampled transformation"""
        if label == 0 and self.neg_probabilities:
            probabilities = self.neg_probabilities
        else:
            probabilities = self.pos_probabilities

        cumulative_probs = list(accumulate(probabilities))
        for i, cumulative_prob in enumerate(cumulative_probs):
            if prob < cumulative_prob:
                return self.transformations[i]
        return None
        
    def _ensure_valid_init(self, transformations:list[AbstractTransformation], pos_probs:list[float], neg_probs:list[float]|None=None):
        """Helper method that ensures the class was initialized correctly"""
        if len(transformations) != len(pos_probs):
            raise ValueError(f"Expected len(transformations) {len(transformations)} to equal len(pos_probs) {len(pos_probs)}")
        if neg_probs and len(transformations) != len(neg_probs): 
            raise ValueError(f"Expected len(transformations) {len(transformations)} to equal len(neg_probs) {len(neg_probs)}")
        
        if sum(pos_probs) > 1.0 or sum(pos_probs) < 0.0:
            raise ValueError("Expected sum of pos_probs to be 0 <= sum <= 1.0")
        if neg_probs and (sum(neg_probs) > 1.0 or sum(neg_probs) < 0.0):
            raise ValueError("Expected sum of neg_probs to be 0 <= sum <= 1.0")
        
        if not all(isinstance(t, AbstractTransformation) for t in transformations):
            raise ValueError("All elements in 'transformations' must be instances of AbstractTransformation.")
    
class GeneralImageAugmentations(AbstractTransformation):
    """An instance of AbstractTransformation.

        Applies v2.ColorJitter, v2.RandomGrayscale, v2.RandomAdjustSharpness, v2.RandomHorizontalFlip,
        v2.RandomResizedCrop, and v2.Normalize transforms
    """
    def __init__(self, config):
        self.config = config
        
    def transform(self, image:torch.Tensor, label:int) -> torch.Tensor:
        """Method that defines the augmentation steps

        Args:
            image (torch.Tensor): Tensor before augmentation

        Returns:
            torch.Tensor: Tensor after augmentation
        """
        hue = self.config.pop('hue')
        contrast = self.config.pop('contrast')
        
        transforms = v2.Compose([
            v2.ColorJitter(brightness=(0.7,1), hue=hue, contrast=contrast), #hockey .1,.1, football .4,.2
            v2.RandomGrayscale(),
            v2.RandomAdjustSharpness(sharpness_factor=2*random.random()),
            v2.RandomHorizontalFlip(),
            v2.RandomResizedCrop((224,224), scale=(0.8, 1), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Imagenet values
        ])
        return transforms(image)
    
class AugmentedImageFolder(ImageFolder):
    """ Dataset that performs image augmentation when data is accessed. Inherits from torchvision.datasets.ImageFolder"""
    def __init__(self, root:str, transform:v2.Compose, augmentation:AbstractTransformation|None=None):
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
            image = self.augmentation.transform(image, label)

        return image, label

class AugmentationFactory:
    def __call__(self, config):
        match config['type']:
            case 'general':
                return GeneralImageAugmentations(config)
            case _:
                return None
        
        