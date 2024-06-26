from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import random
from abc import ABC, abstractmethod

class AugmentedImageFolder(ImageFolder):
    def __init__(self, root, transform, augmentation=None):
        super().__init__(root, transform=transform)
        self.augmentation = augmentation

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        if self.augmentation:
            image = self.augmentation.transform(image)

        return image, label


class AbstractTransformation(ABC):
    @abstractmethod
    def transform(self, image):
        pass

class DefaultsTransformation(AbstractTransformation):
    def transform(self, image):
        if random.random() > 0.75:
            return image
        transforms = v2.Compose([
            v2.ColorJitter(),
            v2.RandomGrayscale(),
            v2.RandomAdjustSharpness(sharpness_factor=(1.5 - random.random())),
            v2.RandomHorizontalFlip()
        ])
        return transforms(image)