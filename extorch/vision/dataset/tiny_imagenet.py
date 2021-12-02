from typing import Optional, List, Dict
import os

from torchvision import datasets, transforms

from extorch.vision.dataset import CVClassificationDataset


class TinyImageNet(CVClassificationDataset):
    r"""
    Tiny ImageNet dataset.

    Args:
        color_jitter (bool): Whether add color_jitter into default train transform. Default: ``False``.
        train_crop_size (int): Default: 64.
        test_crop_size (int): Default: 64.
    """
    def __init__(self, data_dir: str, color_jitter: bool = False, 
                 train_crop_size: int = 64, test_crop_size: int = 64,
                 train_transform: Optional[transforms.Compose] = None,
                 test_transform: Optional[transforms.Compose] = None,
                 cutout_length: Optional[int] = None,
                 cutout_n_holes: Optional[int] = 1) -> None:

        # setting for default transform 
        self.color_jitter = color_jitter
        self.train_crop_size = train_crop_size
        self.test_crop_size = test_crop_size

        super(TinyImageNet, self).__init__(
                data_dir, train_transform, test_transform, cutout_length, cutout_n_holes)

        self.train_data_dir = os.path.join(self.data_dir, "train")
        self.test_data_dir = os.path.join(self.data_dir, "val")
        self.datasets["train"] = datasets.ImageFolder(root = self.train_data_dir,
                transform = self.transforms["train"])
        self.datasets["test"] = datasets.ImageFolder(root = self.test_data_dir,
                transform = self.transforms["test"])

    @classmethod
    def num_classes(cls) -> int:
        return 200

    @classmethod
    def mean(cls) -> List[float]:
        return [0.485, 0.456, 0.406]

    @classmethod
    def std(cls) -> List[float]:
        return [0.229, 0.224, 0.225]

    @property
    def default_transform(self) -> Dict[str, transforms.Compose]:
        train_transform = [
                transforms.RandomResizedCrop(self.train_crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean(), self.std())]

        if self.color_jitter:
            color_jitter_transform = transforms.ColorJitter(
                    brightness = 0.4, contrast = 0.4, saturation = 0.4, hue = 0.2)
            train_transform.insert(2, color_jitter_transform)
            
        train_transform = transforms.Compose(train_transform)

        test_transform = transforms.Compose([
                transforms.Resize(self.test_crop_size + 8),
                transforms.CenterCrop(self.test_crop_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean(), self.std())])

        default_transforms = {
            "train": train_transform,
            "test": test_transform
        }
        return default_transforms
