import abc
from typing import Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from extorch.utils.common import abstractclassmethod


class BaseDataset(object):
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

    @abc.abstractmethod
    def splits(self) -> Dict[str, Dataset]:
        """
        Returns:
            Dict(str: torch.utils.data.Dataset): A dict from split name to dataset.
        """

    @abstractclassmethod
    def data_type(cls) -> str:
        """
        Returns:
            The data type of this dataset.
        """


class CVDataset(BaseDataset):
    def __init__(self, data_dir: str, 
                 train_transform: transforms.Compose, 
                 test_transform: transforms.Compose) -> None:
        super(CVDataset, self).__init__(data_dir)
        self.datasets = {}
        self.transforms = {
            "train": train_transform,
            "test": test_transform
        }

    def data_transforms(self) -> Dict[str, transforms.Compose]:
        """
        Returns:
            Dict(transforms.Compose): A dict from split name to data transformation.
        """
        return self.transforms

    @classmethod
    def data_type(cls) -> str:
        return "image"
    
    def splits(self) -> Dict[str, Dataset]:
        return self.datasets


class CVClassificationDataset(CVDataset):
    def __init__(self, data_dir: str, 
                 train_transform: transforms.Compose, 
                 test_transform: transforms.Compose) -> None:
        super(CVClassificationDataset, self).__init__(data_dir, train_transform, test_transform)
        self._num_classes = None

    @property
    def num_classes(self) -> int:
        """
        Returns:
            int: The number of classes.
        """
        return self._num_classes
