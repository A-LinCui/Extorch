import abc
from typing import Dict, Optional, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from extorch.utils.common import abstractclassmethod
from extorch.vision.transforms import Cutout


class BaseDataset(object):
    r"""
    Base dataset.

    Args:
        data_dir (str): Base path of the data.
    """
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

    @abc.abstractproperty
    def splits(self) -> Dict[str, Dataset]:
        r"""
        Dataset of different splits.

        Returns:
            Dict(str: torch.utils.data.Dataset): A dict from split name to dataset.
        """

    @abstractclassmethod
    def data_type(cls) -> str:
        r"""
        Type of the dataset.

        Returns:
            The data type of this dataset.
        """


class CVDataset(BaseDataset):
    r"""
    Base dataset for computer vision tasks.
    
    Args:
        data_dir (str): Base path of the data.
        train_transform (transforms.Compose): Data transform of the training split. 
        test_transform (transforms.Compose): Data transform of the test split. 
    """
        
    def __init__(self, data_dir: str, 
                 train_transform: transforms.Compose, 
                 test_transform: transforms.Compose) -> None:

        super(CVDataset, self).__init__(data_dir)
        self.datasets = {}
        self.transforms = {
            "train": train_transform or self.default_transform["train"],
            "test": test_transform or self.default_transform["test"]
        }

    @property
    def data_transforms(self) -> Dict[str, transforms.Compose]:
        r"""
        Returns:
            Dict(str: transforms.Compose): A dict from split name to data transformation.
        """
        return self.transforms

    @classmethod
    def data_type(cls) -> str:
        return "image"

    @property
    def splits(self) -> Dict[str, Dataset]:
        return self.datasets

    @abstractclassmethod
    def num_classes(cls) -> int:
        r"""
        Number of classes.

        Returns:
            int: The number of classes.
        """
    
    @abstractclassmethod
    def mean(cls) -> List[float]:
        r"""
        Returns:
            List[float]: Means for each channel.
        """

    @abstractclassmethod
    def std(cls) -> List[float]:
        r"""
        Returns:
            List[float]: Standard deviations for each channel.
        """

    @abc.abstractproperty
    def default_transform(self) -> Dict[str, transforms.Compose]:
        r"""
        Default transforms of different splits.

        Returns:
            Dict(str: transforms.Compose): The default transforms.
        """


class CVClassificationDataset(CVDataset):
    r"""
    Base dataset for computer vision classification tasks.

    Args:
        data_dir (str): Base path of the data.
        train_transform (Optional[transforms.Compose]): Data transform of the training split. Default: ``None``.
        test_transform (Optional[transforms.Compose]): Data transform of the test split. Default: ``None``.
        cutout_length (Optional[int]): The length (in pixels) of each square patch in Cutout.
                                       If train transform is not specified and cutout_length is not None, 
                                       we will add Cutout transform at the end. Default: ``None``.
        cutout_n_holes (Optional[int]): Number of patches to cut out of each image. Default: 1.
    """

    def __init__(self, data_dir: str,
                 train_transform: Optional[transforms.Compose] = None, 
                 test_transform: Optional[transforms.Compose] = None,
                 cutout_length: Optional[int] = None,
                 cutout_n_holes: Optional[int] = 1) -> None:
        
        # If transform is not specified, use the default transform.
        if train_transform is None:
            train_transform = self.default_transform["train"]
            if cutout_length:
                train_transform.transforms.append(Cutout(cutout_length, cutout_n_holes))

        super(CVClassificationDataset, self).__init__(data_dir, train_transform, test_transform)


class SegmentationDataset(CVDataset):
    r"""
    Base dataset for computer vision segmentation tasks.

    Args:
        data_dir (str): Base path of the data.
        train_transform (Optional[transforms.Compose]): Data transform of the training split. Default: ``None``.
        test_transform (Optional[transforms.Compose]): Data transform of the test split. Default: ``None``.
    """

    def __init__(self, data_dir: str, 
                 train_transform: transforms.Compose,
                 test_transform: transforms.Compose) -> None:
        super(SegmentationDataset, self).__init__(data_dir, train_transform, test_transform)
