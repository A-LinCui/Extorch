from typing import Optional, List, Dict

import torch
from torchvision.datasets import VOCSegmentation
from torchvision import transforms

from extorch.vision.dataset import SegmentationDataset
from extorch.vision.transforms import segmentation as segT


class VOCSegmentationDataset(SegmentationDataset):
    def __init__(self, data_dir: str, year: str = "2012",
                 train_transform: Optional[segT.SegCompose] = None,
                 test_transform: Optional[segT.SegCompose] = None) -> None:
        super(VOCSegmentationDataset, self).__init__(data_dir, train_transform, test_transform)
        
        self.datasets["train"] = VOCSegmentation(
                root = self.data_dir, year = year, image_set = "train", download = True,
                transforms = self.transforms["train"])
        self.datasets["test"] = VOCSegmentation(
                root = self.data_dir, year = year, image_set = "val", download = True,
                transforms = self.transforms["test"])

    @classmethod
    def num_classes(cls) -> int:
        r"""
        Number of classes excluding the background.
        """
        return 20

    @classmethod
    def mean(cls) -> List[float]:
        return [0.485, 0.456, 0.406]

    @classmethod
    def std(cls) -> List[float]:
        return [0.229, 0.224, 0.225]

    @property
    def default_transform(self) -> Dict[str, segT.SegCompose]:
        default_transforms = {
            "train": segT.SegCompose([segT.SegRandomResize(min_size = 260, max_size = 1040),
                                      segT.SegRandomHorizontalFlip(p = 0.5),
                                      segT.SegRandomCrop(size = 480),
                                      segT.SegPILToTensor(),
                                      segT.SegConvertImageDtype(torch.float),
                                      segT.SegNormalize(self.mean(), self.std())]),
            
            "test": segT.SegCompose([segT.SegResize(size = 520),
                                     segT.SegPILToTensor(),
                                     segT.SegConvertImageDtype(torch.float),
                                     segT.SegNormalize(self.mean(), self.std())])
        }
        return default_transforms
