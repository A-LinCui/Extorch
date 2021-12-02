from typing import Optional, List, Dict

from torchvision import datasets, transforms

from extorch.vision.dataset import CVClassificationDataset


class SVHN(CVClassificationDataset):
    def __init__(self, data_dir: str, 
                 train_transform: Optional[transforms.Compose] = None, 
                 test_transform: Optional[transforms.Compose] = None,
                 cutout_length: Optional[int] = None,
                 cutout_n_holes: Optional[int] = 1) -> None:
        super(SVHN, self).__init__(data_dir, train_transform, test_transform, cutout_length, cutout_n_holes)
        
        self.datasets["train"] = datasets.SVHN(root = self.data_dir, split = "train", 
                download = True, transform = self.transforms["train"])
        self.datasets["test"] = datasets.SVHN(root = self.data_dir, split = "test",
                download = True, transform = self.transforms["test"])
        self.datasets["extra"] = datasets.SVHN(root = self.data_dir, split = "extra",
                download = True, transform = self.transforms["test"])

    @classmethod
    def num_classes(cls) -> int:
        return 10

    @classmethod
    def mean(cls) -> List[float]:
        return [0.4377, 0.4438, 0.4728]

    @classmethod
    def std(cls) -> List[float]:
        return [0.1980, 0.2010, 0.1970]

    @property
    def default_transform(self) -> Dict[str, transforms.Compose]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean(), self.std())
        ])

        default_transforms = {
            "train": transform,
            "test": transform
        }
        return default_transforms
