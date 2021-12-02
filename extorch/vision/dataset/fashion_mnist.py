from typing import Optional, List, Dict

from torchvision import datasets, transforms

from extorch.vision.dataset import CVClassificationDataset


class FashionMNIST(CVClassificationDataset):
    def __init__(self, data_dir: str, 
                 train_transform: Optional[transforms.Compose] = None, 
                 test_transform: Optional[transforms.Compose] = None,
                 cutout_length: Optional[int] = None,
                 cutout_n_holes: Optional[int] = 1) -> None:
        super(FashionMNIST, self).__init__(
            data_dir, train_transform, test_transform, cutout_length, cutout_n_holes)
        
        self.datasets["train"] = datasets.FashionMNIST(root = self.data_dir, train = True, 
                download = True, transform = self.transforms["train"])
        self.datasets["test"] = datasets.FashionMNIST(root = self.data_dir, train = False, 
                download = True, transform = self.transforms["test"])

    @classmethod
    def num_classes(cls) -> int:
        return 10

    @classmethod
    def mean(cls) -> List[float]:
        raise NotImplementedError

    @classmethod
    def std(cls) -> List[float]:
        raise NotImplementedError

    @property
    def default_transform(self) -> Dict[str, transforms.Compose]:
        default_transforms = {
            "train": transforms.Compose([transforms.ToTensor()]),
            "test": transforms.Compose([transforms.ToTensor()])
        }
        return default_transforms
