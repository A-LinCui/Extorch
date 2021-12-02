from typing import Optional, List, Dict

from torchvision import datasets, transforms

from extorch.vision.dataset import CVClassificationDataset


class CIFAR10(CVClassificationDataset):
    def __init__(self, data_dir: str, 
                 train_transform: Optional[transforms.Compose] = None, 
                 test_transform: Optional[transforms.Compose] = None,
                 cutout_length: Optional[int] = None,
                 cutout_n_holes: Optional[int] = 1) -> None:
        super(CIFAR10, self).__init__(data_dir, train_transform, test_transform, cutout_length, cutout_n_holes)

        self.datasets["train"] = datasets.CIFAR10(root = self.data_dir, train = True, 
                download = True, transform = self.transforms["train"])
        self.datasets["test"] = datasets.CIFAR10(root = self.data_dir, train = False, 
                download = True, transform = self.transforms["test"])

    @classmethod
    def num_classes(cls) -> int:
        return 10

    @classmethod
    def mean(cls) -> List[float]:
        return [0.49139968, 0.48215827, 0.44653124]

    @classmethod
    def std(cls) -> List[float]:
        return [0.24703233, 0.24348505, 0.26158768]

    @property
    def default_transform(self) -> Dict[str, transforms.Compose]:
        default_transforms = {
            "train": transforms.Compose([transforms.RandomCrop(32, padding = 4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(self.mean(), self.std())]),
            "test": transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(self.mean(), self.std())])
        }
        return default_transforms
