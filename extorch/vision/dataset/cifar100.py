from typing import Optional, List, Dict

from torchvision import datasets, transforms

from extorch.vision.dataset import CVClassificationDataset


class CIFAR100(CVClassificationDataset):
    def __init__(self, data_dir: str, 
                 train_transform: Optional[transforms.Compose] = None, 
                 test_transform: Optional[transforms.Compose] = None,
                 cutout_length: Optional[int] = None,
                 cutout_n_holes: Optional[int] = 1) -> None:
        super(CIFAR100, self).__init__(data_dir, train_transform, test_transform, cutout_length, cutout_n_holes)

        self.datasets["train"] = datasets.CIFAR100(root = self.data_dir, train = True, 
                download = True, transform = self.transforms["train"])
        self.datasets["test"] = datasets.CIFAR100(root = self.data_dir, train = False, 
                download = True, transform = self.transforms["test"])

    @classmethod
    def num_classes(cls) -> int:
        return 100

    @classmethod
    def mean(cls) -> List[float]:
        return [0.5070751592371322, 0.4865488733149497, 0.44091784336703466]

    @classmethod
    def std(cls) -> List[float]:
        return [0.26733428587924063, 0.25643846291708833, 0.27615047132568393]

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
