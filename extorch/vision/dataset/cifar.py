from torchvision import datasets, transforms

from extorch.vision.dataset import CVClassificationDataset


# Standard transformation for CIFAR10 datasets
CIFAR10_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR10_STD = [0.24703233, 0.24348505, 0.26158768]

CIFAR10_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

CIFAR10_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])


class CIFAR10(CVClassificationDataset):
    def __init__(self, data_dir: str, 
                 train_transform: transforms.Compose = CIFAR10_TRAIN_TRANSFORM, 
                 test_transform: transforms.Compose = CIFAR10_TEST_TRANSFORM) -> None:
        super(CIFAR10, self).__init__(data_dir, train_transform, test_transform)

        self.datasets["train"] = datasets.CIFAR10(root = self.data_dir, train = True, 
                download = True, transform = self.transforms["train"])
        self.datasets["test"] = datasets.CIFAR10(root = self.data_dir, train = False, 
                download = True, transform = self.transforms["test"])

        self._num_classes = 10


# Standard transformation for CIFAR100 datasets
CIFAR100_MEAN = [0.5070751592371322, 0.4865488733149497, 0.44091784336703466]
CIFAR100_STD = [0.26733428587924063, 0.25643846291708833, 0.27615047132568393]

CIFAR100_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
])

CIFAR100_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
])


class CIFAR100(CVClassificationDataset):
    def __init__(self, data_dir: str, 
                 train_transform: transforms.Compose = CIFAR100_TRAIN_TRANSFORM, 
                 test_transform: transforms.Compose = CIFAR100_TEST_TRANSFORM) -> None:
        super(CIFAR100, self).__init__(data_dir, train_transform, test_transform)

        self.datasets["train"] = datasets.CIFAR100(root = self.data_dir, train = True, 
                download = True, transform = self.transforms["train"])
        self.datasets["test"] = datasets.CIFAR100(root = self.data_dir, train = False, 
                download = True, transform = self.transforms["test"])

        self._num_classes = 100
