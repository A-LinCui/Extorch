from torchvision import datasets, transforms

from extorch.vision.dataset import CVClassificationDataset


MNIST_TRAIN_TRANSFORM = transforms.Compose([transforms.ToTensor()])
MNIST_TEST_TRANSFORM = transforms.Compose([transforms.ToTensor()])


class MNIST(CVClassificationDataset):
    def __init__(self, data_dir: str, 
                 train_transform: transforms.Compose = MNIST_TRAIN_TRANSFORM,
                 test_transform: transforms.Compose = MNIST_TEST_TRANSFORM) -> None:
        super(MNIST, self).__init__(data_dir, train_transform, test_transform)

        self.datasets["train"] = datasets.MNIST(root = self.data_dir, train = True,
                download = True, transform = train_transform)
        self.datasets["test"] = datasets.MNIST(root = self.data_dir, train = False,
                download = True, transform = test_transform)

        self._num_classes = 10


FashionMNIST_TRAIN_TRANSFORM = transforms.Compose([transforms.ToTensor()])
FashionMNIST_TEST_TRANSFORM = transforms.Compose([transforms.ToTensor()])


class FashionMNIST(CVClassificationDataset):
    def __init__(self, data_dir: str, 
                 train_transform: transforms.Compose = FashionMNIST_TRAIN_TRANSFORM,
                 test_transform: transforms.Compose = FashionMNIST_TEST_TRANSFORM) -> None:
        super(FashionMNIST, self).__init__(data_dir, train_transform, test_transform)

        self.datasets["train"] = datasets.FashionMNIST(root = self.data_dir, train = True,
                download = True, transform = train_transform)
        self.datasets["test"] = datasets.FashionMNIST(root = self.data_dir, train = False,
                download = True, transform = test_transform)

        self._num_classes = 10
