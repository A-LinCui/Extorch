from torchvision import datasets, transforms

from extorch.vision.dataset import CVClassificationDataset


# Standard transformation for SVHN datasets
SVHN_MEAN = [0.4377, 0.4438, 0.4728]
SVHN_STD = [0.1980, 0.2010, 0.1970]

SVHN_TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
])

SVHN_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
])


class SVHN(CVClassificationDataset):
    def __init__(self, data_dir: str, 
                 train_transform: transforms.Compose = SVHN_TRAIN_TRANSFORM,
                 test_transform: transforms.Compose = SVHN_TEST_TRANSFORM,
                 extra_transform: transforms.Compose = SVHN_TEST_TRANSFORM) -> None:
        super(SVHN, self).__init__(data_dir, train_transform, test_transform)
        
        self.transforms["extra"] = extra_transform

        self.datasets["train"] = datasets.SVHN(root = self.data_dir, split = "train",
                download = True, transform = train_transform)
        self.datasets["test"] = datasets.SVHN(root = self.data_dir, split = "test",
                download = True, transform = test_transform)
        self.datasets["extra"] = datasets.SVHN(root = self.data_dir, split = "extra",
                download = True, transform = train_transform)

        self._num_classes = 10
