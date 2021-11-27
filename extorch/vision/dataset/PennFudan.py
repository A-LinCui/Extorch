import os
from PIL import Image
from typing import Tuple, Optional, Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

from extorch.vision.dataset import CVDataset
from extorch.vision.transforms import *


PennFudan_TRAIN_TRANSFORM = DetectionCompose([
    DetectionToTensor(),
    DetectionRandomHorizontalFlip(p = 0.5)
])

PennFudan_TEST_TRANSFORM = DetectionCompose([
    DetectionToTensor()
])


class PennFudan(CVDataset):
    r"""
    PennFudan dataset.
    Download the dataset from ``https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip``.

    Args:
        data_dir (str): Path of the dataset.
        train_ratio (float): Ratio of the 170 instances as the training split.
        random_split (bool): Whether randomly shuffle the dataset before split.
        train_transform (transforms.Compose): Train transforms.
        test_transform (transforms.Compose): Test transforms.
    """
    def __init__(self, data_dir: str, train_ratio: float, random_split: bool,
            train_transform: transforms.Compose = PennFudan_TRAIN_TRANSFORM, 
            test_transform: transforms.Compose = PennFudan_TEST_TRANSFORM) -> None:
        super(PennFudan, self).__init__(data_dir, train_transform, test_transform)
        dataset = PennFudanDataset(data_dir, None)
        total_num = len(dataset)
        indices = (torch.randperm if random_split else np.arange)(total_num).tolist()
        train_dataset = Subset(dataset, indices[:int(train_ratio * total_num)])
        train_dataset.transforms = self.transforms["train"]
        test_dataset = Subset(dataset, indices[int(train_ratio * total_num):])
        test_dataset.transforms = self.transforms["test"]
        self.datasets["train"] = train_dataset
        self.datasets["test"] = test_dataset


class PennFudanDataset(object):
    def __init__(self, root: str, transforms: transforms.Compose) -> None:
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        # there is only one class
        labels = torch.ones((num_objs, ), dtype = torch.int64)
        masks = torch.as_tensor(masks, dtype = torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype = torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.imgs)


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class DetectionRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
            ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = F.get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target
