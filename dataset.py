import os
import random
import numpy as np
import tifffile as tiff
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2


class InstanceSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.samples = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        self.samples.sort()
        self.transform = transform

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        image = Image.open(os.path.join(sample_dir, "image.tif")).convert("RGB")
        W, H = image.size

        masks = []
        labels = []

        for class_id in range(1, 5):
            mask_path = os.path.join(sample_dir, f"class{class_id}.tif")
            if os.path.exists(mask_path):
                mask_img = np.array(tiff.imread(mask_path))
                instances = np.unique(mask_img)
                instances = instances[instances != 0]
                for inst_id in instances:
                    masks.append((mask_img == inst_id).astype(bool))
                    labels.append(class_id)

        if masks:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            masks = torch.zeros((0, H, W), dtype=torch.uint8)
            labels = []

        boxes = []
        for m in masks:
            pos = m.nonzero()
            if pos.numel() == 0:
                boxes.append(torch.tensor([0, 0, 0, 0], dtype=torch.float32))
            else:
                ymin = pos[:, 0].min()
                ymax = pos[:, 0].max()
                xmin = pos[:, 1].min()
                xmax = pos[:, 1].max()
                boxes.append(
                    torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
                )

        if boxes:
            boxes = torch.stack(boxes)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": masks,
            "image_id": torch.tensor([idx]),
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.samples)


class SegmentationTransform:
    def __init__(self, train=True):
        self.train = train

    def __call__(self, image, target):
        if self.train:
            if random.random() > 0.5:
                image = F.hflip(image)
                target["masks"] = target["masks"].flip(-1)
                w = image.width
                target["boxes"][:, [0, 2]] = w - target["boxes"][:, [2, 0]]

            if random.random() > 0.5:
                image = F.vflip(image)
                target["masks"] = target["masks"].flip(-2)
                h = image.height
                target["boxes"][:, [1, 3]] = h - target["boxes"][:, [3, 1]]

            color_aug = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
            )
            image = color_aug(image)

        image = F.to_tensor(image)
        return image, target


class AlbumentationsTransformWrapper:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, image, target):
        image_np = np.array(image)

        masks = [mask.numpy() for mask in target["masks"]]
        labels = target["labels"]

        transformed = self.aug(image=image_np, masks=masks)

        image = transformed["image"]

        transformed_masks = torch.stack(
            [m.to(torch.uint8) for m in transformed["masks"]]
        )

        boxes = []
        for mask in transformed_masks:
            pos = mask.nonzero()
            if pos.numel() == 0:
                boxes.append(torch.tensor([0, 0, 0, 0], dtype=torch.float32))
            else:
                ymin = pos[:, 0].min()
                ymax = pos[:, 0].max()
                xmin = pos[:, 1].min()
                xmax = pos[:, 1].max()
                boxes.append(
                    torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
                )

        target["masks"] = transformed_masks
        target["labels"] = labels
        target["boxes"] = torch.stack(boxes)

        return image, target


def get_transform(mode="train"):
    if mode == "train":
        aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                ToTensorV2(),
            ],
            additional_targets={"mask": "mask"},
        )
    elif mode == "val":
        aug = A.Compose(
            [A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)), ToTensorV2()],
            additional_targets={"mask": "mask"},
        )

    return AlbumentationsTransformWrapper(aug)


inference_aug = A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)), ToTensorV2()])
