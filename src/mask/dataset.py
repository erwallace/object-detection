from PIL import Image
import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2

from mask.utils import parse_xml


class MaskedFaceTestDataset(Dataset):
    def __init__(self, root, img_transforms=T.ToTensor(), target_transforms=None):
        super(MaskedFaceTestDataset, self).__init__()
        self.imgs = sorted(glob.glob(os.path.join(root, "*.png")))
        self.img_transforms = img_transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert("RGB")

        xml_path = img_path.replace(".png", ".xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
        filename, boxes, labels = parse_xml(xml_path)

        boxes = torch.tensor(boxes, dtype=torch.float32)

        class_map = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}
        labels = torch.tensor([class_map[label] for label in labels], dtype=torch.int64)

        # image_id = torch.tensor([int(filename[5:].split('.')[0])])
        # image_id = filename.split('.')[0]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            # "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.img_transforms is not None:
            img = self.img_transforms(img)

        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class AugmentedMaskDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        # Convert PIL image to numpy array if needed
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Extract boxes and labels for albumentations format
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()

        # Format conversion - assuming XYXY format
        transformed = self.transform(image=img, bboxes=boxes, category_ids=labels)

        # Update with transformed values
        img_transformed = transformed["image"]
        boxes_transformed = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels_transformed = torch.tensor(
            transformed["category_ids"], dtype=torch.int64
        )

        # Update target with new boxes and labels
        new_target = {
            "boxes": boxes_transformed,
            "labels": labels_transformed,
            "area": (boxes_transformed[:, 3] - boxes_transformed[:, 1])
            * (boxes_transformed[:, 2] - boxes_transformed[:, 0]),
            "iscrowd": target["iscrowd"],
        }

        return img_transformed, new_target


def create_augmented_dataset(dataset):
    # Define different augmentation pipelines

    # 1. Mild augmentations
    transform_mild = A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=0.1, p=0.5),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    )

    # 2. Color/lighting augmentations
    transform_color = A.Compose(
        [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1),
                    A.HueSaturationValue(p=1),
                    A.RGBShift(p=1),
                ],
                p=0.9,
            ),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    )

    # 3. Geometric augmentations
    transform_geometric = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5
            ),
            A.RandomCrop(height=480, width=480, p=0.3),
            A.Resize(height=512, width=512),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    )

    # Create augmented versions of the dataset
    augmented_mild = AugmentedMaskDataset(dataset, transform_mild)
    augmented_color = AugmentedMaskDataset(dataset, transform_color)
    augmented_geometric = AugmentedMaskDataset(dataset, transform_geometric)

    # Combine all datasets
    combined_dataset = ConcatDataset(
        [
            dataset,  # Original data
            augmented_mild,  # With mild augmentations
            augmented_color,  # With color augmentations
            augmented_geometric,  # With geometric augmentations
        ]
    )

    print(f"Original dataset size: {len(dataset)}")
    print(f"Augmented dataset size: {len(combined_dataset)}")

    return combined_dataset
