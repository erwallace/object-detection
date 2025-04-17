from PIL import Image
import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from utils import parse_xml


class MaskedFaceDataset(Dataset):
    def __init__(self, root, img_transforms=[T.ToTensor()], target_transforms=None):
        super(MaskedFaceDataset, self).__init__()
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
