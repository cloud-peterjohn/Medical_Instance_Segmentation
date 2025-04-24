import numpy as np
import skimage.io as sio
import torch
import os
import json
import numpy as np
import skimage.io as sio
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import cv2
import os
import torch
from augments import (
    Compose,
    ToTensor,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    RandomZoom,
    RandomResizedCrop,
    ColorJitter,
    GaussianNoise,
    GaussianBlur,
    RandomGamma,
    Normalize,
)


class TrainDataset(Dataset):
    def __init__(self, root_dir, image_folder="train", transform=None):
        self.root_dir = Path(root_dir)
        self.train_dir = self.root_dir / image_folder
        self.transform = transform
        self.image_folders = [f for f in self.train_dir.iterdir() if f.is_dir()]
        self.categories = {"class1": 1, "class2": 2, "class3": 3, "class4": 4}

    def __len__(self):
        return len(self.image_folders)

    def __getitem__(self, idx):
        img_folder = self.image_folders[idx]
        image_id = img_folder.name

        img_path = img_folder / "image.tif"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = {
            "image": image,
            "image_id": image_id,
            "file_name": f"{image_id}/image.tif",
        }

        masks = []
        labels = []
        boxes = []

        for class_name, class_id in self.categories.items():
            mask_path = img_folder / f"{class_name}.tif"

            if mask_path.exists():
                mask = sio.imread(str(mask_path))

                instance_ids = np.unique(mask)
                instance_ids = instance_ids[instance_ids > 0]

                for instance_id in instance_ids:
                    instance_mask = (mask == instance_id).astype(np.uint8)

                    pos = np.where(instance_mask)
                    if len(pos[0]) == 0:
                        continue

                    y_min, x_min = np.min(pos[0]), np.min(pos[1])
                    y_max, x_max = np.max(pos[0]), np.max(pos[1])

                    if (x_max > x_min) and (y_max > y_min):
                        masks.append(instance_mask)
                        labels.append(class_id)
                        boxes.append([x_min, y_min, x_max, y_max])

        if len(masks) == 0:
            result["target"] = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks": torch.zeros(
                    (0, image.shape[0], image.shape[1]), dtype=torch.uint8
                ),
            }
            print(f"Warning: No masks found for image {image_id}")
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            result["target"] = {"boxes": boxes, "labels": labels, "masks": masks}

        if self.transform:
            result = self.transform(result)

        return result


class TestDataset(Dataset):
    def __init__(self, root_dir, image_folder="test_release", transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / image_folder
        self.transform = transform

        # 读取测试图像ID映射
        id_map_file = self.root_dir / "test_image_name_to_ids.json"

        if id_map_file.exists():
            with open(id_map_file, "r") as f:
                self.image_to_id_map = json.load(f)
        else:
            self.image_to_id_map = []
            print("Warning: test_image_name_to_ids.json not found")

        self.image_paths = list(self.image_dir.glob("*.tif"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        file_name = os.path.basename(img_path)

        image_id = None
        for item in self.image_to_id_map:
            if item["file_name"] == file_name:
                image_id = item["id"]
                height = item.get("height")
                width = item.get("width")
                break

        if image_id is None:
            image_id = os.path.splitext(file_name)[0]
            height, width = None, None

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB

        if height is None or width is None:
            height, width = image.shape[:2]

        result = {
            "image": image,
            "image_id": image_id,
            "file_name": file_name,
            "height": height,
            "width": width,
        }

        if self.transform:
            result = self.transform(result)

        return result


def collate_fn(batch):
    images = []
    targets = []

    for sample in batch:
        images.append(sample["image"])
        if "target" in sample:
            targets.append(sample["target"])

    return images, targets


def collate_fn_test(batch):
    images = []
    image_ids = []

    for sample in batch:
        images.append(sample["image"])
        image_ids.append(sample["image_id"])

    return images, image_ids


def create_data_loaders(
    data_root, batch_size=2, num_workers=0 if os.name == "nt" else 4
):
    train_transforms = Compose(
        [
            ToTensor(),
            RandomHorizontalFlip(prob=0.5),
            RandomVerticalFlip(prob=0.3),
            RandomRotation(degrees=15, prob=0.4),
            RandomZoom(scale=(0.85, 1.15), prob=0.3),
            RandomResizedCrop(scale=(0.8, 1.0), ratio=(0.9, 1.1), prob=0.3),
            ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, prob=0.7
            ),
            GaussianNoise(mean=0, std=0.02, prob=0.2),
            GaussianBlur(kernel_size=3, sigma=(0.1, 1.0), prob=0.2),
            RandomGamma(gamma_range=(0.8, 1.2), prob=0.3),
            Normalize(
                mean=[0.6744110457653618, 0.5386493618583746, 0.7279701662713861],
                std=[0.17714011639969535, 0.23137328996944403, 0.19442110713209376],
            ),
        ]
    )

    test_transforms = Compose(
        [
            ToTensor(),
            Normalize(
                mean=[0.6744110457653618, 0.5386493618583746, 0.7279701662713861],
                std=[0.17714011639969535, 0.23137328996944403, 0.19442110713209376],
            ),
        ]
    )

    full_train_dataset = TrainDataset(
        root_dir=data_root, image_folder="train", transform=train_transforms
    )

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    val_transform = Compose(
        [
            ToTensor(),
            Normalize(
                mean=[0.6744110457653618, 0.5386493618583746, 0.7279701662713861],
                std=[0.17714011639969535, 0.23137328996944403, 0.19442110713209376],
            ),
        ]
    )

    val_dataset_transformed = torch.utils.data.Subset(
        TrainDataset(root_dir=data_root, image_folder="train", transform=val_transform),
        indices=val_dataset.indices,
    )

    test_dataset = TestDataset(root_dir=data_root, transform=test_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset_transformed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_test,
    )

    return train_loader, val_loader, test_loader
