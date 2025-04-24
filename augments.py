from torchvision.transforms.v2 import functional as F
import numpy as np
import torch
import numpy as np
import cv2
import random
import torch
import torch.nn.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class ToTensor:
    def __call__(self, data):
        image = data["image"]

        image = F.to_image(image)
        image = F.to_dtype(image, dtype=torch.float32, scale=True)
        data["image"] = image

        return data


class RandomHorizontalFlip:
    def __init__(self, prob=0.15):
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            image = data["image"]
            data["image"] = F.hflip(image)

            if "target" in data:
                target = data["target"]
                h, w = image.shape[-2:]

                # 翻转边界框
                if "boxes" in target and len(target["boxes"]) > 0:
                    boxes = target["boxes"]
                    boxes = torch.stack(
                        [
                            w - boxes[:, 2],  # x_min becomes w - x_max
                            boxes[:, 1],  # y_min stays the same
                            w - boxes[:, 0],  # x_max becomes w - x_min
                            boxes[:, 3],  # y_max stays the same
                        ],
                        dim=1,
                    )
                    target["boxes"] = boxes

                # 翻转掩码
                if "masks" in target:
                    masks = target["masks"]
                    masks = torch.flip(masks, dims=[-1])  # 水平翻转
                    target["masks"] = masks

        return data


class RandomVerticalFlip:
    def __init__(self, prob=0.15):
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            image = data["image"]
            data["image"] = F.vflip(image)

            if "target" in data:
                target = data["target"]
                h, w = image.shape[-2:]

                # 垂直翻转边界框
                if "boxes" in target and len(target["boxes"]) > 0:
                    boxes = target["boxes"]
                    boxes = torch.stack(
                        [
                            boxes[:, 0],  # x_min stays the same
                            h - boxes[:, 3],  # y_min becomes h - y_max
                            boxes[:, 2],  # x_max stays the same
                            h - boxes[:, 1],  # y_max becomes h - y_min
                        ],
                        dim=1,
                    )
                    target["boxes"] = boxes

                # 垂直翻转掩码
                if "masks" in target:
                    masks = target["masks"]
                    masks = torch.flip(masks, dims=[-2])  # 垂直翻转
                    target["masks"] = masks

        return data


class RandomRotation:
    def __init__(self, degrees=10, prob=0.25):
        self.degrees = degrees
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            image = data["image"]
            angle = random.uniform(-self.degrees, self.degrees)
            if isinstance(image, torch.Tensor):
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image
            h, w = image_np.shape[:2]
            center = (w / 2, h / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(
                image_np,
                rotation_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            if isinstance(image, torch.Tensor):
                data["image"] = torch.from_numpy(rotated_image).permute(2, 0, 1)
            else:
                data["image"] = rotated_image

            if "target" in data:
                target = data["target"]
                if "masks" in target and len(target["masks"]) > 0:
                    masks = target["masks"]
                    rotated_masks = []
                    for i in range(masks.shape[0]):
                        mask = masks[i].cpu().numpy()
                        rotated_mask = cv2.warpAffine(
                            mask,
                            rotation_matrix,
                            (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        rotated_masks.append(rotated_mask)
                    rotated_masks = np.array(rotated_masks)
                    # 过滤无效 box
                    new_boxes = []
                    valid_indices = []
                    for idx, mask in enumerate(rotated_masks):
                        pos = np.where(mask > 0)
                        if len(pos[0]) > 0:
                            y_min, x_min = np.min(pos[0]), np.min(pos[1])
                            y_max, x_max = np.max(pos[0]), np.max(pos[1])
                            if (x_max > x_min) and (y_max > y_min):
                                new_boxes.append([x_min, y_min, x_max, y_max])
                                valid_indices.append(idx)
                    if new_boxes:
                        target["boxes"] = torch.as_tensor(
                            new_boxes, dtype=torch.float32
                        )
                        target["masks"] = torch.as_tensor(
                            rotated_masks[valid_indices], dtype=torch.uint8
                        )
                        if "labels" in target:
                            target["labels"] = target["labels"][valid_indices]
                    else:
                        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                        target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
                        if "labels" in target:
                            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        return data


class RandomResizedCrop:
    def __init__(self, scale=(0.8, 1.0), ratio=(0.9, 1.1), prob=0.25):
        self.scale = scale
        self.ratio = ratio
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            image = data["image"]
            if isinstance(image, torch.Tensor):
                h, w = image.shape[-2:]
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                h, w = image.shape[:2]
                image_np = image
            area = h * w
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            new_w = int(round(np.sqrt(target_area * aspect_ratio)))
            new_h = int(round(np.sqrt(target_area / aspect_ratio)))
            new_w = min(w, new_w)
            new_h = min(h, new_h)
            top = random.randint(0, h - new_h) if h > new_h else 0
            left = random.randint(0, w - new_w) if w > new_w else 0
            cropped_image = image_np[top : top + new_h, left : left + new_w]
            resized_image = cv2.resize(
                cropped_image, (w, h), interpolation=cv2.INTER_LINEAR
            )
            if isinstance(image, torch.Tensor):
                data["image"] = torch.from_numpy(resized_image).permute(2, 0, 1)
            else:
                data["image"] = resized_image

            if "target" in data:
                target = data["target"]
                if "masks" in target and len(target["masks"]) > 0:
                    masks = target["masks"].cpu().numpy()
                    processed_masks = []
                    for i in range(masks.shape[0]):
                        mask = masks[i]
                        cropped_mask = mask[top : top + new_h, left : left + new_w]
                        resized_mask = cv2.resize(
                            cropped_mask, (w, h), interpolation=cv2.INTER_NEAREST
                        )
                        processed_masks.append(resized_mask)
                    processed_masks = np.array(processed_masks)
                    # 过滤无效 box
                    new_boxes = []
                    valid_indices = []
                    for idx, mask in enumerate(processed_masks):
                        pos = np.where(mask > 0)
                        if len(pos[0]) > 0:
                            y_min, x_min = np.min(pos[0]), np.min(pos[1])
                            y_max, x_max = np.max(pos[0]), np.max(pos[1])
                            if (x_max > x_min) and (y_max > y_min):
                                new_boxes.append([x_min, y_min, x_max, y_max])
                                valid_indices.append(idx)
                    if new_boxes:
                        target["boxes"] = torch.as_tensor(
                            new_boxes, dtype=torch.float32
                        )
                        target["masks"] = torch.as_tensor(
                            processed_masks[valid_indices], dtype=torch.uint8
                        )
                        if "labels" in target:
                            target["labels"] = target["labels"][valid_indices]
                    else:
                        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                        target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
                        if "labels" in target:
                            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        return data


class RandomZoom:
    def __init__(self, scale=(0.9, 1.1), prob=0.25):
        self.scale = scale
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            image = data["image"]
            scale = random.uniform(self.scale[0], self.scale[1])
            if isinstance(image, torch.Tensor):
                h, w = image.shape[-2:]
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                h, w = image.shape[:2]
                image_np = image
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, 0, scale)
            zoomed_image = cv2.warpAffine(
                image_np,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            if isinstance(image, torch.Tensor):
                data["image"] = torch.from_numpy(zoomed_image).permute(2, 0, 1)
            else:
                data["image"] = zoomed_image

            if "target" in data:
                target = data["target"]
                if "masks" in target and len(target["masks"]) > 0:
                    masks = target["masks"].cpu().numpy()
                    zoomed_masks = []
                    for i in range(masks.shape[0]):
                        mask = masks[i]
                        zoomed_mask = cv2.warpAffine(
                            mask,
                            M,
                            (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        zoomed_masks.append(zoomed_mask)
                    zoomed_masks = np.array(zoomed_masks)
                    # 过滤无效 box
                    new_boxes = []
                    valid_indices = []
                    for idx, mask in enumerate(zoomed_masks):
                        pos = np.where(mask > 0)
                        if len(pos[0]) > 0:
                            y_min, x_min = np.min(pos[0]), np.min(pos[1])
                            y_max, x_max = np.max(pos[0]), np.max(pos[1])
                            if (x_max > x_min) and (y_max > y_min):
                                new_boxes.append([x_min, y_min, x_max, y_max])
                                valid_indices.append(idx)
                    if new_boxes:
                        target["boxes"] = torch.as_tensor(
                            new_boxes, dtype=torch.float32
                        )
                        target["masks"] = torch.as_tensor(
                            zoomed_masks[valid_indices], dtype=torch.uint8
                        )
                        if "labels" in target:
                            target["labels"] = target["labels"][valid_indices]
                    else:
                        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                        target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
                        if "labels" in target:
                            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        return data


class ColorJitter:
    def __init__(
        self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, prob=0.25
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            image = data["image"]

            # 应用颜色抖动
            if random.random() < 0.5:
                brightness_factor = random.uniform(
                    max(0, 1 - self.brightness), 1 + self.brightness
                )
                image = F.adjust_brightness(image, brightness_factor)

            if random.random() < 0.5:
                contrast_factor = random.uniform(
                    max(0, 1 - self.contrast), 1 + self.contrast
                )
                image = F.adjust_contrast(image, contrast_factor)

            if random.random() < 0.5:
                saturation_factor = random.uniform(
                    max(0, 1 - self.saturation), 1 + self.saturation
                )
                image = F.adjust_saturation(image, saturation_factor)

            if random.random() < 0.5:
                hue_factor = random.uniform(-self.hue, self.hue)
                image = F.adjust_hue(image, hue_factor)

            data["image"] = image

        return data


class GaussianNoise:
    def __init__(self, mean=0.0, std=0.01, prob=0.25):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            image = data["image"]

            if isinstance(image, torch.Tensor):
                noise = torch.randn_like(image) * self.std + self.mean
                data["image"] = torch.clamp(image + noise, 0, 1)
            else:
                noise = np.random.normal(self.mean, self.std, image.shape)
                noisy_image = image + noise
                data["image"] = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return data


class GaussianBlur:
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), prob=0.25):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            image = data["image"]
            sigma = random.uniform(self.sigma[0], self.sigma[1])

            if isinstance(image, torch.Tensor):
                data["image"] = F.gaussian_blur(
                    image,
                    kernel_size=[self.kernel_size, self.kernel_size],
                    sigma=[sigma, sigma],
                )
            else:
                data["image"] = cv2.GaussianBlur(
                    image, (self.kernel_size, self.kernel_size), sigma
                )

        return data


class RandomGamma:
    def __init__(self, gamma_range=(0.8, 1.2), prob=0.25):
        self.gamma_range = gamma_range
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            image = data["image"]
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])

            if isinstance(image, torch.Tensor):
                # 对张量应用伽马变换
                image = image.pow(gamma)
            else:
                # 对numpy数组应用伽马变换
                image = np.power(image / 255.0, gamma) * 255.0
                image = image.clip(0, 255).astype(np.uint8)

            data["image"] = image

        return data


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image = data["image"]
        data["image"] = F.normalize(image, mean=self.mean, std=self.std)
        return data
