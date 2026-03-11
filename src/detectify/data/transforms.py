"""Data Transforms - Augmentations for training and validation."""
from typing import Any, Callable

import cv2
import numpy as np


def resize_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    return image.astype(np.float32) / 255.0


def random_horizontal_flip(image: np.ndarray, boxes: np.ndarray, p: float = 0.5) -> tuple:
    """Randomly flip image horizontally."""
    if np.random.random() < p:
        image = cv2.flip(image, 1)
        if len(boxes) > 0:
            width = image.shape[1]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    return image, boxes


def random_brightness(image: np.ndarray, delta: float = 0.3) -> np.ndarray:
    """Randomly adjust brightness."""
    if np.random.random() < 0.5:
        factor = 1.0 + np.random.uniform(-delta, delta)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return image


def random_contrast(image: np.ndarray, lower: float = 0.8, upper: float = 1.2) -> np.ndarray:
    """Randomly adjust contrast."""
    if np.random.random() < 0.5:
        factor = np.random.uniform(lower, upper)
        mean = image.mean()
        image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    return image


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize:
    """Resize image and scale boxes."""

    def __init__(self, size: tuple[int, int]):
        self.size = size  # (width, height)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image = sample["image"]
        boxes = sample["boxes"]
        
        h, w = image.shape[:2]
        new_w, new_h = self.size
        
        # Scale factors
        scale_x = new_w / w
        scale_y = new_h / h
        
        # Resize image
        image = cv2.resize(image, self.size)
        
        # Scale boxes
        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        
        sample["image"] = image
        sample["boxes"] = boxes
        return sample


class RandomHorizontalFlip:
    """Random horizontal flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image, boxes = random_horizontal_flip(
            sample["image"], sample["boxes"], self.p
        )
        sample["image"] = image
        sample["boxes"] = boxes
        return sample


class ColorJitter:
    """Random color augmentation."""

    def __init__(self, brightness: float = 0.3, contrast: tuple = (0.8, 1.2)):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image = sample["image"]
        image = random_brightness(image, self.brightness)
        image = random_contrast(image, *self.contrast)
        sample["image"] = image
        return sample


class Normalize:
    """Normalize image."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample["image"] = normalize_image(sample["image"])
        return sample


def get_train_transforms(image_size: tuple[int, int] = (640, 640)) -> Compose:
    """Get training transforms with augmentations."""
    return Compose([
        Resize(image_size),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.3, contrast=(0.8, 1.2)),
    ])


def get_val_transforms(image_size: tuple[int, int] = (640, 640)) -> Compose:
    """Get validation transforms (no augmentation)."""
    return Compose([
        Resize(image_size),
    ])
