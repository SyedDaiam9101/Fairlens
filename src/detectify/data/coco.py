"""COCO Dataset Loader."""
import json
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np


class COCODataset:
    """COCO format dataset loader."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_file: str | Path,
        transforms: Optional[Callable] = None,
    ):
        """
        Initialize COCO dataset.
        
        Args:
            images_dir: Directory containing images.
            annotations_file: Path to COCO JSON annotations.
            transforms: Optional transform function.
        """
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.transforms = transforms
        
        # Load annotations
        with open(self.annotations_file) as f:
            self.coco = json.load(f)
        
        # Build image ID to filename mapping
        self.images = {img["id"]: img for img in self.coco.get("images", [])}
        
        # Build image ID to annotations mapping
        self.annotations_by_image: dict[int, list[dict]] = {}
        for ann in self.coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        # Build category mapping
        self.categories = {cat["id"]: cat["name"] for cat in self.coco.get("categories", [])}
        
        # List of image IDs
        self.image_ids = list(self.images.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get image and annotations by index."""
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self.images_dir / img_info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Get annotations
        anns = self.annotations_by_image.get(img_id, [])
        
        # Convert to detection format
        boxes = []
        labels = []
        
        for ann in anns:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
        
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        
        sample = {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
        }
        
        if self.transforms:
            sample = self.transforms(sample)
        
        return sample

    def get_category_name(self, category_id: int) -> str:
        """Get category name by ID."""
        return self.categories.get(category_id, f"class_{category_id}")


def load_coco_dataset(
    root_dir: str | Path,
    split: str = "train",
    year: str = "2017",
    transforms: Optional[Callable] = None,
) -> COCODataset:
    """
    Load COCO dataset with standard directory structure.
    
    Args:
        root_dir: Root directory containing images/ and annotations/.
        split: Dataset split ('train', 'val').
        year: COCO year ('2017', '2014').
        transforms: Optional transforms.
        
    Returns:
        COCODataset instance.
    """
    root = Path(root_dir)
    images_dir = root / f"{split}{year}"
    annotations_file = root / "annotations" / f"instances_{split}{year}.json"
    
    return COCODataset(images_dir, annotations_file, transforms)
