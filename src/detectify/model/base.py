"""Detectify Model Base - Abstract base class for detectors."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class ModelBase(ABC):
    """Abstract base class for object detection models."""

    @abstractmethod
    def load(self, model_path: str | None = None) -> "ModelBase":
        """
        Load the model weights.
        
        Args:
            model_path: Path to model or model identifier.
            
        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        Run inference on an image.
        
        Args:
            image: Input image as numpy array (BGR format, HWC).
            
        Returns:
            List of detection dictionaries, each containing:
            - class_id: int
            - class_name: str
            - confidence: float
            - bbox: dict with x1, y1, x2, y2 (pixel coordinates)
        """
        pass

    @abstractmethod
    def train(self, train_loader: Any, val_loader: Any, cfg: dict) -> None:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            cfg: Training configuration.
        """
        pass

    @abstractmethod
    def export(self, export_path: str | Path, format: str = "onnx") -> Path:
        """
        Export the model to a portable format.
        
        Args:
            export_path: Directory or file path for export.
            format: Export format (e.g., 'onnx', 'tflite', 'saved_model').
            
        Returns:
            Path to the exported model.
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        pass
