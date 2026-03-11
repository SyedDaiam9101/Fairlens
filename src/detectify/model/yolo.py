"""Detectify YOLO Detector - Ultralytics YOLOv8 based object detection."""
from pathlib import Path
from typing import Any, Optional

import numpy as np
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from detectify.config import settings
from detectify.model.base import ModelBase


class YOLODetector(ModelBase):
    """YOLOv8 object detector using Ultralytics library."""

    def __init__(self):
        """Initialize detector (model not loaded yet)."""
        self.model = None
        # Default to 'yolov8n.pt' or config value
        self.model_path = settings.yolo_model_path 
        self.confidence_threshold = settings.confidence
        self._loaded = False

    def load(self, model_path: Optional[str] = None) -> "YOLODetector":
        """
        Load the YOLO model.
        
        Args:
            model_path: Optional path to .pt file. Uses config/default if not provided.
            
        Returns:
            Self for method chaining.
        """
        if YOLO is None:
            raise ImportError(
                "Ultralytics is not installed. Run `pip install ultralytics` to use YOLO."
            )

        if model_path:
            self.model_path = model_path

        print(f"Loading YOLO model from: {self.model_path}")
        
        # Load model (auto-downloads if 'yolov8n.pt' and not found)
        self.model = YOLO(self.model_path)
        self._loaded = True
        
        print(f"Model loaded successfully! Classes: {len(self.model.names)}")
        return self

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        Run object detection on an image.
        
        Args:
            image: BGR image as numpy array (OpenCV format).
            
        Returns:
            List of detection dictionaries with keys:
            - class_id: int
            - class_name: str
            - confidence: float
            - bbox: dict with x1, y1, x2, y2 (pixel coordinates)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Run inference
        # verbose=False prevents spewing logs to stdout
        results = self.model(image, verbose=False, conf=self.confidence_threshold)
        
        detections = []
        
        # Iterate over results (usually just one for a single image)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class ID and Confidence
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = self.model.names[class_id]
                
                # Get BBox Coordinates [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                })

        return detections

    def train(self, train_loader: Any, val_loader: Any, cfg: dict) -> None:
        """
        Train the model using Ultralytics API.
        
        Args:
            train_loader: Path to data.yaml (YOLO format) usually, not a loader object.
            val_loader: specific args for val (optional).
            cfg: Dictionary with 'epochs', 'imgsz', 'data', etc.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        # Extract training args from cfg
        data_path = cfg.get("data", "coco128.yaml")
        epochs = cfg.get("epochs", 10)
        imgsz = cfg.get("imgsz", 640)
        
        print("Starting YOLO training...")
        self.model.train(data=data_path, epochs=epochs, imgsz=imgsz)
        print("Training complete.")

    def export(self, export_path: str | Path, format: str = "onnx") -> Path:
        """
        Export the model.
        
        Args:
            export_path: NOT USED strictly by YOLO export (it auto-names). 
                         But we can move the file after.
            format: 'onnx', 'tflite', 'engine' (TensorRT).
            
        Returns:
            Path to the exported model.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        print(f"Exporting model to {format}...")
        exported_filename = self.model.export(format=format)
        
        # YOLO returns the filename of the exported model
        return Path(exported_filename)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
