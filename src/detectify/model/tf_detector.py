"""Detectify TensorFlow Detector - TF-Hub based object detection."""
from pathlib import Path
from typing import Any, Optional

import numpy as np

from detectify.config import settings
from detectify.model.base import ModelBase


class TFDetector(ModelBase):
    """TensorFlow Hub-based object detector (SSD-MobileNet-V2 / EfficientDet)."""

    # COCO class names (91 classes, index 0 is background)
    COCO_CLASSES = [
        "background", "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    def __init__(self):
        """Initialize detector (model not loaded yet)."""
        self.model = None
        self.model_url = settings.tf_model_url
        self.confidence_threshold = settings.confidence
        self.iou_threshold = settings.iou_threshold
        self._loaded = False

    def load(self, model_url: Optional[str] = None) -> "TFDetector":
        """
        Load the TensorFlow Hub model.
        
        Args:
            model_url: Optional TF-Hub URL. Uses config value if not provided.
            
        Returns:
            Self for method chaining.
        """
        import tensorflow as tf
        import tensorflow_hub as hub

        if model_url:
            self.model_url = model_url

        print(f"Loading TF-Hub model from: {self.model_url}")
        
        # Set cache directory
        import os
        from detectify.utils import ensure_dir
        cache_dir = Path(settings.tfhub_cache_dir).absolute()
        ensure_dir(cache_dir)
        os.environ["TFHUB_CACHE_DIR"] = str(cache_dir)
        
        # Load model from TF-Hub
        self.model = hub.load(self.model_url)
        self._loaded = True
        
        print("Model loaded successfully!")
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
        import tensorflow as tf

        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert BGR to RGB
        image_rgb = image[..., ::-1]
        
        # Get image dimensions
        height, width = image.shape[:2]

        # Prepare input tensor
        input_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        outputs = self.model(input_tensor)

        # Extract detections
        boxes = outputs["detection_boxes"].numpy()[0]  # [N, 4] - ymin, xmin, ymax, xmax (normalized)
        scores = outputs["detection_scores"].numpy()[0]  # [N]
        classes = outputs["detection_classes"].numpy()[0].astype(int)  # [N]

        detections = []
        for i, score in enumerate(scores):
            if score >= self.confidence_threshold:
                class_id = int(classes[i])
                class_name = (
                    self.COCO_CLASSES[class_id]
                    if class_id < len(self.COCO_CLASSES)
                    else f"class_{class_id}"
                )

                # Convert normalized coords to pixel coords
                ymin, xmin, ymax, xmax = boxes[i]
                x1 = float(xmin * width)
                y1 = float(ymin * height)
                x2 = float(xmax * width)
                y2 = float(ymax * height)

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(score),
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
        Train the model (stub - requires TF Object Detection API).
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            cfg: Training configuration dictionary.
        """
        raise NotImplementedError(
            "Training with TensorFlow Object Detection API is not yet implemented. "
            "See https://tensorflow.google.cn/hub/tutorials/tf2_object_detection "
            "for guidance on fine-tuning TF-Hub detection models."
        )

    def export(self, export_path: str | Path, format: str = "saved_model") -> Path:
        """
        Export the model to a specified format.
        
        Args:
            export_path: Path to save the exported model.
            format: Export format ('saved_model', 'tflite', 'onnx').
            
        Returns:
            Path to the exported model.
        """
        import tensorflow as tf

        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)

        if format == "saved_model":
            # TF-Hub models are already SavedModels
            tf.saved_model.save(self.model, str(export_path))
            print(f"Model exported to: {export_path}")
        elif format == "tflite":
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
            tflite_path = export_path / "model.tflite"
            tflite_path.write_bytes(tflite_model)
            print(f"TFLite model exported to: {tflite_path}")
            return tflite_path
        else:
            raise ValueError(f"Unsupported export format: {format}")

        return export_path

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
