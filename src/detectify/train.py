"""Detectify Training - Model training with YOLOv8."""
from pathlib import Path
from typing import Any

import yaml
from detectify.utils import logger
from detectify.model.yolo import YOLODetector

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load training configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        # Fallback/Default config creation if missing
        return {
            "data": "data.yaml",
            "epochs": 10,
            "imgsz": 640
        }
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config or {}

def run_training(config_path: str | Path) -> None:
    """
    Run YOLOv8 model training.
    
    Args:
        config_path: Path to training configuration YAML or 'data.yaml'.
    """
    # If passing a .yaml directly, treat it as the data file
    if str(config_path).endswith("data.yaml"):
        config = {"data": str(config_path), "epochs": 50, "imgsz": 640}
    else:
        config = load_config(config_path)
    
    logger.info(f"Starting Training with config: {config}")
    
    # Initialize detector
    detector = YOLODetector()
    detector.load('yolov8n.pt') # Load pretrained weights to finetune
    
    # Run training
    detector.train(None, None, config)
    
    logger.info("Training complete! Model saved to 'runs/detect/train/weights/best.pt'")

if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "data.yaml"
    run_training(config)
