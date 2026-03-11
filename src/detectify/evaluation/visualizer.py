"""Detectify Visualizer - Drawing and overlay utilities."""
from typing import Any, Optional

import cv2
import numpy as np


# Color palette for different classes (BGR format)
COLORS = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 255),   # Purple
    (255, 128, 0),   # Orange
    (0, 128, 255),   # Light blue
    (128, 255, 0),   # Lime
]


def get_color(class_id: int) -> tuple[int, int, int]:
    """Get a consistent color for a class ID."""
    return COLORS[class_id % len(COLORS)]


def draw_boxes(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    thickness: int = 2,
    font_scale: float = 0.6,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image.
    
    Args:
        image: BGR image as numpy array.
        detections: List of detection dicts with keys:
            - class_id, class_name, confidence, bbox
        thickness: Box line thickness.
        font_scale: Label font scale.
        show_confidence: Whether to show confidence score.
        
    Returns:
        Image with boxes drawn.
    """
    output = image.copy()
    
    for det in detections:
        class_id = det.get("class_id", 0)
        class_name = det.get("class_name", "unknown")
        confidence = det.get("confidence", 0.0)
        bbox = det.get("bbox", {})
        
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))
        
        color = get_color(class_id)
        
        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        if show_confidence:
            label = f"{class_name}: {confidence:.2f}"
        else:
            label = class_name
        
        # Get label size for background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        
        # Draw label background
        label_y = max(y1 - 10, label_h + 10)
        cv2.rectangle(
            output,
            (x1, label_y - label_h - 5),
            (x1 + label_w + 5, label_y + 5),
            color,
            -1,
        )
        
        # Draw label text
        cv2.putText(
            output,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    
    return output


def overlay_fps(
    frame: np.ndarray,
    fps: float,
    position: tuple[int, int] = (10, 30),
) -> np.ndarray:
    """
    Overlay FPS counter on frame.
    
    Args:
        frame: Image to draw on.
        fps: Current FPS value.
        position: (x, y) position for text.
        
    Returns:
        Frame with FPS overlay.
    """
    output = frame.copy()
    fps_text = f"FPS: {fps:.1f}"
    
    cv2.putText(
        output,
        fps_text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    
    return output


def overlay_detection_count(
    frame: np.ndarray,
    count: int,
    position: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Overlay detection count on frame.
    
    Args:
        frame: Image to draw on.
        count: Number of detections.
        position: (x, y) position. Defaults to top-right.
        
    Returns:
        Frame with count overlay.
    """
    output = frame.copy()
    h, w = output.shape[:2]
    
    if position is None:
        position = (w - 150, 30)
    
    text = f"Objects: {count}"
    cv2.putText(
        output,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    
    return output


def create_detection_summary(
    detections: list[dict[str, Any]],
    width: int = 300,
    height: int = 200,
) -> np.ndarray:
    """
    Create a summary panel showing detection statistics.
    
    Args:
        detections: List of detections.
        width: Panel width.
        height: Panel height.
        
    Returns:
        Summary image.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # Dark gray background
    
    # Count classes
    class_counts: dict[str, int] = {}
    for det in detections:
        name = det.get("class_name", "unknown")
        class_counts[name] = class_counts.get(name, 0) + 1
    
    # Draw header
    cv2.putText(
        panel, "Detection Summary", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    cv2.line(panel, (10, 35), (width - 10, 35), (100, 100, 100), 1)
    
    # Draw class counts
    y = 55
    for i, (name, count) in enumerate(sorted(class_counts.items())):
        if y > height - 20:
            break
        text = f"{name}: {count}"
        cv2.putText(
            panel, text, (15, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        y += 22
    
    return panel
