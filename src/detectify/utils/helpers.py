"""Detectify Utility Helpers."""
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def list_cameras(max_search: int = 10) -> list[tuple[int, str]]:
    """
    Enumerate available video capture devices.
    
    Args:
        max_search: Maximum device index to search.
        
    Returns:
        List of tuples: (device_index, device_name).
    """
    cameras = []
    
    for i in range(max_search):
        # On Windows, DirectShow is usually better for USB cameras
        # In some cases, MSMF is better for high-res built-in cameras
        # We try to open the camera to see if it exists
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        
        if not cap.isOpened() and sys.platform == "win32":
            # Try MSMF if DirectShow fails
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
            
        if cap.isOpened():
            backend = cap.getBackendName()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            name = f"Camera {i} ({backend}, {width}x{height})"
            cameras.append((i, name))
            cap.release()
    
    return cameras


def ask_camera(max_search: int = 10) -> int:
    """
    Prompt user to select a camera from available devices.
    
    Args:
        max_search: Maximum device index to search.
        
    Returns:
        Selected device index.
        
    Raises:
        ValueError: If no cameras found or invalid selection.
    """
    cameras = list_cameras(max_search)
    
    if not cameras:
        raise ValueError("No cameras found. Please connect a camera and try again.")
    
    print("\nAvailable camera sources:")
    print("-" * 50)
    for idx, name in cameras:
        print(f"  {idx} - {name}")
    
    # Check for configured ESP32 camera
    from detectify.config import settings
    esp32_url = None
    if settings.esp32_ip:
        esp32_url = f"http://{settings.esp32_ip}:81/stream"
        print(f"  e - External ESP32-CAM ({esp32_url})")
    print("-" * 50)
    
    while True:
        try:
            choice = input("\nSelect camera index or 'e' for ESP32 (or 'q' to quit): ").strip().lower()
            if choice == 'q':
                sys.exit(0)
            
            if choice == 'e' and esp32_url:
                print(f"Connecting to ESP32 stream: {esp32_url}...")
                return esp32_url
                
            camera_idx = int(choice)
            
            # Validate selection
            valid_indices = [idx for idx, _ in cameras]
            if camera_idx in valid_indices:
                # Test if we can actually open it again for capture
                cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
                if cap.isOpened():
                    cap.release()
                    return camera_idx
                else:
                    print(f"\n[!] Error: Could not open Camera {camera_idx}. It might be in use by another app.")
                    print("Please close any apps using the camera (Zoom, Teams, etc.) and try again.")
            else:
                print(f"Invalid selection. Choose from: {valid_indices}")
        except ValueError:
            print("Please enter a valid number or 'q'.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure.
        
    Returns:
        The path (for chaining).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_video_writer(
    output_path: str | Path,
    fps: float,
    frame_size: tuple[int, int],
    codec: str = "mp4v",
) -> cv2.VideoWriter:
    """
    Create a video writer with specified settings.
    
    Args:
        output_path: Path to output video file.
        fps: Frames per second.
        frame_size: (width, height) tuple.
        codec: FourCC codec string.
        
    Returns:
        OpenCV VideoWriter object.
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """
    Download a file from URL.
    
    Args:
        url: URL to download from.
        dest: Destination path.
        chunk_size: Download chunk size in bytes.
        
    Returns:
        Path to downloaded file.
    """
    import requests
    
    ensure_dir(dest.parent)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = (downloaded / total_size) * 100
                print(f"\rDownloading: {pct:.1f}%", end="", flush=True)
    
    print()
    return dest


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_crop(
    image: np.ndarray,
    bbox: dict[str, float],
    class_name: str,
    crop_dir: Path | str,
) -> Optional[str]:
    """
    Crop an object from an image and save it to disk.
    
    Args:
        image: Original image.
        bbox: Dictionary with x1, y1, x2, y2.
        class_name: Name of the detected class.
        crop_dir: Directory to save crops.
        
    Returns:
        Path to the saved crop file, or None if failed.
    """
    import uuid
    import time
    
    try:
        crop_dir = ensure_dir(Path(crop_dir))
        
        # Get coordinates
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        
        # Clip to image boundaries
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Check if crop is valid
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Perform crop
        crop = image[y1:y2, x1:x2]
        
        # Generate unique filename
        filename = f"{class_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
        save_path = crop_dir / filename
        
        # Save image
        cv2.imwrite(str(save_path), crop)
        
        return str(save_path)
    except Exception as e:
        from detectify.utils.logger import logger
        logger.error(f"Failed to save crop: {e}")
        return None


def get_device() -> str:
    """
    Get the compute device (CPU or GPU).
    
    Returns:
        'cuda' if GPU available, else 'cpu'.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        return "cuda" if gpus else "cpu"
    except ImportError:
        return "cpu"
