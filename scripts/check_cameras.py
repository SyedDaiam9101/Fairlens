import sys
import cv2
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from detectify.utils.helpers import list_cameras

def main():
    print("Detectify Camera Diagnostic Tools")
    print("=" * 40)
    print(f"OS: {sys.platform}")
    print(f"OpenCV Version: {cv2.__version__}")
    print("Scanning for USB cameras (indices 0-10)...")
    cameras = list_cameras(max_search=11)
    
    if not cameras:
        print("\n[!] NO USB CAMERAS DETECTED")
    else:
        print(f"\nDetected {len(cameras)} USB camera(s):")
        for idx, name in cameras:
            print(f"  [{idx}] {name}")
            
    # Check ESP32
    from detectify.config import settings
    if settings.esp32_ip:
        print(f"\nConfigured External Camera (ESP32):")
        print(f"  [e] http://{settings.esp32_ip}:81/stream")
    else:
        print("\n[!] No ESP32-CAM configured in .env")

    print(f"\nSuggestions if your camera is missing:")
    print("1. Check if the camera is plugged in (USB) or powered on (ESP32).")
    print("2. Check if another application (Zoom, Teams, etc.) is using the camera.")
    print("3. Try a different USB port.")
    print("4. Check Privacy Settings in Windows (Camera Access).")
            
    print("\n" + "=" * 40)
    print("To run detection on a specific camera, use:")
    print("python -m detectify inference --camera <index>")

if __name__ == "__main__":
    main()
