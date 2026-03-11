"""Detectify IoT - Communication with ESP32 modules."""
import requests
from detectify.config import settings
from detectify.utils.logger import logger


class IoTManager:
    """Manages communication with external ESP32 devices."""

    def __init__(self):
        self.esp32_url = None
        if settings.esp32_ip:
            self.esp32_url = f"http://{settings.esp32_ip}{settings.iot_endpoint}"

    def trigger_alert(self, class_name: str, confidence: float) -> bool:
        """
        Send an alert signal to the ESP32 dev board.
        
        Args:
            class_name: The detected class name.
            confidence: Detection confidence.
            
        Returns:
            True if successful, False otherwise.
        """
        if not settings.enable_iot or not self.esp32_url:
            return False

        try:
            data = {
                "event": "detection",
                "class": class_name,
                "confidence": round(float(confidence), 2),
                "timestamp": int(time.time() if "time" not in globals() else __import__("time").time())
            }
            
            # Non-blocking-ish request with small timeout
            response = requests.post(self.esp32_url, json=data, timeout=2.0)
            
            if response.status_code == 200:
                logger.debug(f"IoT alert sent to {settings.esp32_ip}")
                return True
            else:
                logger.warning(f"ESP32 returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to reach ESP32 at {settings.esp32_ip}: {e}")
            return False


# Global IoT manager instance
iot_manager = IoTManager()
