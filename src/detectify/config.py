"""Detectify Configuration - Pydantic Settings."""
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Project paths
    project_root: Path = Path(__file__).parent.parent.parent.parent
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    # Runtime settings
    device: Literal["cpu", "cuda"] = "cpu"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # YOLO model settings
    yolo_model_path: str = "yolov8n.pt"
    confidence_threshold: float = 80.0
    confidence: float = 0.5
    iou_threshold: float = 0.5

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database settings
    database_url: str = "sqlite:///detectify.db"

    # Detection logging
    save_crops: bool = False
    crop_dir: str = "crops"

    # Notification settings
    enable_notifications: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None  # Use App Password for Gmail
    notification_recipient: str | None = None
    notification_cooldown: int = 300  # seconds between emails

    # IoT / ESP32 Settings
    enable_iot: bool = False
    esp32_ip: str | None = None
    iot_endpoint: str = "/alert"  # Default endpoint on ESP32


# Global settings instance
settings = Settings()
