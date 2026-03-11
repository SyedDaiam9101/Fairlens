"""Detectify Database Models - SQLAlchemy ORM."""
import enum
import uuid

from sqlalchemy import (  # pyright: ignore[reportMissingImports]
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase  # pyright: ignore[reportMissingImports]


class SourceType(str, enum.Enum):
    """Source type for detection events."""

    IMAGE = "image"
    VIDEO = "video"
    WEBCAM = "webcam"


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class Detection(Base):
    """Detection event stored in database."""

    __tablename__ = "detections"

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        nullable=False,
    )
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    camera_id = Column(String(50), nullable=True, index=True)
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    distance_cm = Column(Float, nullable=True)
    motion = Column(Boolean, nullable=True)
    person_detected = Column(Boolean, default=False)
    unauthorised = Column(Boolean, default=False)
    image_url = Column(String(255), nullable=True)
    source_type = Column(
        Enum(SourceType),
        nullable=False,
        default=SourceType.IMAGE,
    )
    alert_sent = Column(Boolean, default=False)

    def __repr__(self) -> str:
        return (
            f"<Detection(id={self.id}, class={self.class_name}, "
            f"confidence={self.confidence:.2f})>"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "camera_id": self.camera_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": {
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2,
            },
            "image_url": self.image_url,
            "source_type": self.source_type.value if self.source_type else None,
        }
