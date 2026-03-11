"""Detectify Database CRUD Operations."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from detectify.db.models import Detection, SourceType


class DetectionCreate(BaseModel):
    """Schema for creating a detection."""

    camera_id: Optional[str] = None
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    distance_cm: Optional[float] = None
    motion: bool = False
    person_detected: bool = False
    unauthorised: bool = False
    image_url: Optional[str] = None
    source_type: SourceType = SourceType.IMAGE
    alert_sent: bool = False


class DetectionResponse(BaseModel):
    """Schema for detection response."""

    id: str
    timestamp: datetime
    camera_id: Optional[str]
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    image_url: Optional[str]
    source_type: str

    model_config = ConfigDict(from_attributes=True)


def save_detection(db: Session, event: DetectionCreate) -> Detection:
    """Save a detection event to the database."""
    detection = Detection(
        camera_id=event.camera_id,
        class_id=event.class_id,
        class_name=event.class_name,
        confidence=event.confidence,
        x1=event.x1,
        y1=event.y1,
        x2=event.x2,
        y2=event.y2,
        distance_cm=event.distance_cm,
        motion=event.motion,
        person_detected=event.person_detected,
        unauthorised=event.unauthorised,
        image_url=event.image_url,
        source_type=event.source_type,
        alert_sent=event.alert_sent,
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection


def get_detections(
    db: Session,
    camera_id: Optional[int] = None,
    class_name: Optional[str] = None,
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[Detection], int]:
    """
    Get detections with filtering and pagination.
    
    Returns tuple of (detections, total_count).
    """
    query = db.query(Detection)

    # Apply filters
    if camera_id is not None:
        query = query.filter(Detection.camera_id == camera_id)
    if class_name:
        query = query.filter(Detection.class_name.ilike(f"%{class_name}%"))
    if start_ts:
        query = query.filter(Detection.timestamp >= start_ts)
    if end_ts:
        query = query.filter(Detection.timestamp <= end_ts)

    # Get total count before pagination
    total = query.count()

    # Apply pagination and ordering
    detections = (
        query.order_by(Detection.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return detections, total


def get_detection_by_id(db: Session, detection_id: str) -> Optional[Detection]:
    """Get a single detection by ID."""
    return db.query(Detection).filter(Detection.id == detection_id).first()


def delete_detection(db: Session, detection_id: str) -> bool:
    """Delete a detection by ID."""
    detection = get_detection_by_id(db, detection_id)
    if detection:
        db.delete(detection)
        db.commit()
        return True
    return False
