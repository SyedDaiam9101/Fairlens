"""Tests for database operations."""
import pytest
from datetime import datetime, timezone


class TestDatabaseModels:
    """Tests for database models."""

    def test_detection_model_creation(self):
        """Test Detection model can be created."""
        from detectify.db import Detection, SourceType
        
        detection = Detection(
            class_id=1,
            class_name="person",
            confidence=0.95,
            x1=100.0,
            y1=100.0,
            x2=200.0,
            y2=300.0,
            source_type=SourceType.IMAGE,
        )
        
        assert detection.class_name == "person"
        assert detection.confidence == 0.95

    def test_detection_to_dict(self):
        """Test Detection.to_dict() method."""
        from detectify.db import Detection, SourceType
        
        detection = Detection(
            id="test-id",
            class_id=1,
            class_name="person",
            confidence=0.95,
            x1=100.0,
            y1=100.0,
            x2=200.0,
            y2=300.0,
            source_type=SourceType.WEBCAM,
            camera_id=0,
        )
        
        result = detection.to_dict()
        
        assert result["id"] == "test-id"
        assert result["class_name"] == "person"
        assert result["bbox"]["x1"] == 100.0
        assert result["source_type"] == "webcam"


class TestCRUDOperations:
    """Tests for CRUD operations."""

    def test_save_detection(self, db_session, sample_detections):
        """Test saving a detection."""
        from detectify.db import DetectionCreate, SourceType, save_detection
        
        det = sample_detections[0]
        event = DetectionCreate(
            class_id=det["class_id"],
            class_name=det["class_name"],
            confidence=det["confidence"],
            x1=det["bbox"]["x1"],
            y1=det["bbox"]["y1"],
            x2=det["bbox"]["x2"],
            y2=det["bbox"]["y2"],
            source_type=SourceType.IMAGE,
        )
        
        result = save_detection(db_session, event)
        
        assert result.id is not None
        assert result.class_name == "person"
        assert result.confidence == 0.95

    def test_get_detections_pagination(self, db_session, sample_detections):
        """Test getting detections with pagination."""
        from detectify.db import DetectionCreate, SourceType, save_detection, get_detections
        
        # Save multiple detections
        for det in sample_detections:
            event = DetectionCreate(
                class_id=det["class_id"],
                class_name=det["class_name"],
                confidence=det["confidence"],
                x1=det["bbox"]["x1"],
                y1=det["bbox"]["y1"],
                x2=det["bbox"]["x2"],
                y2=det["bbox"]["y2"],
                source_type=SourceType.IMAGE,
            )
            save_detection(db_session, event)
        
        # Query with limit
        detections, total = get_detections(db_session, limit=1, offset=0)
        
        assert len(detections) == 1
        assert total >= 2

    def test_get_detections_filter_by_class(self, db_session, sample_detections):
        """Test filtering detections by class name."""
        from detectify.db import DetectionCreate, SourceType, save_detection, get_detections
        
        # Save detections
        for det in sample_detections:
            event = DetectionCreate(
                class_id=det["class_id"],
                class_name=det["class_name"],
                confidence=det["confidence"],
                x1=det["bbox"]["x1"],
                y1=det["bbox"]["y1"],
                x2=det["bbox"]["x2"],
                y2=det["bbox"]["y2"],
                source_type=SourceType.IMAGE,
            )
            save_detection(db_session, event)
        
        # Filter by class
        detections, total = get_detections(db_session, class_name="person")
        
        assert all(d.class_name == "person" for d in detections)
