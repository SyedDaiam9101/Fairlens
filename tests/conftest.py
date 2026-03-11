"""Pytest Configuration and Fixtures."""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_image() -> np.ndarray:
    """Create a random test image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="session")  
def sample_detections() -> list[dict]:
    """Sample detection results."""
    return [
        {
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.95,
            "bbox": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 300.0},
        },
        {
            "class_id": 3,
            "class_name": "car",
            "confidence": 0.87,
            "bbox": {"x1": 300.0, "y1": 150.0, "x2": 500.0, "y2": 350.0},
        },
    ]


@pytest.fixture
def mock_video_capture(monkeypatch):
    """Mock cv2.VideoCapture for camera tests."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_cap.get.return_value = 30.0
    mock_cap.getBackendName.return_value = "DSHOW"
    
    def mock_video_capture_init(index, *args, **kwargs):
        if index < 2:
            return mock_cap
        mock_closed = MagicMock()
        mock_closed.isOpened.return_value = False
        return mock_closed
    
    monkeypatch.setattr("cv2.VideoCapture", mock_video_capture_init)
    return mock_cap


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Create a temporary SQLite database."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    
    monkeypatch.setenv("DATABASE_URL", db_url)
    
    # Re-import to pick up new URL
    from detectify.db.session import create_tables, engine
    create_tables()
    
    return db_url


@pytest.fixture
def db_session(temp_db):
    """Get a database session for testing."""
    from detectify.db.session import SessionLocal
    
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
