"""Tests for FastAPI endpoints."""
import io
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Create test client with mocked detector."""
    from unittest.mock import MagicMock
    
    # Mock TFDetector
    mock_detector = MagicMock()
    mock_detector.is_loaded = True
    mock_detector.predict.return_value = [
        {
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.95,
            "bbox": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 300.0},
        }
    ]
    
    # Set temp database
    db_url = f"sqlite:///{tmp_path}/test.db"
    monkeypatch.setenv("DATABASE_URL", db_url)
    
    # Import and patch
    from detectify.api import server
    monkeypatch.setattr(server, "_detector", mock_detector)
    monkeypatch.setattr(server, "get_detector", lambda: mock_detector)
    
    # Create tables
    from detectify.db import create_tables
    create_tables()
    
    from detectify.api.server import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestDetectEndpoint:
    """Tests for /detect endpoint."""

    def test_detect_image_returns_jpeg(self, client, test_image):
        """Test POST /detect returns JPEG image."""
        # Encode test image to JPEG
        _, encoded = cv2.imencode(".jpg", test_image)
        
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", io.BytesIO(encoded.tobytes()), "image/jpeg")},
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert "X-Detection-Count" in response.headers

    def test_detect_image_returns_json(self, client, test_image):
        """Test POST /detect?format=json returns JSON."""
        _, encoded = cv2.imencode(".jpg", test_image)
        
        response = client.post(
            "/detect?format=json",
            files={"file": ("test.jpg", io.BytesIO(encoded.tobytes()), "image/jpeg")},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert "count" in data
        assert "processing_time_ms" in data
        assert len(data["detections"]) > 0

    def test_detect_invalid_file_type(self, client):
        """Test POST /detect with non-image returns 400."""
        response = client.post(
            "/detect",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        
        assert response.status_code == 400


class TestDetectionsEndpoint:
    """Tests for /detections endpoint."""

    def test_get_detections_empty(self, client):
        """Test GET /detections returns empty list initially."""
        response = client.get("/detections")
        
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

    def test_get_detections_after_detect(self, client, test_image):
        """Test GET /detections returns results after detection."""
        # First, run a detection
        _, encoded = cv2.imencode(".jpg", test_image)
        client.post(
            "/detect?format=json",
            files={"file": ("test.jpg", io.BytesIO(encoded.tobytes()), "image/jpeg")},
        )
        
        # Then check detections
        response = client.get("/detections")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1

    def test_get_detections_pagination(self, client):
        """Test GET /detections pagination parameters."""
        response = client.get("/detections?limit=10&offset=0")
        
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 0


class TestCamerasEndpoint:
    """Tests for /cameras endpoint."""

    def test_list_cameras(self, client, mock_video_capture):
        """Test GET /cameras returns camera list."""
        response = client.get("/cameras")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
