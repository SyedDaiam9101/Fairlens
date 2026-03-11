"""Tests for utility helpers."""
import pytest


class TestCameraHelpers:
    """Tests for camera enumeration helpers."""

    def test_list_cameras_returns_list(self, mock_video_capture):
        """Test list_cameras returns a list of tuples."""
        from detectify.utils import list_cameras
        
        cameras = list_cameras(max_search=3)
        
        assert isinstance(cameras, list)
        assert len(cameras) > 0
        
        for idx, name in cameras:
            assert isinstance(idx, int)
            assert isinstance(name, str)

    def test_list_cameras_with_no_devices(self, monkeypatch):
        """Test list_cameras when no devices available."""
        from unittest.mock import MagicMock
        import cv2
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        monkeypatch.setattr("cv2.VideoCapture", lambda *args, **kwargs: mock_cap)
        
        from detectify.utils import list_cameras
        
        cameras = list_cameras(max_search=3)
        assert cameras == []

    def test_ask_camera_no_cameras_raises(self, monkeypatch):
        """Test ask_camera raises when no cameras available."""
        from unittest.mock import MagicMock
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        monkeypatch.setattr("cv2.VideoCapture", lambda *args, **kwargs: mock_cap)
        
        from detectify.utils import ask_camera
        
        with pytest.raises(ValueError, match="No cameras found"):
            ask_camera(max_search=3)


class TestDirectoryHelpers:
    """Tests for directory utilities."""

    def test_ensure_dir_creates_directory(self, tmp_path):
        """Test ensure_dir creates directory."""
        from detectify.utils import ensure_dir
        
        new_dir = tmp_path / "new" / "nested" / "dir"
        assert not new_dir.exists()
        
        result = ensure_dir(new_dir)
        
        assert new_dir.exists()
        assert result == new_dir

    def test_ensure_dir_existing_directory(self, tmp_path):
        """Test ensure_dir with existing directory."""
        from detectify.utils import ensure_dir
        
        existing = tmp_path / "existing"
        existing.mkdir()
        
        result = ensure_dir(existing)
        
        assert existing.exists()
        assert result == existing


class TestDeviceHelpers:
    """Tests for device detection helpers."""

    def test_get_device_returns_string(self):
        """Test get_device returns cpu or cuda."""
        from detectify.utils import get_device
        
        device = get_device()
        assert device in ["cpu", "cuda"]

    def test_set_seed_runs_without_error(self):
        """Test set_seed doesn't raise."""
        from detectify.utils import set_seed
        
        set_seed(42)  # Should not raise
