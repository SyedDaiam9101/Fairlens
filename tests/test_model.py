"""Tests for TFDetector model."""
import pytest
import numpy as np


class TestTFDetector:
    """Tests for TensorFlow detector."""

    def test_detector_initialization(self):
        """Test detector can be instantiated."""
        from detectify.model import TFDetector
        
        detector = TFDetector()
        assert detector is not None
        assert not detector.is_loaded

    def test_detector_predict_without_load_raises(self, test_image):
        """Test that predict raises if model not loaded."""
        from detectify.model import TFDetector
        
        detector = TFDetector()
        
        with pytest.raises(RuntimeError, match="not loaded"):
            detector.predict(test_image)

    @pytest.mark.skip(reason="Requires TensorFlow and network access")
    def test_detector_load_and_predict(self, test_image):
        """Test loading model and running prediction."""
        from detectify.model import TFDetector
        
        detector = TFDetector()
        detector.load()
        
        assert detector.is_loaded
        
        detections = detector.predict(test_image)
        
        assert isinstance(detections, list)
        for det in detections:
            assert "class_id" in det
            assert "class_name" in det
            assert "confidence" in det
            assert "bbox" in det
            assert all(k in det["bbox"] for k in ["x1", "y1", "x2", "y2"])

    def test_train_not_implemented(self):
        """Test that train raises NotImplementedError."""
        from detectify.model import TFDetector
        
        detector = TFDetector()
        
        with pytest.raises(NotImplementedError):
            detector.train(None, None, {})

    def test_coco_classes_defined(self):
        """Test that COCO classes are defined."""
        from detectify.model import TFDetector
        
        assert len(TFDetector.COCO_CLASSES) > 80
        assert TFDetector.COCO_CLASSES[0] == "background"
        assert TFDetector.COCO_CLASSES[1] == "person"
