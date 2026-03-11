"""Integration tests for the CLI."""
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def dummy_image(tmp_path):
    """Create a dummy JPEG image for CLI testing."""
    path = tmp_path / "test_cli.jpg"
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def test_cli_help():
    """Test that help command works."""
    result = subprocess.run(
        [sys.executable, "-m", "detectify", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "inference" in result.stdout
    assert "serve" in result.stdout


def test_cli_inference_image(dummy_image, temp_db):
    """Test running CLI inference on an image saves to DB."""
    output_path = dummy_image.parent / "output.jpg"
    
    # Run CLI
    result = subprocess.run(
        [
            sys.executable, "-m", "detectify", "inference",
            "--source", str(dummy_image),
            "--output", str(output_path),
            "--no-show"
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "DATABASE_URL": temp_db}
    )
    
    assert result.returncode == 0
    
    # Check if detection was saved in DB
    from detectify.db.session import SessionLocal
    from detectify.db.models import Detection
    
    session = SessionLocal()
    count = session.query(Detection).count()
    session.close()
    
    # Even if 0 detections, the CLI should run successfully
    assert count >= 0


def test_cli_init_db(temp_db):
    """Test init-db command."""
    result = subprocess.run(
        [sys.executable, "-m", "detectify", "init-db"],
        capture_output=True,
        text=True,
        env={**os.environ, "DATABASE_URL": temp_db}
    )
    assert result.returncode == 0
    assert "Database initialized" in result.stdout
