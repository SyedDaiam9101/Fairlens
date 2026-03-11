#!/usr/bin/env python3
"""Initialize Database - Create all tables for SQLite."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detectify.config import settings
from detectify.db import create_tables
from detectify.db.session import drop_tables
from detectify.utils import logger


def main():
    """Initialize the database."""
    logger.info(f"Initializing database: {settings.database_url}")
    
    # Drop existing tables to recreate with correct schema
    logger.info("Dropping existing tables...")
    try:
        drop_tables()
        logger.info("Old tables dropped.")
    except Exception as e:
        logger.warning(f"Could not drop tables (may not exist): {e}")
    
    # Create tables with current schema
    logger.info("Creating tables with current schema...")
    create_tables()
    
    logger.info("✅ Database tables created successfully!")
    logger.info("Tables: detections")
    logger.info("Schema includes: id, timestamp, camera_id, class_id, class_name, confidence, x1, y1, x2, y2, distance_cm, motion, person_detected, unauthorised, image_url, source_type, alert_sent")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
