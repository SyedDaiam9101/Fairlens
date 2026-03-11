"""Detectify Logging - Structured logging configuration."""
import logging
import sys
from typing import Optional

from detectify.config import settings


def setup_logger(
    name: str = "detectify",
    level: Optional[str] = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Set up a structured logger.
    
    Args:
        name: Logger name.
        level: Log level (uses config if not provided).
        json_format: Use JSON format for production.
        
    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    log_level = getattr(logging, level or settings.log_level)
    logger.setLevel(log_level)
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    if json_format:
        # JSON format for production
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"name": "%(name)s", "message": "%(message)s"}'
        )
    else:
        # Pretty format for development
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


# Default logger instance
logger = setup_logger()


def get_logger(name: str = "detectify") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
