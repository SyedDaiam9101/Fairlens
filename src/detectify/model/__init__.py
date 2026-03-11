"""Detectify Model Package."""
from detectify.model.base import ModelBase
from detectify.model.tf_detector import TFDetector

__all__ = ["ModelBase", "TFDetector"]
