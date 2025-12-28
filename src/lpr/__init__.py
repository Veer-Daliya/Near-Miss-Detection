"""License Plate Recognition module."""

from src.lpr.plate_detector import PlateDetector
from src.lpr.plate_types import PlateResult
from src.lpr.ocr import PlateOCR

__all__ = ["PlateDetector", "PlateResult", "PlateOCR"]

