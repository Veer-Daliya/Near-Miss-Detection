"""License Plate Recognition module."""

from src.lpr.aggregator import PlateAggregator
from src.lpr.ocr import PlateOCR
from src.lpr.plate_detector import PlateDetector
from src.lpr.plate_types import PlateResult

__all__ = ["PlateDetector", "PlateResult", "PlateOCR", "PlateAggregator"]


