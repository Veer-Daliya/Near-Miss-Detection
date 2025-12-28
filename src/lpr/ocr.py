"""OCR module for extracting text from license plates."""

from typing import List, Optional

import cv2
import numpy as np

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

try:
    import easyocr
except ImportError:
    easyocr = None

from src.lpr.plate_types import PlateResult


def _bbox_points_to_rect(bbox_points: List[List[float]]) -> tuple[int, int, int, int]:
    """Convert bbox points to [x1, y1, x2, y2] rectangle coordinates."""
    x_coords = [point[0] for point in bbox_points]
    y_coords = [point[1] for point in bbox_points]
    return int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))


class PlateOCR:
    """OCR engine for license plate text extraction."""

    def __init__(self, ocr_engine: str = "paddleocr") -> None:
        """
        Initialize OCR engine.

        Args:
            ocr_engine: 'paddleocr' or 'easyocr'
        """
        self.ocr_engine = ocr_engine.lower()

        if self.ocr_engine == "paddleocr":
            if PaddleOCR is None:
                raise ImportError(
                    "PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr"
                )
            self.ocr = PaddleOCR(use_angle_cls=True, lang="en")
        elif self.ocr_engine == "easyocr":
            if easyocr is None:
                raise ImportError(
                    "EasyOCR not installed. Install with: pip install easyocr"
                )
            self.reader = easyocr.Reader(["en"], gpu=False)
        else:
            raise ValueError(f"Unknown OCR engine: {ocr_engine}")

    def extract_text(self, plate_image: np.ndarray) -> PlateResult:
        """
        Extract text from license plate image.

        Args:
            plate_image: Cropped license plate image

        Returns:
            PlateResult with text and confidence
        """
        if plate_image.size == 0:
            return PlateResult(text="UNKNOWN", confidence=0.0, bbox=[])

        # Preprocess image
        processed = self._preprocess(plate_image)

        if self.ocr_engine == "paddleocr":
            return self._extract_paddleocr(processed)
        else:
            return self._extract_easyocr(processed)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Resize if too small
        h, w = enhanced.shape
        if h < 30 or w < 100:
            scale = max(30 / h, 100 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return enhanced

    def _extract_paddleocr(self, image: np.ndarray) -> PlateResult:
        """Extract text using PaddleOCR."""
        results = self.ocr.ocr(image, cls=True)

        if not results or not results[0]:
            return PlateResult(text="UNKNOWN", confidence=0.0, bbox=[])

        # Get best result
        best_text = ""
        best_confidence = 0.0
        best_bbox = []

        for line in results[0]:
            if line:
                bbox, (text, confidence) = line
                if confidence > best_confidence:
                    best_text = text.strip()
                    best_confidence = confidence
                    # Convert bbox to [x1, y1, x2, y2]
                    best_bbox = list(_bbox_points_to_rect(bbox))

        # Clean text (remove spaces, special chars)
        cleaned_text = self._clean_plate_text(best_text)

        return PlateResult(
            text=cleaned_text if cleaned_text else "UNKNOWN",
            confidence=best_confidence,
            bbox=best_bbox,
        )

    def _extract_easyocr(self, image: np.ndarray) -> PlateResult:
        """Extract text using EasyOCR."""
        results = self.reader.readtext(image)

        if not results:
            return PlateResult(text="UNKNOWN", confidence=0.0, bbox=[])

        # Get best result
        best_text = ""
        best_confidence = 0.0
        best_bbox = []

        for bbox, text, confidence in results:
            if confidence > best_confidence:
                best_text = text.strip()
                best_confidence = confidence
                # Convert bbox to [x1, y1, x2, y2]
                best_bbox = list(_bbox_points_to_rect(bbox))

        cleaned_text = self._clean_plate_text(best_text)

        return PlateResult(
            text=cleaned_text if cleaned_text else "UNKNOWN",
            confidence=best_confidence,
            bbox=best_bbox,
        )

    def _clean_plate_text(self, text: str) -> str:
        """Clean and normalize plate text."""
        # Optimized: Use filter + join (faster than list comprehension for short strings)
        cleaned = "".join(filter(str.isalnum, text))
        return cleaned.upper()

