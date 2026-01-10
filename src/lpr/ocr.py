"""OCR module for extracting text from license plates."""

from typing import List

import cv2
import numpy as np

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

try:
    import easyocr  # type: ignore[reportMissingImports]
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
            # Initialize PaddleOCR with GPU support
            # Set PaddlePaddle device before initializing PaddleOCR
            gpu_available = False
            try:
                import paddle
                import os
                
                # Try to set GPU device if available
                if paddle.device.is_compiled_with_cuda():
                    try:
                        # Set PaddlePaddle to use GPU
                        paddle.device.set_device("gpu")
                        gpu_available = True
                        # Also set environment variable for PaddleOCR
                        os.environ["USE_GPU"] = "1"
                    except Exception:
                        # GPU not available or error setting device
                        paddle.device.set_device("cpu")
                        gpu_available = False
                else:
                    # Check if we can use CPU with optimizations
                    paddle.device.set_device("cpu")
                    gpu_available = False
            except (ImportError, AttributeError):
                # PaddlePaddle not available or error
                gpu_available = False
            
            # Initialize PaddleOCR
            # PaddleOCR will use GPU if PaddlePaddle device is set to GPU
            # Note: use_angle_cls is deprecated in newer versions
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang="en")
            except (TypeError, ValueError):
                # Fallback for newer PaddleOCR versions
                self.ocr = PaddleOCR(lang="en")
            
            if gpu_available:
                print("PaddleOCR initialized with GPU acceleration")
            else:
                print("PaddleOCR initialized (CPU mode)")
                print("  Note: For GPU support, install: pip install paddlepaddle-gpu")
                print("  (Note: GPU support requires CUDA, not available on macOS)")
        elif self.ocr_engine == "easyocr":
            if easyocr is None:
                raise ImportError(
                    "EasyOCR not installed. Install with: pip install easyocr"
                )
            # Auto-detect GPU availability for EasyOCR
            # EasyOCR supports CUDA (NVIDIA) and can use PyTorch MPS (Apple Silicon)
            use_gpu = False
            gpu_type = None
            try:
                import torch
                if torch.cuda.is_available():
                    use_gpu = True
                    gpu_type = "CUDA"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    # Apple Silicon MPS support
                    # Note: EasyOCR's gpu parameter only works for CUDA
                    # For MPS, we need to let PyTorch handle it automatically
                    use_gpu = False  # EasyOCR doesn't support MPS via gpu=True
                    gpu_type = "MPS (Apple Silicon)"
                    # PyTorch will automatically use MPS if available
                    print("EasyOCR initialized (PyTorch MPS backend will be used automatically)")
                else:
                    use_gpu = False
                    gpu_type = None
            except ImportError:
                use_gpu = False
                gpu_type = None
            
            # Initialize EasyOCR
            # Note: gpu=True only works for CUDA, not MPS
            # For Apple Silicon, PyTorch will use MPS automatically if available
            self.reader = easyocr.Reader(["en"], gpu=use_gpu)
            if use_gpu:
                print(f"EasyOCR initialized with {gpu_type} GPU acceleration")
            elif gpu_type == "MPS (Apple Silicon)":
                print("EasyOCR initialized (will use Apple Silicon GPU via PyTorch MPS)")
            else:
                print("EasyOCR initialized with CPU")
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
        # Run PaddleOCR - cls parameter not supported in this version
        results = self.ocr.ocr(image)

        if not results:
            return PlateResult(text="UNKNOWN", confidence=0.0, bbox=[])

        # Handle new PaddleOCR API (OCRResult object) vs old API (list)
        ocr_result = results[0] if isinstance(results, list) else results
        
        # Check if it's the new OCRResult format
        if hasattr(ocr_result, 'text_lines') or hasattr(ocr_result, 'rec_res'):
            # New API format - extract from OCRResult object
            if hasattr(ocr_result, 'text_lines') and ocr_result.text_lines:
                text_lines = ocr_result.text_lines
            elif hasattr(ocr_result, 'rec_res') and ocr_result.rec_res:
                text_lines = ocr_result.rec_res
            else:
                return PlateResult(text="UNKNOWN", confidence=0.0, bbox=[])
        elif isinstance(ocr_result, list):
            # Old API format - list of results
            text_lines = ocr_result
        else:
            return PlateResult(text="UNKNOWN", confidence=0.0, bbox=[])

        # Get best result
        best_text = ""
        best_confidence = 0.0
        best_bbox = []

        for line in text_lines:
            if not line:
                continue
            
            # Handle different line formats
            if isinstance(line, tuple) and len(line) == 2:
                bbox, (text, confidence) = line
            elif isinstance(line, dict):
                # New format might be dict
                bbox = line.get('bbox', [])
                text = line.get('text', '')
                confidence = line.get('confidence', 0.0)
            else:
                continue
                
            if confidence > best_confidence:
                best_text = text.strip()
                best_confidence = confidence
                # Convert bbox to [x1, y1, x2, y2]
                if isinstance(bbox, list) and len(bbox) > 0:
                    best_bbox = list(_bbox_points_to_rect(bbox))
                else:
                    continue

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

