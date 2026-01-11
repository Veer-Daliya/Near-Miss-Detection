"""GPU utility functions for memory management and optimization."""

from typing import Optional


def clear_gpu_cache(device: Optional[str] = None) -> None:
    """
    Clear GPU cache to free up memory.

    Args:
        device: Device to clear cache for ('cuda', 'mps', or None for auto-detect)
    """
    if device is None:
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                return  # No GPU available
        except ImportError:
            return

    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
    elif device == "mps":
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def get_gpu_memory_info(device: Optional[str] = None) -> Optional[dict]:
    """
    Get GPU memory information.

    Args:
        device: Device to check ('cuda', 'mps', or None for auto-detect)

    Returns:
        Dictionary with memory info or None if GPU not available
    """
    if device is None:
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                return None
        except ImportError:
            return None

    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                return {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "total_gb": total,
                    "free_gb": total - reserved,
                }
        except Exception:
            return None
    elif device == "mps":
        # MPS doesn't have detailed memory info API yet
        return {"device": "mps", "note": "Memory info not available for MPS"}

    return None


def optimize_gpu_settings(device: Optional[str] = None) -> None:
    """
    Optimize GPU settings for better performance.

    Args:
        device: Device to optimize ('cuda', 'mps', or None for auto-detect)
    """
    if device is None:
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                return
        except ImportError:
            return

    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                # Enable cuDNN benchmarking for consistent input sizes (faster)
                torch.backends.cudnn.benchmark = True
                # Allow non-deterministic algorithms for speed
                torch.backends.cudnn.deterministic = False
                # Enable TensorFloat-32 for faster training on Ampere+ GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
