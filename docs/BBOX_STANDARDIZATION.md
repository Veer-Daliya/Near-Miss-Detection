# Bounding Box Standardization

## Current Standard Format

All bounding boxes in the system use the **standard [x1, y1, x2, y2] format**:
- `x1, y1`: Top-left corner coordinates
- `x2, y2`: Bottom-right corner coordinates
- Coordinates are in **pixel values** (integers)
- Coordinate system: (0, 0) is top-left corner of image

## Bounding Box Coordinate System

```
(0, 0) ────────────────> X (width)
  │
  │    ┌─────────────┐
  │    │             │
  │    │   Object    │
  │    │             │
  │    └─────────────┘
  │
  ▼
  Y (height)
```

**Example:** A bounding box `[100, 50, 300, 200]` means:
- Top-left corner: (100, 50)
- Bottom-right corner: (300, 200)
- Width: 200 pixels (300 - 100)
- Height: 150 pixels (200 - 50)

## Polygon to Bbox Conversion

OCR engines (PaddleOCR, EasyOCR) return text regions as **polygons** (4+ points), which must be converted to rectangular bounding boxes.

### Standard Conversion Function

```python
def _standardize_bbox_from_polygon(polygon) -> tuple[int, int, int, int]:
    """
    Convert polygon points to standardized [x1, y1, x2, y2] bbox.
    
    Handles various polygon formats:
    - List of [x, y] pairs: [[x1, y1], [x2, y2], ...]
    - Numpy array (n_points, 2): Standard format
    - Numpy array (2, n_points): Transposed format
    - Flat array: [x1, y1, x2, y2, ...]
    
    Returns: (x1, y1, x2, y2) where:
        x1 = min(x_coords)
        y1 = min(y_coords)
        x2 = max(x_coords)
        y2 = max(y_coords)
    """
```

### Conversion Logic

1. **Extract all x and y coordinates** from polygon points
2. **Find min/max** of x and y coordinates
3. **Create rectangle**: (min_x, min_y) to (max_x, max_y)

## Coordinate System Conversions

### ROI Coordinates to Frame Coordinates

When OCR is run on a cropped ROI, coordinates must be converted:

```python
# ROI bbox in ROI coordinate system
roi_bbox = [roi_x1, roi_y1, roi_x2, roi_y2]

# Vehicle ROI position in frame
vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = vehicle_bbox

# Convert to frame coordinates
frame_x1 = vehicle_x1 + roi_x1
frame_y1 = vehicle_y1 + roi_y1
frame_x2 = vehicle_x1 + roi_x2
frame_y2 = vehicle_y1 + roi_y2
```

## Bbox Storage Format

### JSON/JSONL Format

```json
{
  "bbox": [x1, y1, x2, y2],
  "frame_bbox": [x1, y1, x2, y2],  // In frame coordinates
  "roi_bbox": [x1, y1, x2, y2]     // In ROI coordinates (if applicable)
}
```

### PlateResult Format

```python
@dataclass
class PlateResult:
    bbox: List[int]  # [x1, y1, x2, y2] in frame coordinates
```

## Common Issues

### Issue 1: Polygon Format Mismatch
**Problem:** PaddleOCR returns polygons in different formats (numpy arrays, lists, different shapes)

**Solution:** Use standardized `_standardize_bbox_from_polygon()` function that handles all formats

### Issue 2: Coordinate System Confusion
**Problem:** Mixing ROI coordinates with frame coordinates

**Solution:** Always store both `roi_bbox` and `frame_bbox` when applicable, clearly label coordinate system

### Issue 3: Bbox Alignment Issues
**Problem:** Bounding boxes appear offset or incorrect

**Solution:** 
- Verify polygon extraction is correct
- Check coordinate system conversions
- Ensure consistent bbox format throughout pipeline

## Best Practices

1. **Always use standardized conversion function** for polygon-to-bbox conversion
2. **Store both ROI and frame coordinates** when processing ROIs
3. **Label coordinate systems clearly** in data structures
4. **Validate bbox coordinates** are within image bounds
5. **Use consistent format** [x1, y1, x2, y2] throughout the system

## Validation

Bbox coordinates should satisfy:
- `0 <= x1 < x2 <= image_width`
- `0 <= y1 < y2 <= image_height`
- `width = x2 - x1 > 0`
- `height = y2 - y1 > 0`

