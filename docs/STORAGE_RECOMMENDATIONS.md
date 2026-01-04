# Data Storage Recommendations for Per-Frame JSON Outputs

## Current Implementation ✅

The system now saves per-frame data incrementally using **JSON Lines format (.jsonl)**:
- `yolo_detections.jsonl` - YOLO object detection results
- `ocr_bboxes.jsonl` - OCR text detection results

Both files use **append-only writes** (one JSON object per line) for maximum performance and resilience. Each frame/image is written as a single line, making it fast and crash-resistant.

## Storage Options & Recommendations

### Option 1: Single Incremental JSON Files (Current Approach) ✅ **RECOMMENDED FOR SMALL-MEDIUM SCALE**

**Pros:**
- Simple to implement and read
- Human-readable format
- Easy to parse with any JSON library
- No additional dependencies
- Works well for videos up to ~10,000 frames

**Cons:**
- Entire file must be read/written for each update (slower for large files)
- Risk of data loss if process crashes mid-write
- File size grows linearly with frames

**Best For:** Development, testing, small to medium videos (< 1 hour)

**Location:** `data/outputs/yolo_detections.json` and `data/outputs/ocr_bboxes.json`

---

### Option 2: JSON Lines Format (.jsonl) ⭐ **CURRENT STANDARD**

**Format:** One JSON object per line (newline-delimited JSON)

**Pros:**
- Append-only (very fast writes)
- Can process line-by-line without loading entire file
- More resilient to crashes (each line is independent)
- Standard format for streaming data
- Easy to parallelize processing
- No need to read entire file for updates

**Cons:**
- Not as human-readable (no pretty formatting)
- Requires streaming parser for reading

**Example:**
```jsonl
{"frame_id": 1, "timestamp": 0.0, "detections": [...]}
{"frame_id": 2, "timestamp": 0.033, "detections": [...]}
```

**Best For:** ✅ **ALL USE CASES** - This is now the standard format

**Location:** `data/outputs/yolo_detections.jsonl` and `data/outputs/ocr_bboxes.jsonl`

**Reading JSONL files:**
```python
# Read line by line
with open('yolo_detections.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        # Process data

# Or load all at once (for small files)
all_data = [json.loads(line) for line in open('yolo_detections.jsonl')]
```

---

### Option 3: Directory Structure (One File Per Frame)

**Structure:**
```
data/outputs/
  yolo_detections/
    frame_000001.json
    frame_000002.json
    ...
  ocr_bboxes/
    frame_000001.json
    frame_000002.json
    ...
```

**Pros:**
- Parallel processing friendly
- Easy to delete/process specific frames
- No file locking issues
- Can compress old frames

**Cons:**
- Many small files (filesystem overhead)
- Harder to query across frames
- More complex directory management

**Best For:** Distributed processing, archival systems, frame-level analysis

**Location:** `data/outputs/yolo_detections/` and `data/outputs/ocr_bboxes/`

---

### Option 4: SQLite Database 🗄️ **RECOMMENDED FOR QUERYING**

**Structure:**
- Single database file with tables for YOLO and OCR results
- Indexed by frame_id and timestamp

**Pros:**
- Fast queries (SQL)
- ACID transactions (data integrity)
- Efficient storage (compression)
- Easy filtering, aggregation, joins
- Standard tooling support

**Cons:**
- Requires SQL knowledge
- Additional dependency (sqlite3)
- Less human-readable

**Example Schema:**
```sql
CREATE TABLE yolo_detections (
    frame_id INTEGER,
    timestamp REAL,
    bbox TEXT,  -- JSON array
    class_name TEXT,
    confidence REAL,
    track_id INTEGER
);

CREATE TABLE ocr_results (
    frame_id INTEGER,
    timestamp REAL,
    bbox TEXT,
    text TEXT,
    confidence REAL,
    ocr_engine TEXT,
    all_candidates TEXT  -- JSON array
);
```

**Best For:** Analysis, reporting, complex queries, production systems

**Location:** `data/outputs/detections.db`

---

### Option 5: Parquet Format 📊 **RECOMMENDED FOR ANALYTICS**

**Format:** Columnar storage format (Apache Parquet)

**Pros:**
- Highly compressed (10-100x smaller than JSON)
- Fast columnar queries
- Schema evolution support
- Industry standard for analytics
- Works with pandas, Spark, etc.

**Cons:**
- Requires pandas/pyarrow dependency
- Not human-readable
- Append-only (write new file periodically)

**Best For:** Data analysis, machine learning pipelines, large-scale processing

**Location:** `data/outputs/yolo_detections.parquet` and `data/outputs/ocr_bboxes.parquet`

---

### Option 6: Time-Series Database (InfluxDB/TimescaleDB) ⏱️

**Pros:**
- Optimized for time-series data
- Built-in aggregation functions
- Efficient compression
- Real-time dashboards

**Cons:**
- Requires separate database server
- More complex setup
- Overkill for simple use cases

**Best For:** Real-time monitoring, dashboards, IoT applications

---

## Recommended Storage Strategy by Use Case

### Development & Testing ✅
**Use:** Option 2 (JSON Lines) - **CURRENT STANDARD**
- Fast writes, resilient, scalable
- Can still read with any text editor (one object per line)
- Location: `data/outputs/` with `.jsonl` extension

### Production (Small-Medium Scale) ✅
**Use:** Option 2 (JSON Lines) - **CURRENT STANDARD**
- Fast writes, resilient, scalable
- Location: `data/outputs/` with `.jsonl` extension

### Production (Large Scale)
**Use:** Option 2 (JSON Lines) for writes, Option 4 (SQLite) or Option 5 (Parquet) for analysis
- JSON Lines for real-time writes
- SQLite for querying and reporting (import from .jsonl)
- Parquet for analytics and ML pipelines (convert from .jsonl)
- Location: `data/outputs/detections.db` or `.parquet` files (converted from .jsonl)

### Archival & Long-term Storage
**Use:** Option 3 (Directory structure) + Compression
- One file per frame, compressed with gzip
- Easy to archive specific time ranges
- Location: `data/archive/YYYY-MM-DD/`

---

## Implementation Notes

### Current Implementation (Option 2 - JSON Lines) ✅
- Files are updated incrementally after each frame using append-only writes
- Each frame is written as a single line (no need to read entire file)
- Very fast and efficient, suitable for videos of any length
- Crash-resistant (each line is independent)

### Performance Considerations
- **Small videos (< 1000 frames):** Current approach is fine
- **Medium videos (1000-10,000 frames):** Consider JSON Lines
- **Large videos (> 10,000 frames):** Use SQLite or Parquet

### Data Retention
- Consider implementing rotation/archival:
  - Keep last N frames in active files
  - Archive older data to compressed files
  - Delete data older than X days

### Backup Strategy
- JSON files can be easily backed up
- Consider incremental backups for large datasets
- Use version control for configuration, not data files

---

## Migration Path

If you need to switch storage formats later:

1. **JSON → JSON Lines:** Simple script to convert
2. **JSON → SQLite:** Use pandas or direct SQL inserts
3. **JSON → Parquet:** Use pandas DataFrame
4. **Any → Archive:** Compress and move to archive directory

---

## File Size Estimates

For a 1-hour video at 30 FPS (108,000 frames):

| Format | Estimated Size | Notes |
|--------|---------------|-------|
| JSON (pretty) | ~500 MB - 2 GB | Depends on detection density |
| JSON Lines | ~300 MB - 1.5 GB | Slightly smaller (no indentation) |
| SQLite | ~200 MB - 800 MB | Compressed, indexed |
| Parquet | ~50 MB - 200 MB | Highly compressed columnar |
| Directory (gzip) | ~150 MB - 600 MB | Compressed individual files |

---

## Recommendations Summary

**Current Standard:** ✅ **Option 2 (JSON Lines)** - `.jsonl` format
- Fast append-only writes
- Crash-resistant
- Scalable to any video length
- Easy to process line-by-line

**When you need advanced features:**
1. ✅ **JSON Lines** is already the standard (no change needed)
2. Add **SQLite** if you need querying capabilities (import from .jsonl)
3. Use **Parquet** for analytics/ML workflows (convert from .jsonl)

**Storage Location:**
- Active data: `data/outputs/yolo_detections.jsonl` and `data/outputs/ocr_bboxes.jsonl`
- Archived data: `data/archive/YYYY-MM-DD/` (can compress .jsonl files)
- Temporary: `data/temp/` (auto-cleanup)

