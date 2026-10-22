from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FALLBACK_PATH = ROOT / "data" / "fallback" / "all_week.geojson"
STREAM_DIR = ROOT / "data" / "stream"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
TARGET = "big_quake"

NUMERIC: list[str] = [
    "lon",
    "lat",
    "depth_km",
    "nst",
    "hour",
    "dayofweek",
    "update_lag_hours",
    "abs_lat",
    "is_shallow",
    "nst_missing",
    "is_reviewed",
]
NOMINAL: list[str] = ["region", "status"]
LEAKY: list[str] = [
    "title",
    "sig",
    "mmi",
    "cdi",
    "felt",
    "alert",
    "tsunami",
    "magType",
]
