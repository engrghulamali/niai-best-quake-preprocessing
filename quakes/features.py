"""Task 4: feature engineering."""
import pandas as pd

from . import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'hour' and 'dayofweek' (UTC) from 'time'."""
    out = df.copy()
    if "time" in out.columns:
        out["hour"] = pd.to_datetime(out["time"], utc=True).dt.hour
        out["dayofweek"] = pd.to_datetime(out["time"], utc=True).dt.dayofweek
    return out


def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add:
    update_lag_hours = hours between 'time' and 'updated'
    is_reviewed      = 1 if status == 'reviewed' else 0
    nst_missing      = 1 if 'nst' is missing else 0
    """
    out = df.copy()
    if {"time", "updated"}.issubset(out.columns):
        out["update_lag_hours"] = (
            pd.to_datetime(out["updated"], utc=True) - pd.to_datetime(out["time"], utc=True)
        ).dt.total_seconds() / 3600.0
    else:
        out["update_lag_hours"] = 0.0

    if "status" in out.columns:
        out["is_reviewed"] = out["status"].astype(str).str.lower().eq("reviewed").astype(int)
    else:
        out["is_reviewed"] = 0

    if "nst" in out.columns:
        out["nst_missing"] = out["nst"].isna().astype(int)
    else:
        out["nst_missing"] = 0

    return out


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'abs_lat' and 'is_shallow' (1 if depth_km < 70 else 0)."""
    out = df.copy()
    if "lat" in out.columns:
        out["abs_lat"] = out["lat"].abs()
    else:
        out["abs_lat"] = 0.0

    if "depth_km" in out.columns:
        out["is_shallow"] = (out["depth_km"] < 70).fillna(False).astype(int)
    else:
        out["is_shallow"] = 0

    return out


def group_rare(s: pd.Series, top_k: int = 15) -> pd.Series:
    """Keep the top_k most frequent values; replace all others with 'Other'."""
    values = s.fillna("Unknown").astype(str)
    if len(values) == 0:
        return values
    top_values = values.value_counts().head(top_k).index
    return values.where(values.isin(top_values), "Other")


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'big_quake' = 1 if mag >= 4.5 else 0."""
    out = df.copy()
    if "mag" in out.columns:
        out["big_quake"] = (out["mag"] >= 4.5).fillna(False).astype(int)
    return out


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every column listed in config.LEAKY (ignore ones that are absent)."""
    columns_to_drop = [column for column in config.LEAKY if column in df.columns]
    return df.drop(columns=columns_to_drop, errors="ignore")
