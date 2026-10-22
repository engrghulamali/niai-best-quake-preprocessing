"""Task 3: cleaning."""
import pandas as pd


def epoch_ms_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'time' and 'updated' (epoch milliseconds) to UTC datetimes."""
    out = df.copy()
    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], unit="ms", utc=True)
    if "updated" in out.columns:
        out["updated"] = pd.to_datetime(out["updated"], unit="ms", utc=True)
    return out


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """One row per 'id', keeping the row with the greatest 'updated'."""
    if df.empty:
        return df.copy()
    return df.sort_values("updated", kind="mergesort").drop_duplicates("id", keep="last").reset_index(drop=True)


def keep_earthquakes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise 'type' (strip, lowercase) and keep only 'earthquake'."""
    if "type" not in df.columns:
        return df.copy()
    out = df.copy()
    out["type"] = out["type"].astype(str).str.strip().str.lower()
    return out[out["type"] == "earthquake"].reset_index(drop=True)


def extract_region(place: pd.Series) -> pd.Series:
    """Text after the last comma, or the whole string if there is no comma.
    Missing places become 'Unknown'."""
    s = place.copy()
    s = s.fillna("Unknown")
    s = s.astype(str)
    result = s.str.split(",").str[-1].str.strip()
    return result.where(s.str.contains(","), s).replace({"": "Unknown"})


def iqr_outlier_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    """True where a value lies outside [Q1 - k*IQR, Q3 + k*IQR]."""
    s = pd.to_numeric(s, errors="coerce")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (s < lower) | (s > upper)


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where 'mag' is missing."""
    if "mag" not in df.columns:
        return df.copy()
    return df.dropna(subset=["mag"]).reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Chain the steps above (think about the order) and add a 'region' column."""
    out = df.copy()
    out = epoch_ms_to_datetime(out)
    out = dedupe_latest(out)
    out = keep_earthquakes(out)
    out = drop_missing_target(out)
    if "place" in out.columns:
        out["region"] = extract_region(out["place"])
    return out.reset_index(drop=True)
