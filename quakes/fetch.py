"""Task 1: fetch the USGS GeoJSON feed and turn it into a DataFrame."""
import pandas as pd
import requests

BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"


def fetch_feed(feed: str = "all_week") -> dict:
    """GET {BASE}/{feed}.geojson and return the parsed dict.

    Use a timeout and raise_for_status(). A misspelled feed name returns
    HTTP 200 with a plain-text body, so raise a clear error if the body
    is not valid JSON.
    """
    url = f"{BASE}/{feed}.geojson"
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(f"Invalid GeoJSON response from {url}: expected JSON data.") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid GeoJSON response from {url}: expected a JSON object.")

    return payload


def geojson_to_df(payload: dict) -> pd.DataFrame:
    """One row per event.

    Columns: 'id' (top level of each feature), every key in 'properties',
    plus 'lon', 'lat', 'depth_km' from geometry.coordinates = [lon, lat, depth].
    """
    rows = []
    for feature in payload.get("features", []):
        properties = feature.get("properties", {}) or {}
        geometry = feature.get("geometry", {}) or {}
        coords = geometry.get("coordinates", [None, None, None])

        row = {"id": feature.get("id")}
        row.update(properties)
        row["lon"] = coords[0] if len(coords) > 0 else None
        row["lat"] = coords[1] if len(coords) > 1 else None
        row["depth_km"] = coords[2] if len(coords) > 2 else None
        rows.append(row)

    return pd.DataFrame(rows)
