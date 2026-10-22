"""Task 5: run the whole pipeline end to end.

Steps (see LAB_GUIDE.md):
 1. fetch all_week (fall back to config.FALLBACK_PATH if the network fails)
 2. merge any data/stream/*.jsonl rows, then dedupe_latest
 3. clean, engineer features, drop leaky columns
 4. split, fit_transform on train only, transform test
 5. print the report, save train.csv, test.csv, preprocessor.joblib
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from . import config
from .cleaning import clean, dedupe_latest
from .features import (
    add_location_features,
    add_quality_features,
    add_target,
    add_time_features,
    drop_leaky_columns,
    group_rare,
)
from .fetch import fetch_feed, geojson_to_df
from .transform import build_preprocessor, split_data


def _read_stream_jsonl() -> pd.DataFrame:
    frames = []
    stream_dir = Path(config.STREAM_DIR)
    if stream_dir.exists():
        for path in sorted(stream_dir.glob("*.jsonl")):
            try:
                frames.append(pd.read_json(path, lines=True))
            except ValueError:
                continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the earthquake feature dataset.")
    parser.add_argument("--feed", default="all_week")
    parser.add_argument("--scaler", default="robust")
    args = parser.parse_args()

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        payload = fetch_feed(args.feed)
        raw = geojson_to_df(payload)
    except Exception:
        raw = geojson_to_df(pd.read_json(config.FALLBACK_PATH))

    stream_df = _read_stream_jsonl()
    merged = pd.concat([raw, stream_df], ignore_index=True) if not stream_df.empty else raw.copy()
    merged = dedupe_latest(merged)
    cleaned = clean(merged)

    feature_df = add_time_features(cleaned)
    feature_df = add_quality_features(feature_df)
    feature_df = add_location_features(feature_df)
    if "region" in feature_df.columns:
        feature_df["region"] = group_rare(feature_df["region"], top_k=15)
    feature_df = add_target(feature_df)
    feature_df = drop_leaky_columns(feature_df)

    X_train, X_test, y_train, y_test = split_data(feature_df)
    preprocessor = build_preprocessor(scaler=args.scaler)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    train = pd.DataFrame(X_train_proc, columns=preprocessor.get_feature_names_out())
    train["big_quake"] = y_train.reset_index(drop=True).to_numpy()
    test = pd.DataFrame(X_test_proc, columns=preprocessor.get_feature_names_out())
    test["big_quake"] = y_test.reset_index(drop=True).to_numpy()

    train_path = config.PROCESSED_DIR / "train.csv"
    test_path = config.PROCESSED_DIR / "test.csv"
    model_path = config.PROCESSED_DIR / "preprocessor.joblib"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    joblib.dump(preprocessor, model_path)

    print(f"raw rows: {len(raw)}")
    print(f"rows after cleaning: {len(cleaned)}")
    print(f"stream rows merged: {len(stream_df)}")
    print(f"train shape: {X_train.shape}, positive rate: {y_train.mean():.3f}")
    print(f"test shape: {X_test.shape}, positive rate: {y_test.mean():.3f}")
    print(f"saved to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
