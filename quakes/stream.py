"""Task 2: simulate streaming by polling a feed on a timer."""
import argparse
import time
from pathlib import Path

import pandas as pd

from quakes.fetch import fetch_feed, geojson_to_df


def filter_unseen(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Return only rows whose (id, updated) pair is not in `seen`.

    Add the new pairs to `seen` (modify the set in place).
    """
    if df.empty:
        return df.copy()

    new_pairs = []
    unseen_mask = []
    for _, row in df.iterrows():
        pair = (row["id"], row["updated"])
        is_new = pair not in seen
        unseen_mask.append(is_new)
        if is_new:
            new_pairs.append(pair)

    out = df.loc[unseen_mask].copy()
    seen.update(new_pairs)
    return out


def run_stream(feed: str, interval_s: int, duration_s: int, out_path: str) -> None:
    """Poll `feed` every `interval_s` seconds for `duration_s` seconds.

    Each poll: fetch, parse, keep unseen rows, append them to `out_path`
    as JSON lines, and print e.g. "[14:02:11] 3 new / 12 fetched".
    A failed poll must not stop the loop.
    """
    seen = set()
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + duration_s

    while time.monotonic() < deadline:
        try:
            payload = fetch_feed(feed)
            fetched = geojson_to_df(payload)
            new_rows = filter_unseen(fetched, seen)

            if not new_rows.empty:
                with out_file.open("a", encoding="utf-8") as handle:
                    new_rows.to_json(handle, orient="records", lines=True)
                    handle.write("\n")

            print(f"[{time.strftime('%H:%M:%S')}] {len(new_rows)} new / {len(fetched)} fetched")
        except Exception as exc:  # pragma: no cover - network variability
            print(f"[{time.strftime('%H:%M:%S')}] error: {exc}")

        time.sleep(interval_s)


def main() -> None:
    """argparse: --feed (default all_hour), --interval (60), --duration (2400),
    --out (data/stream/stream.jsonl), then call run_stream."""
    parser = argparse.ArgumentParser(description="Stream USGS earthquake events.")
    parser.add_argument("--feed", default="all_hour")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--duration", type=int, default=2400)
    parser.add_argument("--out", default="data/stream/stream.jsonl")
    args = parser.parse_args()
    run_stream(args.feed, args.interval, args.duration, args.out)


if __name__ == "__main__":
    main()
