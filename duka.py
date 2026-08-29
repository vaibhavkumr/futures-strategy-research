"""Shared Dukascopy loader.

dukascopy-node emits EITHER ISO strings or epoch-milliseconds depending on
version/flags. Parsing ms as ns silently dates everything to 1969, the rows
then fail every date filter, and a market quietly runs with NO data instead
of raising. That is exactly what happened to S&P/DOW/DAX. Detect and handle
both, and refuse to return a frame that is obviously wrong.
"""
import glob, os
import pandas as pd


def load(slug: str, tf: str, tag: str = "2026-07-24") -> pd.DataFrame:
    fs = [f for f in glob.glob(f"download/{slug}-{tf}-bid-*.csv") if tag in f]
    if not fs:
        raise FileNotFoundError(f"{slug} {tf}")
    d = pd.read_csv(max(fs, key=os.path.getsize))
    d.columns = [c.lower() for c in d.columns]
    ts = d["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        idx = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        idx = pd.to_datetime(ts, utc=True)
    d.index = idx.dt.tz_convert("America/New_York")
    d = d[["open", "high", "low", "close"]].astype(float).sort_index()
    if d.index[0].year < 2000:
        raise ValueError(f"{slug} {tf}: timestamps parsed to {d.index[0]} -- bad units")
    return d
