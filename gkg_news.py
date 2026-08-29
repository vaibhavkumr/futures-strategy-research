"""FINANCIAL NEWS SENTIMENT from GDELT's bulk archive.

The DOC API rate-limits to uselessness for historical work, but the raw GKG
archive is open. Each 15-minute file lists articles with THEMES (column 8)
and TONE (column 16). Filtering to ECON_* themes gives a financial news
sentiment reading with a real timestamp -- which is the only reason this can
be backtested at all.

We sample ONE file per trading day at 13:00 UTC (08:00 ET, pre-market). That
timestamp matters: the reading is complete before the US open, so a signal
built from it is tradeable at the open without looking into the future.

This is the first input in the entire project that is not derived from price.
"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests

OUT = "gkg"
os.makedirs(OUT, exist_ok=True)
BASE = "http://data.gdeltproject.org/gdeltv2"
STAMP = "130000"          # 13:00 UTC == 08:00 ET, before the US cash open


def day_file(day: pd.Timestamp) -> str:
    return f"{day:%Y%m%d}{STAMP}.gkg.csv.zip"


def fetch_day(day: pd.Timestamp) -> dict | None:
    """Mean tone across ECON-themed articles for one pre-market snapshot."""
    cache = f"{OUT}/{day:%Y%m%d}.pkl"
    if os.path.exists(cache):
        try:
            return pd.read_pickle(cache).iloc[0].to_dict()
        except Exception:
            pass
    url = f"{BASE}/{day_file(day)}"
    try:
        r = requests.get(url, timeout=90)
        if r.status_code != 200:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        raw = z.read(z.namelist()[0]).decode("utf-8", errors="ignore")
    except Exception:
        return None

    tones, econ_tones, n_econ, n_all = [], [], 0, 0
    for line in raw.split("\n"):
        p = line.split("\t")
        if len(p) < 17:
            continue
        themes, tone_f = p[7], p[15]
        try:
            t = float(tone_f.split(",")[0])
        except (ValueError, IndexError):
            continue
        n_all += 1
        tones.append(t)
        if "ECON_" in themes or "MARKET" in themes:
            econ_tones.append(t)
            n_econ += 1
    if n_all < 50:
        return None
    row = dict(date=day.normalize(),
               tone_all=float(np.mean(tones)),
               tone_econ=float(np.mean(econ_tones)) if econ_tones else np.nan,
               n_econ=n_econ, n_all=n_all,
               econ_share=n_econ / max(n_all, 1))
    pd.DataFrame([row]).to_pickle(cache)
    return row


def build(days) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(days):
        r = fetch_day(d)
        if r:
            rows.append(r)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(days)}  ({len(rows)} ok)", flush=True)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


if __name__ == "__main__":
    days = pd.bdate_range("2024-01-01", "2025-12-31")
    print(f"fetching {len(days)} pre-market snapshots (13:00 UTC)...")
    df = build(days)
    if len(df):
        df.to_pickle("gkg_news.pkl")
        print(f"\nsaved {len(df)} days  {df.index.min():%Y-%m-%d} -> "
              f"{df.index.max():%Y-%m-%d}")
        print(df.describe().round(3).to_string())
