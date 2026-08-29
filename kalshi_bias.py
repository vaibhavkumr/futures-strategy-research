"""FAVOURITE-LONGSHOT BIAS on Kalshi.

The single most robustly documented anomaly in empirical finance: in betting
markets, longshots trade ABOVE their true probability and favourites BELOW.
Observed in horse racing since the 1940s and replicated in essentially every
betting market since.

If it holds on Kalshi it is mechanical money -- you do not predict anything,
you sell overpriced longshots and buy underpriced favourites, and the edge
comes from a structural behavioural bias rather than from forecasting.

And unlike Binance perps, Kalshi is CFTC-regulated and legal to trade.

METHOD
  1. Collect settled markets across many series, skipping Kalshi's
     auto-generated multi-leg parlays (KXMVE*), which are ~97% "no" by
     construction and would poison the sample.
  2. For each, take the price at a FIXED LEAD TIME before close (24h), from
     hourly candlesticks. Settlement prices are all 0 or 1 and say nothing.
  3. Bucket by price and compare price to realised YES frequency.
     A 5c contract should resolve YES 5% of the time. If it resolves 2%,
     longshots are overpriced by 3 points and selling them is the edge.
  4. Subtract real costs, then split by time so the finding has to survive
     a period it was not measured on.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
OUT = "kalshi"
LEAD_HOURS = 24
os.makedirs(OUT, exist_ok=True)


def get(path, **params):
    for _ in range(4):
        try:
            r = requests.get(f"{BASE}{path}", params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                time.sleep(2)
            else:
                return None
        except Exception:
            time.sleep(2)
    return None


def list_series():
    d = get("/series", limit=1000)
    if not d:
        return []
    out = []
    for s in d.get("series", []):
        t = s.get("ticker") or ""
        if t.startswith("KXMVE"):          # parlays -- excluded
            continue
        out.append((t, (s.get("category") or "")))
    return out


def settled_for_series(ticker, limit=200):
    d = get("/markets", series_ticker=ticker, status="settled", limit=limit)
    if not d:
        return []
    keep = []
    for m in d.get("markets", []):
        if m.get("result") not in ("yes", "no"):
            continue
        if float(m.get("volume_fp") or 0) < 50:
            continue
        keep.append(m)
    return keep


def price_at_lead(m, lead_hours=LEAD_HOURS):
    """Mid price `lead_hours` before the market closed."""
    try:
        close = pd.Timestamp(m["close_time"]).timestamp()
    except Exception:
        return None
    target = close - lead_hours * 3600
    ev = m["ticker"].split("-")[0]
    d = get(f"/series/{ev}/markets/{m['ticker']}/candlesticks",
            start_ts=int(target - 6 * 3600), end_ts=int(close),
            period_interval=60)
    if not d or not d.get("candlesticks"):
        return None
    best, bestdt = None, 1e18
    for c in d["candlesticks"]:
        ts = c.get("end_period_ts")
        if ts is None:
            continue
        dt = abs(ts - target)
        px = c.get("price", {}).get("mean_dollars")
        if px in (None, ""):
            px = c.get("price", {}).get("close_dollars")
        if px in (None, ""):
            continue
        p = float(px)
        if p <= 0 or p >= 1:
            continue
        if dt < bestdt:
            best, bestdt = p, dt
    return best


def build(max_series=4000, per_series=200):
    cache = f"{OUT}/bias_data.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    series = list_series()
    print(f"{len(series)} non-parlay series found", flush=True)
    rows = []
    for i, (tk, cat) in enumerate(series[:max_series]):
        mkts = settled_for_series(tk, per_series)
        for m in mkts:
            p = price_at_lead(m)
            if p is None:
                continue
            rows.append(dict(ticker=m["ticker"], series=tk, category=cat,
                             price=p, yes=int(m["result"] == "yes"),
                             volume=float(m.get("volume_fp") or 0),
                             close_time=m.get("close_time")))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{min(len(series),max_series)} series, "
                  f"{len(rows):,} contracts", flush=True)
        time.sleep(0.05)
        if len(rows) >= 6000:
            break
    df = pd.DataFrame(rows)
    if len(df):
        df.to_pickle(cache)
    return df


if __name__ == "__main__":
    df = build()
    print(f"\ncollected {len(df):,} settled contracts with a {LEAD_HOURS}h price")
    if len(df):
        print(df.category.value_counts().head(10).to_string())
