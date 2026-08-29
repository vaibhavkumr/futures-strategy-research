"""Collect SETTLED Kalshi markets to test for favourite-longshot bias.

Why prediction markets are worth testing when everything else failed:

  * The order books are retail-dominated. There is no market-making arms race
    on "will it rain in Chicago", so the microstructure edge that made
    scalping impossible does not exist here in the same way.
  * FAVOURITE-LONGSHOT BIAS is one of the most robustly documented anomalies
    in all of empirical finance -- observed in horse racing since the 1940s
    and in betting markets ever since. Longshots trade ABOVE their true
    probability, favourites BELOW. If it shows up here, it is mechanical and
    exploitable without predicting anything.
  * Kalshi is CFTC-regulated, so unlike Binance perps it is actually legal
    and accessible.

The test: bucket contracts by price, then compare the price to how often
that bucket actually resolved YES. A 5c contract should win 5% of the time.
If it wins 2%, longshots are overpriced and the edge is selling them.

Prices are taken at a FIXED LEAD TIME before expiry, not at settlement --
settlement prices are all 0 or 1 and would tell us nothing.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
OUT = "kalshi"
os.makedirs(OUT, exist_ok=True)


def get(path, **params):
    for attempt in range(4):
        try:
            r = requests.get(f"{BASE}{path}", params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(3)
        except Exception:
            time.sleep(2)
    return None


def collect_markets(target=8000):
    """Page through settled markets, keeping only ones that actually traded."""
    cache = f"{OUT}/settled.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    rows, cursor, seen = [], None, 0
    while len(rows) < target:
        d = get("/markets", limit=1000, status="settled",
                **({"cursor": cursor} if cursor else {}))
        if not d or not d.get("markets"):
            break
        for m in d["markets"]:
            seen += 1
            vol = float(m.get("volume_fp") or 0)
            res = m.get("result")
            if vol < 100 or res not in ("yes", "no"):
                continue
            rows.append(dict(
                ticker=m["ticker"], title=(m.get("title") or "")[:90],
                result=res, volume=vol,
                open_time=m.get("open_time"), close_time=m.get("close_time"),
                last_price=float(m.get("last_price_dollars") or 0),
                prev_price=float(m.get("previous_price_dollars") or 0),
                oi=float(m.get("open_interest_fp") or 0),
            ))
        cursor = d.get("cursor")
        if not cursor:
            break
        print(f"  scanned {seen:,}  kept {len(rows):,}", flush=True)
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df.to_pickle(cache)
    print(f"scanned {seen:,} settled markets, kept {len(df):,} with volume>=100")
    return df


def candles(ticker, start_ts, end_ts, period=60):
    """Hourly candlesticks so we can price a market at a fixed lead time."""
    ev = ticker.split("-")[0]
    d = get(f"/series/{ev}/markets/{ticker}/candlesticks",
            start_ts=start_ts, end_ts=end_ts, period_interval=period)
    return d.get("candlesticks") if d else None


if __name__ == "__main__":
    df = collect_markets()
    print(df[["result", "volume"]].describe().to_string())
    print("\nresult mix:", df.result.value_counts().to_dict())
    print(df.head(5)[["title", "result", "volume", "prev_price"]].to_string())
