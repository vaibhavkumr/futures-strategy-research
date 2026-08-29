"""FUTURES ORDER FLOW via Databento -- the last untested lever.

On Binance the order-flow signal was real and worth ~1 basis point, against
20bp round-trip costs. Dead on arrival economically, but the SIGNAL existed:
3 of 4 symbols cleared their shuffled null, both holdouts included.

Futures change one number and only one: costs. MNQ round-trip is roughly 2bp
rather than 20bp. The same 1-2bp signal that is hopeless on Binance is
marginal-to-viable there. And CME flow is institutional rather than retail,
so the aggressor imbalance plausibly carries more information, not less.

That is the whole hypothesis, and it is falsifiable:
    does CME aggressor flow predict returns by MORE than 2bp?

Cost control matters -- $125 of credit is finite. This deliberately pulls the
cheap 'trades' schema (every print with its aggressor side) rather than full
MBO order book, which is orders of magnitude more data for information we do
not need yet. Start narrow, and only spend more if the narrow test shows
something.

The API key is read from the environment. It is never printed, logged, or
written to disk.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

DATASET = "GLBX.MDP3"           # CME Globex
SYMBOLS = ["NQ.c.0", "ES.c.0"]  # continuous front-month Nasdaq + S&P
SCHEMA = "trades"               # every trade with aggressor side; cheapest
CACHE = "dbn_cache"


def client():
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        sys.exit("DATABENTO_API_KEY not set. Set it and re-run; I never see "
                 "the value.")
    try:
        import databento as db
    except ImportError:
        sys.exit("pip install databento")
    return db.Historical(key)


def estimate_cost(start, end, symbols=SYMBOLS):
    """ALWAYS run this before downloading. $125 is finite and a careless
    request can consume it in one call."""
    c = client()
    cost = c.metadata.get_cost(dataset=DATASET, symbols=symbols,
                               schema=SCHEMA, start=start, end=end,
                               stype_in="continuous")
    size = c.metadata.get_billable_size(dataset=DATASET, symbols=symbols,
                                        schema=SCHEMA, start=start, end=end,
                                        stype_in="continuous")
    print(f"  {start} -> {end}")
    print(f"  billable size : {size/1e6:,.1f} MB")
    print(f"  COST          : ${cost:,.2f}")
    return cost


def download(start, end, symbols=SYMBOLS):
    os.makedirs(CACHE, exist_ok=True)
    tag = f"{start}_{end}".replace("-", "")
    path = f"{CACHE}/{tag}.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    c = client()
    data = c.timeseries.get_range(dataset=DATASET, symbols=symbols,
                                  schema=SCHEMA, start=start, end=end,
                                  stype_in="continuous")
    df = data.to_df()
    df.to_parquet(path)
    return df


def to_flow_bars(trades: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Aggregate raw prints into bars carrying AGGRESSOR imbalance.

    Databento marks side as 'A' (aggressor lifted the ASK -> buyer initiated)
    or 'B' (aggressor hit the BID -> seller initiated). That distinction is
    exactly what OHLC destroys, and it is what every 'smart money' claim is
    really about.
    """
    t = trades.copy()
    t.index = pd.to_datetime(t.index, utc=True)
    buy = t["size"].where(t["side"] == "A", 0.0)
    sell = t["size"].where(t["side"] == "B", 0.0)
    g = t.resample(freq)
    out = pd.DataFrame({
        "open": g["price"].first(), "high": g["price"].max(),
        "low": g["price"].min(), "close": g["price"].last(),
        "vol": g["size"].sum(),
        "ntrades": g["price"].count(),
        "buy_vol": buy.resample(freq).sum(),
        "sell_vol": sell.resample(freq).sum(),
    }).dropna()
    out["delta"] = out["buy_vol"] - out["sell_vol"]
    out["imb"] = out["delta"] / out["vol"].replace(0, np.nan)
    out["avg_size"] = out["vol"] / out["ntrades"].replace(0, np.nan)
    return out


if __name__ == "__main__":
    print("Step 1: COST ESTIMATE ONLY. Nothing is downloaded.\n")
    for start, end in [("2025-01-01", "2025-02-01"),
                       ("2025-01-01", "2025-07-01"),
                       ("2024-01-01", "2026-01-01")]:
        try:
            estimate_cost(start, end)
        except Exception as e:
            print(f"  failed: {e}")
        print()
    print("Pick the largest window the $125 covers, then run download().")
