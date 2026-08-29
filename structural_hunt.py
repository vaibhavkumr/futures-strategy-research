"""HUNTING FOR MORE STRUCTURAL EDGES.

The calendar effect works (Sharpe 1.11) because it is not a forecast -- it is
mechanical money moving on a schedule. This hunts for more of the same
species. Every hypothesis below was documented by someone else, on other
data, before I touched anything, so this is out-of-sample replication rather
than mining.

  PRE-FOMC DRIFT      Lucca & Moench (Journal of Finance, 2015). Equity
                      returns are strongly positive in the 24h BEFORE
                      scheduled Fed announcements. One of the largest
                      calendar anomalies ever documented.
  HOLIDAY EFFECT      Ariel (1990). Returns before market holidays are
                      abnormally positive.
  OPTION EXPIRY WEEK  Dealer hedging and pinning flows into the third
                      Friday.
  QUARTERLY REBALANCE Index funds must trade the rebalance. Third Friday of
                      Mar/Jun/Sep/Dec ("quad witching").
  TURN OF YEAR        Tax-loss selling reverses in January.
  DAY OF MONTH        The full profile, to see whether turn-of-month is the
                      only pocket.

Costs 0.5bp round trip. Each is a one-trade-per-event, hold-the-session bet,
so friction is small relative to the move.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

COST_BP = 0.5
OPEN, CLOSE = 9*60+30, 16*60
MK = {"S&P 500": "usa500idxusd", "NASDAQ": "usatechidxusd",
      "DOW": "usa30idxusd", "DAX": "deuidxeur"}

# Scheduled FOMC announcement dates 2022-2026 (2-day meetings, decision day).
FOMC = [
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
]
US_HOLIDAYS = [
    "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20",
    "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-26",
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29",
    "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03",
]


def load(slug):
    fs = [f for f in glob.glob(f"download/{slug}-m5-bid-*.csv") if "2026-07-24" in f]
    d = pd.read_csv(max(fs, key=os.path.getsize))
    d.columns = [c.lower() for c in d.columns]
    ts = d["timestamp"]
    idx = (pd.to_datetime(ts, unit="ms", utc=True)
           if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts, utc=True))
    d.index = idx.dt.tz_convert("America/New_York")
    d = d[["open", "high", "low", "close"]].astype(float).sort_index()
    assert d.index[0].year > 2000
    return d


def sessions(slug):
    d = load(slug)
    m = d.index.hour*60 + d.index.minute
    k = (m >= OPEN) & (m < CLOSE)
    d = d[k]
    day = d.index.normalize().tz_localize(None)
    o = d.groupby(day)["open"].first()
    c = d.groupby(day)["close"].last()
    S = pd.DataFrame({"r": (c/o - 1)*1e4})
    S["dow"] = S.index.dayofweek
    S["dom"] = S.index.day
    S["month"] = S.index.month
    # trading-day position within the month
    S["tdom"] = S.groupby([S.index.year, S.index.month]).cumcount() + 1
    S["tdom_rev"] = S.groupby([S.index.year, S.index.month]).cumcount(ascending=False) + 1
    return S


def rep(x, lab, ann=252, cost=COST_BP):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)] - cost
    if len(x) < 25:
        print(f"  {lab:<30} n={len(x)}  (too few)")
        return None
    m, sd = x.mean(), x.std(ddof=1)
    se = sd/np.sqrt(len(x))
    sh = m/sd*np.sqrt(ann) if sd > 0 else 0
    print(f"  {lab:<30} n={len(x):<5} {m:+7.2f}bp  t={m/se:+6.2f}  "
          f"win {(x>0).mean()*100:5.1f}%  Sharpe {sh:+5.2f}")
    return m/se


if __name__ == "__main__":
    S = {nm: sessions(sl) for nm, sl in MK.items()}
    idx0 = S["S&P 500"].index
    fomc = pd.DatetimeIndex([pd.Timestamp(x) for x in FOMC])
    hol = pd.DatetimeIndex([pd.Timestamp(x) for x in US_HOLIDAYS])

    def pooled(mask_fn, lab, n_per_yr):
        parts = []
        for nm, s in S.items():
            m = np.asarray(mask_fn(s))
            if m.sum():
                parts.append(s.r.values[m])
        if parts:
            return rep(np.concatenate(parts), lab, ann=n_per_yr)
        return None

    print("="*76)
    print("1. PRE-FOMC DRIFT  (Lucca & Moench, J. Finance 2015)")
    print("="*76)
    for lag, lab in ((1, "day BEFORE FOMC"), (0, "FOMC day itself"),
                     (2, "2 days before"), (-1, "day AFTER FOMC")):
        pooled(lambda s, L=lag: s.index.isin(fomc - pd.Timedelta(days=L)),
               lab, 8)
    print("  baseline (all other days):")
    pooled(lambda s: ~s.index.isin(
        np.concatenate([(fomc - pd.Timedelta(days=k)).values for k in (0, 1, 2)])),
        "  all non-FOMC days", 252)

    print("\n" + "="*76)
    print("2. HOLIDAY EFFECT  (Ariel 1990)")
    print("="*76)
    pooled(lambda s: s.index.isin(hol - pd.Timedelta(days=1)), "day before holiday", 9)
    pooled(lambda s: s.index.isin(hol + pd.Timedelta(days=1)), "day after holiday", 9)

    print("\n" + "="*76)
    print("3. OPTION EXPIRY / QUAD WITCHING (third Friday)")
    print("="*76)
    def is_third_fri(s):
        return (s.index.dayofweek == 4) & (s.index.day >= 15) & (s.index.day <= 21)
    pooled(is_third_fri, "third Friday", 12)
    pooled(lambda s: is_third_fri(s) & s.index.month.isin([3, 6, 9, 12]),
           "quad witching (quarterly)", 4)
    pooled(lambda s: (s.index.dayofweek < 5) & (s.index.day >= 15) &
           (s.index.day <= 21), "expiry week (Mon-Fri)", 60)

    print("\n" + "="*76)
    print("4. TURN OF YEAR")
    print("="*76)
    pooled(lambda s: (s.month == 12) & (s.tdom_rev <= 3), "last 3 days of December", 3)
    pooled(lambda s: (s.month == 1) & (s.tdom <= 5), "first 5 days of January", 5)

    print("\n" + "="*76)
    print("5. DAY-OF-MONTH PROFILE (is turn-of-month the only pocket?)")
    print("="*76)
    for lo, hi, lab in ((1, 3, "trading days 1-3"), (4, 8, "days 4-8"),
                        (9, 13, "days 9-13"), (14, 18, "days 14-18"),
                        (19, 23, "days 19-23")):
        pooled(lambda s, a=lo, b=hi: (s.tdom >= a) & (s.tdom <= b),
               lab, 252*(hi-lo+1)/21)
    pooled(lambda s: s.tdom_rev == 1, "LAST trading day of month", 12)
    pooled(lambda s: s.tdom_rev <= 2, "last 2 trading days", 24)
