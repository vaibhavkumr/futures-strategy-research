"""PRE-REGISTERED tests of PUBLISHED effects -- not data mining.

Everything else in this project searched for a pattern and then tested it, so
every result had to fight the suspicion that the search itself created it.
These hypotheses were written down by other people, on other data, before I
touched anything. My 2022-2026 sample is genuinely out-of-sample for them.

  1. MARKET INTRADAY MOMENTUM (Gao, Han, Li & Zhou, JFE 2018)
     The first half-hour return predicts the last half-hour return.
     Published on SPY 1993-2013. Direction is fixed in advance: SAME sign.

  2. INTRADAY MOMENTUM + 12th half-hour (their enhanced version)
     Adding the 15:00-15:30 return raised their R^2 from 1.6% to 2.6%.

  3. OVERNIGHT vs INTRADAY DECOMPOSITION
     Robustly documented: the equity premium accrues OVERNIGHT (close->open),
     while open->close is flat or negative. Fixed prediction: overnight > 0.

No thresholds are tuned. No subsets are searched. Each hypothesis has ONE
pre-committed direction, and it either shows up on four markets or it does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from duka import load

MK = {"NASDAQ": "usatechidxusd", "S&P 500": "usa500idxusd",
      "DOW": "usa30idxusd", "DAX": "deuidxeur"}

# US regular trading hours, ET, in minutes from midnight
OPEN, CLOSE = 9 * 60 + 30, 16 * 60


def sessions(slug: str) -> pd.DataFrame:
    """One row per trading day with the returns the papers use."""
    d = load(slug, "m5")
    mins = d.index.hour * 60 + d.index.minute
    d = d[(mins >= OPEN) & (mins < CLOSE)].copy()
    mins = d.index.hour * 60 + d.index.minute
    day = d.index.normalize()

    def seg(a, b):
        m = (mins >= a) & (mins < b)
        g = d[m].groupby(day[m])
        return g["open"].first(), g["close"].last()

    o1, c1 = seg(OPEN, OPEN + 30)            # 09:30-10:00  first half hour
    o12, c12 = seg(CLOSE - 60, CLOSE - 30)   # 15:00-15:30  twelfth
    o13, c13 = seg(CLOSE - 30, CLOSE)        # 15:30-16:00  last

    dayo = d.groupby(day)["open"].first()
    dayc = d.groupby(day)["close"].last()

    out = pd.DataFrame({
        "r_first": c1 / o1 - 1,
        "r_12th": c12 / o12 - 1,
        "r_last": c13 / o13 - 1,
        "open": dayo, "close": dayc,
        "day_range": (d.groupby(day)["high"].max()
                      - d.groupby(day)["low"].min()) / dayo,
    }).dropna()
    out["r_overnight"] = out["open"] / out["close"].shift(1) - 1
    out["r_intraday"] = out["close"] / out["open"] - 1
    return out.dropna()


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan, len(x)
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x))), len(x)


def report(x, label, ann=252, cost_bp=0.0):
    """cost_bp = round-trip cost in basis points, subtracted from every trade."""
    x = np.asarray(x, float) - cost_bp / 1e4
    m, t, n = tstat(x)
    if not np.isfinite(m):
        print(f"  {label:<34} n={n}")
        return
    sharpe = m / np.std(x, ddof=1) * np.sqrt(ann)
    print(f"  {label:<34} n={n:<5} mean {m*1e4:+7.2f}bp  t={t:+6.2f}  "
          f"win {(x>0).mean()*100:5.1f}%  Sharpe {sharpe:+5.2f}  "
          f"ann {m*ann*100:+6.1f}%")


if __name__ == "__main__":
    S = {name: sessions(slug) for name, slug in MK.items()}
    for name, s in S.items():
        print(f"{name:<9} {len(s):>5} trading days  "
              f"{s.index.min():%Y-%m-%d} -> {s.index.max():%Y-%m-%d}")

    print("\n" + "=" * 78)
    print("H1  INTRADAY MOMENTUM -- trade the last half hour in the SAME")
    print("    direction as the first half hour  (Gao/Han/Li/Zhou, JFE 2018)")
    print("=" * 78)
    for name, s in S.items():
        sig = np.sign(s.r_first)
        report(sig * s.r_last, name)
    pooled = np.concatenate([np.sign(s.r_first) * s.r_last for s in S.values()])
    print()
    report(pooled, "POOLED (4 markets)")
    print()
    report(pooled, "POOLED after 2bp round-trip cost", cost_bp=2.0)

    print("\n  regression check -- does r_first actually predict r_last?")
    for name, s in S.items():
        b = np.polyfit(s.r_first, s.r_last, 1)[0]
        r = np.corrcoef(s.r_first, s.r_last)[0, 1]
        tt = r * np.sqrt(len(s) - 2) / np.sqrt(max(1 - r ** 2, 1e-12))
        print(f"    {name:<9} slope {b:+7.3f}  corr {r:+.4f}  t={tt:+5.2f}  "
              f"R2 {r**2*100:5.2f}%   (paper: slope +6.94, R2 1.6%)")

    print("\n" + "=" * 78)
    print("H2  ENHANCED -- sign of (first half hour + 12th half hour)")
    print("=" * 78)
    for name, s in S.items():
        report(np.sign(s.r_first + s.r_12th) * s.r_last, name)
    print()
    report(np.concatenate([np.sign(s.r_first + s.r_12th) * s.r_last
                           for s in S.values()]), "POOLED (4 markets)")

    print("\n" + "=" * 78)
    print("H3  OVERNIGHT vs INTRADAY -- where does the return actually accrue?")
    print("=" * 78)
    for name, s in S.items():
        print(f"  --- {name} ---")
        report(s.r_overnight, "    overnight (close->open)")
        report(s.r_intraday, "    intraday  (open->close)")
        report(s.r_overnight + s.r_intraday, "    buy & hold")
    print()
    report(np.concatenate([s.r_overnight for s in S.values()]),
           "POOLED overnight")
    report(np.concatenate([s.r_intraday for s in S.values()]),
           "POOLED intraday")
