"""THE TWO UNTESTED STRATEGIES FROM THE ARTICLE.

Already measured in this project, so not repeated here:
  breakout          pat_break, tested across volume regimes -- fair odds
  scalping          0.44bp edge against a 1bp spread -- loses to costs
  momentum intraday 0.77bp against 2bp costs -- loses to costs
  reversal/mean-rev s_reversion across trend regimes -- fair odds
  position sizing   multiplicative on edge; cannot change the sign of a mean
  stop-losses       measured; on concentrated momentum they COST ~3pp/yr

These two are specified precisely enough to test exactly as written, and
neither has been run:

  1. PIVOT POINTS
       P  = (High + Low + Close) / 3          <- prior session
       R1 = 2P - Low        S1 = 2P - High
       R2 = P + (R1 - S1)   S2 = P - (R1 - S1)
     Traded both ways, since the article describes both uses: range traders
     fade S1/R1, breakout traders trade the break of them.

  2. MOVING AVERAGE CROSSOVER, 20/60/100
       long when MA20 crosses above MA60, exit on the reverse cross
       trend filter: price must sit above the MA100 for longs

Same bar as everything else: DEV/HOLDOUT split, real costs, at least 3 of 4
markets positive, and a t-stat that clears the multiple-testing bar.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

COST_BP = 0.5
MK = {"S&P 500": "usa500idxusd", "NASDAQ": "usatechidxusd",
      "DOW": "usa30idxusd", "DAX": "deuidxeur"}
OPEN, CLOSE = 9*60+30, 16*60


def load(slug):
    fs = [f for f in glob.glob(f"download/{slug}-m5-bid-*.csv") if "2026-07-24" in f]
    if not fs:
        return None
    d = pd.read_csv(max(fs, key=os.path.getsize))
    d.columns = [c.lower() for c in d.columns]
    ts = d["timestamp"]
    idx = (pd.to_datetime(ts, unit="ms", utc=True)
           if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts, utc=True))
    d.index = idx.dt.tz_convert("America/New_York")
    d = d[["open", "high", "low", "close"]].astype(float).sort_index()
    assert d.index[0].year > 2000
    return d


def sessions(d):
    m = d.index.hour*60 + d.index.minute
    k = (m >= OPEN) & (m < CLOSE)
    dd = d[k]
    day = dd.index.normalize().tz_localize(None)
    g = dd.groupby(day)
    return pd.DataFrame({"o": g["open"].first(), "h": g["high"].max(),
                         "l": g["low"].min(), "c": g["close"].last()})


def pivots(S):
    """Prior session's pivot levels, exactly as the article specifies."""
    P = (S.h + S.l + S.c)/3
    R1 = 2*P - S.l
    S1 = 2*P - S.h
    R2 = P + (R1 - S1)
    S2 = P - (R1 - S1)
    out = pd.DataFrame({"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2})
    return out.shift(1)          # prior day's levels apply to today


def test_pivot(d, S, mode):
    """mode 'fade': buy S1 / sell R1.  mode 'break': trade the break."""
    PV = pivots(S)
    m = d.index.hour*60 + d.index.minute
    day = d.index.normalize().tz_localize(None)
    rows = []
    for dt, lv in PV.dropna().iterrows():
        seg = d[(day == dt) & (m >= OPEN) & (m < CLOSE)]
        if len(seg) < 20:
            continue
        c = seg["close"].values
        entry = side = None
        for i in range(1, len(c)-1):
            if entry is None:
                if mode == "fade":
                    if c[i] <= lv.S1:
                        entry, side = c[i], 1
                    elif c[i] >= lv.R1:
                        entry, side = c[i], -1
                else:
                    if c[i-1] <= lv.R1 < c[i]:
                        entry, side = c[i], 1
                    elif c[i-1] >= lv.S1 > c[i]:
                        entry, side = c[i], -1
                if entry is not None:
                    tgt = lv.P if mode == "fade" else (
                        lv.R2 if side > 0 else lv.S2)
                    stop = (lv.S2 if side > 0 else lv.R2) if mode == "fade" else lv.P
                    j0 = i
            else:
                hit_t = (c[i] >= tgt) if side > 0 else (c[i] <= tgt)
                hit_s = (c[i] <= stop) if side > 0 else (c[i] >= stop)
                if hit_t or hit_s or i == len(c)-2:
                    r = side*(c[i]/entry - 1)*1e4 - COST_BP
                    rows.append(dict(ts=dt, r=r))
                    entry = None
    return pd.DataFrame(rows)


def test_ma(d, S):
    """20/60/100 crossover on session closes, with the MA100 trend filter."""
    c = S.c
    f, s, t = c.rolling(20).mean(), c.rolling(60).mean(), c.rolling(100).mean()
    long = (f > s) & (c > t)
    pos = long.astype(float).shift(1).fillna(0)
    ret = c.pct_change()*1e4
    trades = pos.diff().abs().fillna(0)
    return (pos*ret - trades*COST_BP).dropna()


def report(name, per):
    P = pd.DataFrame(per)
    pooled = P.mean(axis=1).dropna()
    dev = pooled[pooled.index < "2025-01-01"]
    hold = pooled[pooled.index >= "2025-01-01"]

    def m(x):
        x = x[x != 0]
        if len(x) < 40:
            return None
        return dict(n=len(x), mean=x.mean(),
                    t=x.mean()/(x.std(ddof=1)/np.sqrt(len(x))))
    a, b = m(dev), m(hold)
    if not a or not b:
        print(f"  {name:<26} (too few trades)")
        return
    npos = sum(1 for col in P.columns
               if len(P[col][P[col] != 0]) > 40 and P[col][P[col] != 0].mean() > 0)
    ok = a["mean"] > 0 and b["mean"] > 0 and npos >= 3
    print(f"  {name:<26}{a['mean']:>9.2f}{a['t']:>7.2f}{b['mean']:>9.2f}"
          f"{b['t']:>7.2f}{npos:>4}/4{'  SURVIVES' if ok else '':>12}")


if __name__ == "__main__":
    D = {}
    for nm, slug in MK.items():
        d = load(slug)
        if d is not None:
            D[nm] = (d, sessions(d))
    print(f"{len(D)} markets loaded\n")
    print(f"{'strategy':<26}{'DEV bp':>9}{'t':>7}{'HOLD bp':>9}{'t':>7}"
          f"{'mkts+':>7}{'verdict':>12}")
    print("-" * 78)

    for mode, lab in (("fade", "pivot: fade S1/R1"),
                      ("break", "pivot: break S1/R1")):
        per = {}
        for nm, (d, S) in D.items():
            t = test_pivot(d, S, mode)
            if len(t):
                per[nm] = t.groupby("ts").r.mean()
        if per:
            report(lab, per)

    per = {}
    for nm, (d, S) in D.items():
        per[nm] = test_ma(d, S)
    report("MA cross 20/60/100", per)

    print("\n" + "=" * 78)
    print("ALREADY TESTED IN THIS PROJECT")
    print("=" * 78)
    for lab, res in (("breakout (all vol regimes)", "fair odds"),
                     ("scalping", "0.44bp edge vs 1bp spread -- loses"),
                     ("intraday momentum", "0.77bp vs 2bp costs -- loses"),
                     ("mean reversion", "fair odds across all trend regimes"),
                     ("position sizing", "multiplies edge, cannot create it"),
                     ("stop losses", "cost ~3pp/yr on concentrated momentum"),
                     ("full TJR/ICT method", "48.4% win, -0.100R, t=-9.60")):
        print(f"  {lab:<32}{res}")
