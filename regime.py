"""REGIME-CONDITIONAL STRATEGIES.

Every test so far ran across the whole 4.5 years, which averages regimes
together. If mean-reversion pays in ranges and momentum pays in trends, then
testing both across everything gives zero -- which is exactly the result I
keep getting.

There is direct evidence for this in my own work: the mirror test found longs
at -0.285R and shorts at +0.022R, and flipping the chart upside down proved
that asymmetry was the MARKET (a bull run), not the code. So the strategies
demonstrably behave differently depending on what the market is doing.

REGIMES (all computed from data available BEFORE the day being traded):
  TREND   close vs 100-day mean, plus the slope of that mean
          -> UP / DOWN / RANGE
  VOL     20-day realised vol vs its own 1-year median -> HIGH / LOW

STRATEGIES tested in each regime:
  MOMENTUM   breakout of the prior 20 bars, hold for the move
  REVERSION  fade a stretched move back toward the mean

Pre-registered: 3 trend regimes x 2 strategies = 6 cells. Anything that looks
good on DEV must repeat on HOLDOUT, or it is noise.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

COST_BP = 0.5
MK = {"S&P 500": "usa500idxusd", "NASDAQ": "usatechidxusd",
      "DOW": "usa30idxusd", "DAX": "deuidxeur"}


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


def daily_regime(d):
    """Regime label per DAY, using only prior days' data."""
    m = d.index.hour * 60 + d.index.minute
    k = (m >= 570) & (m < 960)
    day = d.index.normalize().tz_localize(None)
    g = d[k].groupby(day[k])
    C = g["close"].last()
    ret = C.pct_change()

    ma = C.rolling(100).mean()
    slope = ma.diff(20) / ma
    dist = (C - ma) / ma
    vol = ret.rolling(20).std()
    volmed = vol.rolling(252).median()

    # shift(1): the label must be known BEFORE the session it applies to
    R = pd.DataFrame({"dist": dist.shift(1), "slope": slope.shift(1),
                      "vol": vol.shift(1), "volmed": volmed.shift(1)})
    trend = pd.Series("RANGE", index=R.index)
    trend[(R.slope > 0.01) & (R.dist > 0)] = "UP"
    trend[(R.slope < -0.01) & (R.dist < 0)] = "DOWN"
    R["trend"] = trend
    R["volregime"] = np.where(R.vol > R.volmed, "HIGH", "LOW")
    return R


def prep(d):
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(),
                    (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().bfill().values
    return o, h, l, c, atr


def s_momentum(o, h, l, c, a, i, look=20):
    if i < look + 1:
        return 0
    hi, lo = h[i-look:i].max(), l[i-look:i].min()
    if c[i] > hi:
        return 1
    if c[i] < lo:
        return -1
    return 0


def s_reversion(o, h, l, c, a, i, look=20):
    """Fade a stretched move: price far from its own recent mean."""
    if i < look + 1:
        return 0
    ma = c[i-look:i].mean()
    z = (c[i] - ma) / max(a[i], 1e-9)
    if z > 1.5:
        return -1
    if z < -1.5:
        return 1
    return 0


STRATS = {"momentum": s_momentum, "reversion": s_reversion}


def run(d, fn, tmult=2.0, sess=(570, 960)):
    o, h, l, c, a = prep(d)
    idx = d.index
    m = idx.hour * 60 + idx.minute
    n = len(d)
    rows = []
    busy = -1
    for i in range(60, n - 2):
        if i < busy or not (sess[0] <= m[i] < sess[1]):
            continue
        s = fn(o, h, l, c, a, i)
        if s == 0:
            continue
        e = c[i]
        stop = (min(l[i-1], l[i]) - 0.1*a[i]) if s > 0 else (max(h[i-1], h[i]) + 0.1*a[i])
        risk = abs(e - stop)
        if risk < 0.25*a[i] or risk > 3*a[i]:
            continue
        tgt = e + s*tmult*risk
        R = None
        for k in range(i+1, min(i+80, n)):
            if (l[k] <= stop) if s > 0 else (h[k] >= stop):
                R = -1.0 - 0.05*a[i]/risk
                busy = k
                break
            if (h[k] >= tgt) if s > 0 else (l[k] <= tgt):
                R = tmult
                busy = k
                break
        if R is None:
            continue
        rows.append(dict(ts=idx[i], R=R - (COST_BP/1e4)*e/risk, side=s))
    return pd.DataFrame(rows)


def st(x, lab):
    x = np.asarray(x, float)
    if len(x) < 40:
        return f"{lab:<22} n={len(x):<5} --"
    m, se = x.mean(), x.std(ddof=1)/np.sqrt(len(x))
    return (f"{lab:<22} n={len(x):<5} {m:+7.3f}  t={m/se:+6.2f}")


if __name__ == "__main__":
    ALL={}
    for nm,slug in MK.items():
        d=load(slug); R=daily_regime(d)
        for sname,fn in STRATS.items():
            t=run(d,fn)
            if not len(t): continue
            t["day"]=pd.to_datetime(t.ts).dt.normalize().dt.tz_localize(None)
            t=t.join(R[["trend","volregime"]],on="day")
            t["mkt"]=nm; t["strat"]=sname
            ALL[(nm,sname)]=t
    T=pd.concat(ALL.values(),ignore_index=True).dropna(subset=["trend"])
    T["ts"]=pd.to_datetime(T.ts)
    T.to_pickle("regime_trades.pkl")
    dev=T[T.ts<"2025-01-01"]; hold=T[T.ts>="2025-01-01"]
    print(f"{len(T):,} trades  |  dev {len(dev):,}  holdout {len(hold):,}\n")

    print("="*78)
    print("TREND REGIME x STRATEGY   (dev 2022-24  ->  HOLDOUT 2025-26)")
    print("="*78)
    print(f"  {'regime':<8}{'strategy':<12}{'DEV expR':>10}{'t':>8}   {'HOLD expR':>11}{'t':>8}")
    print("  "+"-"*58)
    for tr in ("UP","DOWN","RANGE"):
        for s in STRATS:
            a=dev[(dev.trend==tr)&(dev.strat==s)].R.values
            b=hold[(hold.trend==tr)&(hold.strat==s)].R.values
            if len(a)<40 or len(b)<40:
                print(f"  {tr:<8}{s:<12}  (too few)"); continue
            ma,sa=a.mean(),a.std(ddof=1)/np.sqrt(len(a))
            mb,sb=b.mean(),b.std(ddof=1)/np.sqrt(len(b))
            flag=" <<<" if (ma>0 and mb>0) else ""
            print(f"  {tr:<8}{s:<12}{ma:>10.3f}{ma/sa:>8.2f}   {mb:>11.3f}{mb/sb:>8.2f}{flag}")
        print()

    print("="*78)
    print("ADDING VOLATILITY REGIME")
    print("="*78)
    print(f"  {'trend':<7}{'vol':<6}{'strategy':<12}{'DEV':>9}{'t':>7}   {'HOLD':>9}{'t':>7}")
    print("  "+"-"*54)
    for tr in ("UP","DOWN","RANGE"):
        for vr in ("HIGH","LOW"):
            for s in STRATS:
                a=dev[(dev.trend==tr)&(dev.volregime==vr)&(dev.strat==s)].R.values
                b=hold[(hold.trend==tr)&(hold.volregime==vr)&(hold.strat==s)].R.values
                if len(a)<40 or len(b)<40: continue
                ma,sa=a.mean(),a.std(ddof=1)/np.sqrt(len(a))
                mb,sb=b.mean(),b.std(ddof=1)/np.sqrt(len(b))
                flag=" <<<" if (ma>0 and mb>0) else ""
                print(f"  {tr:<7}{vr:<6}{s:<12}{ma:>9.3f}{ma/sa:>7.2f}   {mb:>9.3f}{mb/sb:>7.2f}{flag}")
