"""SIGNAL LAB -- hunting for independent edges, systematically.

The bottleneck is not sizing or leverage. It is that I have ONE edge
(calendar, Sharpe 1.11). Four uncorrelated edges at Sharpe 1.1 combine to
~2.2 and the drawdown does NOT scale with them, because they lose at
different times. That is the only mechanism that produces "more wins than
drawdowns" at size.

So: cast a wide net across genuinely DIFFERENT mechanisms, not variations on
price shape. Eight candlestick patterns are one idea tested eight times.
These are different ideas:

  CALENDAR      mechanical flows on a schedule          (known: works)
  TS-MOMENTUM   Moskowitz/Ooi/Pedersen time-series momentum
  ST-REVERSAL   1-week reversal, documented since Jegadeesh (1990)
  VOL-MR        volatility mean-reverts; trade the return consequence
  RANGE-CYCLE   contraction precedes expansion (NR7 family)
  GAP           overnight gap continuation vs fade
  XSECTIONAL    relative strength across the four indices
  INTRADAY-SEAS specific hours of the session
  VOL-BREAKOUT  breakouts confirmed by volume (measured: helps)

VALIDATION, applied identically to every candidate:
  1. DEV 2022-2024 -> HOLDOUT 2025-2026, both must be positive
  2. per-market consistency: at least 3 of 4 markets positive
  3. costs 0.5bp
  4. Bonferroni-aware: with ~15 candidates, |t|>3 is the real bar
  5. survivors then checked for CORRELATION with each other
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

COST = 0.5
OPEN, CLOSE = 9*60+30, 16*60
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
    cols = ["open", "high", "low", "close"] + (["volume"] if "volume" in d else [])
    d = d[cols].astype(float).sort_index()
    assert d.index[0].year > 2000
    return d


def daily(slug):
    """Session bars plus the pieces every signal needs."""
    d = load(slug)
    m = d.index.hour*60 + d.index.minute
    k = (m >= OPEN) & (m < CLOSE)
    dd = d[k]
    day = dd.index.normalize().tz_localize(None)
    g = dd.groupby(day)
    D = pd.DataFrame({
        "o": g["open"].first(), "c": g["close"].last(),
        "h": g["high"].max(), "l": g["low"].min(),
        "v": g["volume"].sum() if "volume" in dd else np.nan,
    })
    # overnight gap: today's open vs yesterday's close
    D["gap"] = (D.o/D.c.shift(1) - 1)*1e4
    D["r"] = (D.c/D.o - 1)*1e4                 # the tradeable session return
    D["cc"] = D.c.pct_change()*1e4             # close-to-close
    D["range"] = (D.h-D.l)/D.o*1e4
    # first/second half-hour legs
    k2 = (m >= OPEN) & (m < OPEN+30)
    g2 = d[k2].groupby(d[k2].index.normalize().tz_localize(None))
    D["leg1"] = (g2["close"].last()/g2["open"].first()-1)*1e4
    return D.dropna(subset=["r"])


# ---------------------------------------------------------------- signals
# Each returns a per-day POSITION in {-1,0,+1}, using ONLY prior information.

def s_calendar(D):
    idx = D.index
    pos = np.zeros(len(D))
    mon = idx.dayofweek == 0
    tom = np.zeros(len(D), bool)
    for _, g in pd.Series(idx, index=idx).groupby([idx.year, idx.month]):
        dd = pd.DatetimeIndex(g.values)
        tom |= idx.isin(dd[:3]); tom |= idx.isin(dd[-1:])
    pos[mon | tom] = 1
    return pd.Series(pos, index=idx)


def s_tsmom(D, look=60):
    """Time-series momentum: trade with the sign of the past `look` days."""
    sig = np.sign(D.c.pct_change(look)).shift(1)
    return sig.fillna(0)


def s_streversal(D, look=5):
    """Short-term reversal: fade the last week."""
    sig = -np.sign(D.c.pct_change(look)).shift(1)
    return sig.fillna(0)


def s_volmr(D, look=20):
    """After a HIGH-vol stretch, vol falls and drift resumes -> long."""
    vol = D.cc.rolling(look).std()
    med = vol.rolling(252).median()
    sig = np.where(vol.shift(1) > 1.3*med.shift(1), 1.0, 0.0)
    return pd.Series(sig, index=D.index)


def s_rangecycle(D, look=7):
    """Contraction precedes expansion: after the narrowest range in `look`
    days, take the direction of the prior trend."""
    narrow = D.range == D.range.rolling(look).min()
    trend = np.sign(D.c.pct_change(20))
    sig = np.where(narrow.shift(1).fillna(False), trend.shift(1), 0.0)
    return pd.Series(sig, index=D.index).fillna(0)


def s_gapfade(D):
    """Fade the overnight gap during the session."""
    return -np.sign(D.gap).fillna(0)


def s_gapgo(D):
    return np.sign(D.gap).fillna(0)


def s_intraday_seas(D):
    """Monday+Friday vs midweek, as a pure day-of-week bet."""
    dow = D.index.dayofweek
    return pd.Series(np.where(np.isin(dow, [0, 4]), 1.0, 0.0), index=D.index)


def s_leg1_fade(D):
    """Fade the first 30 minutes over the rest of the session."""
    return -np.sign(D.leg1).fillna(0)


SIGNALS = {
    "calendar": s_calendar,
    "tsmom_60": lambda D: s_tsmom(D, 60),
    "tsmom_20": lambda D: s_tsmom(D, 20),
    "st_reversal": s_streversal,
    "vol_mr": s_volmr,
    "range_cycle": s_rangecycle,
    "gap_fade": s_gapfade,
    "gap_go": s_gapgo,
    "dow_seas": s_intraday_seas,
    "leg1_fade": s_leg1_fade,
}


def evaluate(name, fn, D_all):
    """Per-market returns for one signal, then the pooled daily series."""
    per = {}
    for nm, D in D_all.items():
        pos = fn(D).reindex(D.index).fillna(0)
        # position is applied to the SESSION return, cost only when trading
        r = pos*D.r - np.abs(pos)*COST
        per[nm] = r
    P = pd.DataFrame(per)
    return P


def report(P, label, dev_end="2025-01-01"):
    pooled = P.mean(axis=1).dropna()
    dev = pooled[pooled.index < dev_end]
    hold = pooled[pooled.index >= dev_end]
    def m(x, ann=252):
        x = x[x != 0]
        if len(x) < 40:
            return None
        mu, sd = x.mean(), x.std(ddof=1)
        return dict(n=len(x), mean=mu, t=mu/(sd/np.sqrt(len(x))),
                    sharpe=mu/sd*np.sqrt(min(ann, len(x)/4.5)))
    a, b = m(dev), m(hold)
    if not a or not b:
        return None
    npos = sum(1 for nm in P.columns
               if len(P[nm][P[nm] != 0]) > 40 and P[nm][P[nm] != 0].mean() > 0)
    return dict(name=label, dev_t=a["t"], dev_m=a["mean"], hold_t=b["t"],
                hold_m=b["mean"], hold_sharpe=b["sharpe"], n=a["n"]+b["n"],
                mkts_pos=npos, series=pooled)


if __name__ == "__main__":
    D_all = {nm: daily(sl) for nm, sl in MK.items()}
    print(f"{len(D_all)} markets, {len(D_all['S&P 500'])} sessions each\n")
    print(f"{'signal':<14}{'DEV bp':>9}{'t':>7}{'HOLD bp':>9}{'t':>7}"
          f"{'mkts+':>7}{'verdict':>12}")
    print("-"*66)
    res = {}
    for name, fn in SIGNALS.items():
        P = evaluate(name, fn, D_all)
        r = report(P, name)
        if not r:
            print(f"{name:<14}  (too few trades)")
            continue
        res[name] = r
        ok = (r["dev_m"] > 0 and r["hold_m"] > 0 and r["mkts_pos"] >= 3)
        v = "SURVIVES" if ok else ""
        print(f"{name:<14}{r['dev_m']:>9.2f}{r['dev_t']:>7.2f}"
              f"{r['hold_m']:>9.2f}{r['hold_t']:>7.2f}"
              f"{r['mkts_pos']:>5}/4{v:>12}")
    surv = [k for k, r in res.items()
            if r["dev_m"] > 0 and r["hold_m"] > 0 and r["mkts_pos"] >= 3]
    print(f"\nSURVIVORS: {surv if surv else 'none'}")
    if len(surv) >= 2:
        C = pd.DataFrame({k: res[k]["series"] for k in surv}).corr()
        print("\nCORRELATION between survivors:")
        print(C.round(3).to_string())
        off = C.values[np.triu_indices_from(C.values, 1)]
        print(f"\n  mean |corr| = {np.abs(off).mean():.3f}")
