"""Riley Coleman's ACTUAL five-step method, from 339,284 words of transcripts.

Not the candlestick PDF -- the method he describes doing every morning:

  1. 15-MINUTE chart of ES/NQ. Draw support/resistance off key swings,
     BEFORE the open.
  2. Wait for price to reach one of those levels. "The market has to come
     to a level." No level, no trade.
  3. TIMING WINDOW. He marks 15 and 30 minutes after the open -- "those
     reversal timings I look for" -- i.e. roughly 09:45 and 10:00 ET.
  4. Drop to the 1-MINUTE chart.
  5. STRUCTURE CONFIRMATION, and this is the part he stresses most: he will
     not buy the touch. He waits for a pullback that makes a HIGHER LOW
     (mirror for shorts) -- "the structure of the market showing you that
     it's going to actually reverse instead of me trying to predict".

  REVERSALS ONLY. "I don't trade continuation patterns."

Two things here that my earlier test of his patterns did NOT have: the timing
window, and the 1-minute higher-low confirmation. Both are central to how he
describes trading, so this is the fair test of his method.

Costs: 0.5bp round trip (real index-futures cost).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

COST_BP = 0.5
OPEN = 9 * 60 + 30
MK = {"S&P 500": "usa500idxusd", "NASDAQ": "usatechidxusd"}


def load(slug, tf):
    fs = [f for f in glob.glob(f"download/{slug}-{tf}-bid-*.csv") if "2026-07-24" in f]
    d = pd.read_csv(max(fs, key=os.path.getsize))
    d.columns = [c.lower() for c in d.columns]
    ts = d["timestamp"]
    idx = (pd.to_datetime(ts, unit="ms", utc=True)
           if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts, utc=True))
    d.index = idx.dt.tz_convert("America/New_York")
    d = d[["open", "high", "low", "close"]].astype(float).sort_index()
    assert d.index[0].year > 2000
    return d


def zones_15m(d5):
    """Step 1: S/R from 15-minute swings, shifted so only CLOSED swings count."""
    d15 = d5.resample("15min").agg({"open": "first", "high": "max",
                                    "low": "min", "close": "last"}).dropna()
    w = 5
    sh = (d15["high"].rolling(w, center=True).max() == d15["high"])
    sl = (d15["low"].rolling(w, center=True).min() == d15["low"])
    hi = pd.Series(np.where(sh, d15["high"], np.nan), index=d15.index).ffill().shift(3)
    lo = pd.Series(np.where(sl, d15["low"], np.nan), index=d15.index).ffill().shift(3)
    return hi, lo


def run(name, slug, win=(15, 45), need_hl=True, tmult=2.0, max_hold=120):
    d5 = load(slug, "m5")
    d1 = load(slug, "m1")
    hi15, lo15 = zones_15m(d5)
    H = hi15.reindex(d1.index, method="ffill").values
    L = lo15.reindex(d1.index, method="ffill").values

    o, h, l, c = (d1[x].values for x in ("open", "high", "low", "close"))
    tr = pd.concat([d1["high"] - d1["low"],
                    (d1["high"] - d1["close"].shift()).abs(),
                    (d1["low"] - d1["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(30).mean().bfill().values
    idx = d1.index
    m = idx.hour * 60 + idx.minute
    n = len(d1)

    rows = []
    busy = -1
    last_day = None
    for i in range(60, n - 2):
        mins_after = m[i] - OPEN
        if not (win[0] <= mins_after <= win[1]):     # step 3: timing window
            continue
        if i < busy:
            continue
        day = idx[i].date()
        if day == last_day:
            continue                                 # one trade per morning
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue

        # step 2: has price REACHED a 15m level in the last 20 minutes?
        seg = slice(max(i - 20, 0), i + 1)
        touched = None
        if np.isfinite(L[i]) and np.min(l[seg]) <= L[i] and c[i] > L[i]:
            touched = 1                              # at support -> look long
        elif np.isfinite(H[i]) and np.max(h[seg]) >= H[i] and c[i] < H[i]:
            touched = -1                             # at resistance -> look short
        if touched is None:
            continue

        # step 5: structure confirmation on the 1-minute -- a HIGHER LOW
        # (for longs) after the extreme. He refuses to buy the touch itself.
        if need_hl:
            lo_i = int(np.argmin(l[seg])) + seg.start if touched > 0 else None
            hi_i = int(np.argmax(h[seg])) + seg.start if touched < 0 else None
            if touched > 0:
                after = slice(lo_i + 1, i + 1)
                if after.stop - after.start < 3:
                    continue
                # must have pushed up, pulled back, and made a higher low
                pk = int(np.argmax(h[after])) + after.start
                if pk >= i - 1:
                    continue
                if np.min(l[pk:i + 1]) <= l[lo_i]:
                    continue                          # not a higher low
                if c[i] <= c[pk]:
                    continue                          # not resuming up
            else:
                after = slice(hi_i + 1, i + 1)
                if after.stop - after.start < 3:
                    continue
                tr_i = int(np.argmin(l[after])) + after.start
                if tr_i >= i - 1:
                    continue
                if np.max(h[tr_i:i + 1]) >= h[hi_i]:
                    continue                          # not a lower high
                if c[i] >= c[tr_i]:
                    continue

        sgn = touched
        e = c[i]
        stop = (np.min(l[seg]) - 0.1 * a) if sgn > 0 else (np.max(h[seg]) + 0.1 * a)
        risk = abs(e - stop)
        if risk < 0.5 * a or risk > 8 * a:
            continue
        tgt = e + sgn * tmult * risk
        R = None
        for k in range(i + 1, min(i + max_hold, n)):
            if idx[k].date() != day:
                R = (c[k - 1] - e) * sgn / risk
                busy = k
                break
            if (l[k] <= stop) if sgn > 0 else (h[k] >= stop):
                R = -1.0 - 0.05 * a / risk
                busy = k
                break
            if (h[k] >= tgt) if sgn > 0 else (l[k] <= tgt):
                R = tmult
                busy = k
                break
        if R is None:
            continue
        rows.append(dict(ts=idx[i], R=R - (COST_BP / 1e4) * e / risk,
                         side="long" if sgn > 0 else "short"))
        last_day = day
    return pd.DataFrame(rows)


def st(x, lab):
    x = np.asarray(x, float)
    if len(x) < 25:
        print(f"  {lab:<34} n={len(x)}  (too few)")
        return
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    print(f"  {lab:<34} n={len(x):<5} win {(x>0).mean()*100:5.1f}%  "
          f"expR {m:+.3f}  t={m/se:+6.2f}")


if __name__ == "__main__":
    for nm, slug in MK.items():
        print(f"\n{'='*72}\n{nm}\n{'='*72}")
        for hl, win, lab in ((True, (15, 45), "FULL method (timing + higher-low)"),
                             (False, (15, 45), "  without structure confirmation"),
                             (True, (0, 390), "  without timing window")):
            t = run(nm, slug, win=win, need_hl=hl)
            st(t.R, lab)
            if hl and win == (15, 45) and len(t) > 40:
                t["ts"] = pd.to_datetime(t.ts)
                st(t[t.ts < "2025-01-01"].R, "     dev 2022-2024")
                st(t[t.ts >= "2025-01-01"].R, "     HOLDOUT 2025-2026")
