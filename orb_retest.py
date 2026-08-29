"""OPENING RANGE BREAKOUT + RETEST -- Scarface Trades' method, coded to spec.

Transcribed locally from the video file (YouTube API is IP-blocked), so these
are his exact stated rules:

  1. Mark the HIGH and LOW of the first 5-minute candle (09:30-09:35 ET).
  2. Wait for a 1-minute candle to CLOSE beyond that range.
  3. Do NOT enter the breakout. Wait for the RETEST of the broken level --
     "if we took these first two examples with just a breakout, we would have
     lost. That's why the retest opportunity is key."
  4. Stop: a break back INSIDE the range ("back below the five minute high").
  5. Target: 2R.
  6. Cutoff 11:00 ET. No retest by then = no trade, and he explicitly shows a
     day with no entry rather than forcing one.
  7. One trade per day.

Worth testing seriously for two reasons: it is the most mechanical of the four
strategies I have been sent, and opening-range breakout has genuine published
support (Zarattini & Aziz 2023 on SPY) unlike the candlestick material.

The retest requirement is the interesting part. It is testable directly:
breakout-only vs breakout-plus-retest, same data.

Costs 0.5bp round trip (real index-futures cost).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

COST_BP = 0.5
OPEN = 9*60 + 30
CUTOFF = 11*60
MK = {"S&P 500": "usa500idxusd", "NASDAQ": "usatechidxusd",
      "DOW": "usa30idxusd", "DAX": "deuidxeur"}


def load(slug, tf="m1"):
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


def backtest(d1, require_retest=True, tmult=2.0, cutoff=CUTOFF, stop_mode="level"):
    o, h, l, c = (d1[x].values for x in ("open", "high", "low", "close"))
    idx = d1.index
    m = idx.hour*60 + idx.minute
    day = idx.normalize()
    n = len(d1)

    rows = []
    # group by session
    starts = np.where(np.diff(np.concatenate([[0], day.astype("int64")])) != 0)[0]
    for s0 in starts:
        # locate the opening 5 minutes of this day
        d0 = day[s0]
        sel = np.where((day == d0) & (m >= OPEN) & (m < cutoff))[0]
        if len(sel) < 40:
            continue
        orb = sel[(m[sel] >= OPEN) & (m[sel] < OPEN + 5)]
        if len(orb) < 3:
            continue
        hi, lo = h[orb].max(), l[orb].min()
        rng = hi - lo
        if rng <= 0:
            continue

        rest = sel[m[sel] >= OPEN + 5]
        broke = None          # (+1 up / -1 down, index of breakout bar)
        entry_i = None
        for i in rest:
            if broke is None:
                if c[i] > hi:
                    broke = (1, i)
                elif c[i] < lo:
                    broke = (-1, i)
                if broke and not require_retest:
                    entry_i = i
                    break
                continue
            sgn, bi = broke
            # RETEST: price comes back to the broken level, then resumes
            if sgn > 0:
                if l[i] <= hi and c[i] > hi:
                    entry_i = i
                    break
                if c[i] < lo:            # broke the other way -> reset
                    broke = (-1, i)
            else:
                if h[i] >= lo and c[i] < lo:
                    entry_i = i
                    break
                if c[i] > hi:
                    broke = (1, i)
        if entry_i is None or broke is None:
            continue

        sgn = broke[0]
        e = c[entry_i]
        # HIS RULE: "a break back below the five minute high" -- the stop is
        # just beyond the BROKEN LEVEL, not the far side of the range. I had
        # this wrong: using the far side made risk 3-5x too wide, which moves
        # the 2R target much further away and changes every result.
        pad = 0.05 * rng
        if stop_mode == "level":
            stop = (hi - pad) if sgn > 0 else (lo + pad)
        elif stop_mode == "impulse":            # "below the impulsive candle"
            bi = broke[1]
            stop = min(l[bi:entry_i+1]) if sgn > 0 else max(h[bi:entry_i+1])
        else:                                    # far side of the range
            stop = lo if sgn > 0 else hi
        risk = abs(e - stop)
        if risk <= 0:
            continue
        tgt = e + sgn*tmult*risk
        R = None
        for k in range(entry_i+1, n):
            if day[k] != d0:
                R = (c[k-1]-e)*sgn/risk
                break
            if (l[k] <= stop) if sgn > 0 else (h[k] >= stop):
                R = -1.0
                break
            if (h[k] >= tgt) if sgn > 0 else (l[k] <= tgt):
                R = tmult
                break
        if R is None:
            continue
        rows.append(dict(ts=idx[entry_i], R=R - (COST_BP/1e4)*e/risk,
                         side="long" if sgn > 0 else "short",
                         risk=risk, rng=rng))
    return pd.DataFrame(rows)


def st(x, lab):
    x = np.asarray(x, float)
    if len(x) < 25:
        print(f"  {lab:<34} n={len(x)}  (too few)")
        return
    m, se = x.mean(), x.std(ddof=1)/np.sqrt(len(x))
    print(f"  {lab:<34} n={len(x):<5} win {(x>0).mean()*100:5.1f}%  "
          f"expR {m:+.3f}  t={m/se:+6.2f}")


if __name__ == "__main__":
    for nm, slug in MK.items():
        d1 = load(slug, "m1")
        print(f"\n{'='*72}\n{nm}   ({len(d1):,} 1-min bars)\n{'='*72}")
        a = backtest(d1, require_retest=True)
        b = backtest(d1, require_retest=False)
        st(b.R, "breakout only (no retest)")
        st(a.R, "HIS METHOD (breakout + retest)")
        if len(a) > 50:
            a["ts"] = pd.to_datetime(a.ts)
            st(a[a.ts < "2025-01-01"].R, "   dev 2022-2024")
            st(a[a.ts >= "2025-01-01"].R, "   HOLDOUT 2025-2026")
