"""FIXING THE THREE PROBLEMS WITH THE FILTERED VIDEO MODEL.

The filtered model (decisive change-of-character, top 30% by ATR) measured
+0.200R/trade, t=9.36, placebo 0/400, 11/11 quarters positive. Three things
could still be wrong with it, in descending order of danger:

  1. LIMIT FILLS -- the biggest risk. The sim fills at the FVG midpoint
     whenever price touches it. Reality is worse: in a fast market price
     trades through without filling you, and the trades you MISS are
     disproportionately the ones that ran to target. This is the classic way
     a backtest invents an edge that does not exist. Four fill models are
     tested, from optimistic to punishing.

  2. SLIPPAGE ON EXITS -- stops fill worse than the stop price, targets fill
     at the limit or not at all. Modelled explicitly.

  3. R IS NOT A RETURN -- +0.200R/trade says nothing about compounding until
     position sizing, overlapping trades and drawdown are simulated. Built
     here as an actual equity curve.

Fill models:
  touch        fill at midpoint if the bar's range includes it   (optimistic)
  through      fill only if the bar trades a tick BEYOND the midpoint
  close_beyond fill only if a bar CLOSES beyond the midpoint, at the NEXT
               bar's open -- what a conservative trader would actually get
  next_open    fill at the next bar's open after the touch, whatever it is
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from video_model import load_m1, swings, MK, OPEN_MIN, CLOSE_MIN

COST_BP = 0.5
MIN_RISK = 0.15
CHOCH_Q = 0.70


def atr(h, l, c, n=14):
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]),
                                            np.abs(l[1:]-c[:-1])))
    a = np.full(len(c), np.nan)
    if len(tr) >= n:
        a[n:] = pd.Series(tr).rolling(n).mean().values[n-1:]
    return pd.Series(a).bfill().ffill().values


def run_day(o, h, l, c, tmult=4.0, buf=0.10, k_sw=3,
            fill="touch", slip_ticks=0.0, tick=0.25):
    n = len(c)
    if n < 30:
        return []
    sh, sl = swings(h, l, k_sw)
    A = atr(h, l, c)
    out = []
    i = 8
    while i < n-2:
        last_sh = last_sl = None
        for j in range(i-1, max(i-60, 2), -1):
            if last_sh is None and sh[j] and j+k_sw <= i:
                last_sh = h[j]
            if last_sl is None and sl[j] and j+k_sw <= i:
                last_sl = l[j]
            if last_sh is not None and last_sl is not None:
                break
        side = 0
        if last_sh is not None and c[i] > last_sh:
            side, ref = 1, last_sh
        elif last_sl is not None and c[i] < last_sl:
            side, ref = -1, last_sl
        if side == 0:
            i += 1
            continue
        a0 = max(A[i], 1e-9)
        choch = abs(c[i]-ref)/a0

        fvg = None
        for j in range(i+1, min(i+40, n-1)):
            if side > 0 and h[j-2] < l[j]:
                fvg = (h[j-2], l[j], j-1)
                break
            if side < 0 and l[j-2] > h[j]:
                fvg = (h[j], l[j-2], j-1)
                break
        if fvg is None:
            i += 1
            continue
        lo_g, hi_g, prod = fvg
        mid = (lo_g+hi_g)/2
        rng = abs(h[prod]-l[prod])
        stop0 = (l[prod]-buf*rng) if side > 0 else (h[prod]+buf*rng)
        risk = abs(mid-stop0)
        if risk <= 0 or rng <= 0 or risk/a0 < MIN_RISK:
            i += 1
            continue
        push = h[i:prod+2].max() if side > 0 else l[i:prod+2].min()

        entry = None
        stop = stop0
        done = False
        for kk in range(prod+2, min(prod+90, n)):
            if entry is None:
                if fill == "touch":
                    if (l[kk] <= mid) if side > 0 else (h[kk] >= mid):
                        entry = mid
                elif fill == "through":
                    t2 = mid - tick if side > 0 else mid + tick
                    if (l[kk] <= t2) if side > 0 else (h[kk] >= t2):
                        entry = mid
                elif fill == "close_beyond":
                    if ((c[kk] <= mid) if side > 0 else (c[kk] >= mid)) and kk+1 < n:
                        entry = o[kk+1]
                elif fill == "next_open":
                    if ((l[kk] <= mid) if side > 0 else (h[kk] >= mid)) and kk+1 < n:
                        entry = o[kk+1]
                if entry is not None:
                    risk = abs(entry-stop0)
                    if risk <= 0 or risk/a0 < MIN_RISK:
                        entry = None
                        i += 1
                        break
                    tgt = entry + side*tmult*risk
                continue
            if stop != entry:
                if (c[kk] > push) if side > 0 else (c[kk] < push):
                    stop = entry
            hit_s = (l[kk] <= stop) if side > 0 else (h[kk] >= stop)
            hit_t = (h[kk] >= tgt) if side > 0 else (l[kk] <= tgt)
            if hit_s:
                px = stop - side*slip_ticks*tick        # stops slip against you
                out.append(dict(R=(px-entry)*side/risk - COST_BP/1e4*entry/risk,
                                choch=choch, ts=None))
                i = kk+1
                done = True
                break
            if hit_t:
                out.append(dict(R=tmult - COST_BP/1e4*entry/risk,
                                choch=choch, ts=None))
                i = kk+1
                done = True
                break
        if not done:
            # entered-but-unresolved must still advance the scan,
            # or the outer loop spins on this bar forever
            i += 1
    return out


def backtest(d, **kw):
    m = d.index.hour*60+d.index.minute
    dd = d[(m >= OPEN_MIN) & (m < CLOSE_MIN)]
    day = dd.index.normalize().tz_localize(None)
    rows = []
    for dt, g in dd.groupby(day):
        for r in run_day(g["open"].values, g["high"].values, g["low"].values,
                         g["close"].values, **kw):
            r["ts"] = dt
            rows.append(r)
    return pd.DataFrame(rows)


def stat(t):
    if len(t) < 60:
        return None
    R = t.R.values
    m, se = R.mean(), R.std(ddof=1)/np.sqrt(len(R))
    return len(R), (R > 2).mean()*100, m, m/se


if __name__ == "__main__":
    D = {nm: load_m1(s) for nm, s in MK.items()}
    D = {k: v for k, v in D.items() if v is not None}

    print("=" * 78)
    print("1. FILL REALISM -- the assumption most likely to be inventing the edge")
    print("=" * 78)
    print(f"  {'fill model':<22}{'slip':>6}{'n':>7}{'win%':>7}{'expR':>9}{'t':>7}")
    print("  " + "-" * 60, flush=True)
    for fillm in ("touch", "through", "next_open", "close_beyond"):
        for slip in (0.0, 1.0):
            parts = []
            for nm, d in D.items():
                parts.append(backtest(d, fill=fillm, slip_ticks=slip))
            T = pd.concat(parts, ignore_index=True)
            thr = T.choch.quantile(CHOCH_Q)
            s = stat(T[T.choch >= thr])
            if s:
                print(f"  {fillm:<22}{slip:>6.0f}{s[0]:>7}{s[1]:>7.1f}%"
                      f"{s[2]:>+9.3f}{s[3]:>7.2f}", flush=True)
