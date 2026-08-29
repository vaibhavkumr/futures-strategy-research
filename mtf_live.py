"""Live signal function for the multi-timeframe model.

Same structure as mtf.generate but returns only the most recent actionable
signal, so live_paper.py can drive it with its existing accounting, risk
controls, news blackout and logging.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mtf


def find_signal_mtf(df5: pd.DataFrame, *, use_bias=True, use_mss=True,
                    entry_mode="both", eq_tol_atr=0.20, sweep_window=36):
    """Return (ts, side, entry, stop, risk) for the latest signal, or None."""
    if len(df5) < 60:
        return None
    ctx = mtf.htf_bias(df5)
    pools = mtf.liquidity_pools(df5)
    h, l, c, o = (df5[x].to_numpy(float) for x in ("high", "low", "close", "open"))
    n = len(df5)
    tr = pd.concat([df5["high"] - df5["low"],
                    (df5["high"] - df5["close"].shift()).abs(),
                    (df5["low"] - df5["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().to_numpy()
    bias = ctx["bias"].to_numpy()

    r15 = mtf.resample(df5, "15min")
    if len(r15) < 5:
        return None
    sh15, sl15 = mtf.swings(r15, 1)
    last_sh = pd.Series(np.where(sh15, r15["high"], np.nan),
                        index=r15.index).ffill().shift(1)
    last_sl = pd.Series(np.where(sl15, r15["low"], np.nan),
                        index=r15.index).ffill().shift(1)
    sh5 = last_sh.reindex(df5.index, method="ffill").to_numpy()
    sl5 = last_sl.reindex(df5.index, method="ffill").to_numpy()

    P = {k: pools[k].to_numpy(float) for k in pools.columns}
    is_h5, is_l5 = mtf.swings(df5, 2)
    sw_hi, sw_lo = [], []
    armed, swept, armed_i, mss = 0, np.nan, -1, False
    last_sig = None

    for i in range(30, n):
        conf = i - 2
        if is_h5[conf]:
            sw_hi.append(h[conf])
        if is_l5[conf]:
            sw_lo.append(l[conf])
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        highs = [P[k][i] for k in ("pdh", "asia_h", "lon_h")]
        lows = [P[k][i] for k in ("pdl", "asia_l", "lon_l")]
        for arr, dst in ((sw_hi, highs), (sw_lo, lows)):
            rec = arr[-40:]
            for p in set(rec):
                if sum(1 for q in rec if abs(q - p) <= eq_tol_atr * a) >= 2:
                    dst.append(p)
        highs = sorted({x for x in highs if np.isfinite(x)})
        lows = sorted({x for x in lows if np.isfinite(x)})

        hit_lo = [x for x in lows if l[i] < x <= l[i] + 2 * a]
        hit_hi = [x for x in highs if h[i] > x >= h[i] - 2 * a]
        if hit_lo:
            armed, swept, armed_i, mss = 1, max(hit_lo), i, False
        elif hit_hi:
            armed, swept, armed_i, mss = -1, min(hit_hi), i, False
        if armed != 0 and i - armed_i > sweep_window:
            armed = 0
        if armed == 0:
            continue
        if use_bias and bias[i] != 0 and bias[i] != armed:
            continue
        if use_mss and not mss:
            ref = sh5[i] if armed == 1 else sl5[i]
            if not np.isfinite(ref):
                continue
            if (armed == 1 and c[i] > ref) or (armed == -1 and c[i] < ref):
                mss = True
            else:
                continue

        entry = np.nan
        if entry_mode in ("fvg", "both"):
            if armed == 1 and h[i - 2] < l[i]:
                entry = l[i]
            elif armed == -1 and l[i - 2] > h[i]:
                entry = h[i]
        if not np.isfinite(entry) and entry_mode in ("ob", "both"):
            for k in range(i - 1, max(i - 8, armed_i - 1), -1):
                if armed == 1 and c[k] < o[k]:
                    entry = h[k]
                    break
                if armed == -1 and c[k] > o[k]:
                    entry = l[k]
                    break
        if not np.isfinite(entry):
            continue

        sgn = armed
        stop = (l[armed_i] - 0.1 * a) if sgn == 1 else (h[armed_i] + 0.1 * a)
        risk = abs(entry - stop)
        if risk < 0.2 * a or risk > 4 * a:
            continue
        last_sig = (df5.index[i], "long" if sgn == 1 else "short",
                    float(entry), float(stop), float(risk))
        armed = 0
    return last_sig
