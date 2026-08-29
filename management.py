"""Trade MANAGEMENT — the dimension we never tested.

Every test so far asked "which trades should we take?". Each trade was a
fixed bracket: stop here, target there, walk away. That is not how a
discretionary trader operates, and it forces the entire return distribution
into two outcomes.

Here we hold entries CONSTANT (the same fair-odds signals) and vary only the
exit, which is an orthogonal dimension:

  fixed_R        baseline: stop, fixed target
  breakeven      move stop to entry once +X R is reached
  trail_atr      trail the stop by k*ATR behind the high-water mark
  partial        take half at +X R, let the rest run to a further target
  time_stop      exit at market after N bars regardless
  runner         breakeven stop + no target at all, ride until stopped

THEORY NOTE, stated up front: for a pure martingale no exit rule can create
positive expectancy -- that is a theorem, not an opinion. Management can only
help if real price paths have exploitable structure (momentum or reversion)
that a fixed bracket throws away. So this is a genuine test of whether such
structure exists, not a search for a free lunch.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

STOP_SLIP = 1.5


def simulate_managed(paths, mode="fixed_R", tmult=1.0, be_trigger=0.5,
                     trail_k=1.0, partial_at=0.5, runner_target=3.0,
                     time_bars=24):
    """paths: list of dicts with entry, stop, risk, sgn, and the arrays of
    highs/lows AFTER entry (already fill-adjusted). Returns array of R."""
    out = []
    for p in paths:
        e, st, risk, sgn = p["entry"], p["stop"], p["risk"], p["sgn"]
        H, L, A = p["H"], p["L"], p["atr"]
        stop = st
        tgt = e + sgn * tmult * risk
        half_done = False
        realized = 0.0
        R = None
        peak = 0.0
        for k in range(len(H)):
            hi, lo = H[k], L[k]
            fav = (hi - e) * sgn if sgn == 1 else (e - lo)
            adv = (e - lo) * sgn if sgn == 1 else (hi - e)
            fav_R = ((hi - e) if sgn == 1 else (e - lo)) / risk
            peak = max(peak, fav_R)

            # --- adjust stop per mode (before checking exits) ---
            if mode in ("breakeven", "partial", "runner") and peak >= be_trigger:
                stop = max(stop, e) if sgn == 1 else min(stop, e)
            if mode == "trail_atr" and peak > 0:
                hw = e + sgn * peak * risk
                cand = hw - sgn * trail_k * A
                stop = max(stop, cand) if sgn == 1 else min(stop, cand)

            hit_stop = (lo <= stop) if sgn == 1 else (hi >= stop)
            if hit_stop:
                px = stop - sgn * STOP_SLIP
                r = ((px - e) if sgn == 1 else (e - px)) / risk
                R = realized + r * (0.5 if half_done else 1.0)
                break

            if mode == "partial" and not half_done and fav_R >= partial_at:
                realized += 0.5 * partial_at
                half_done = True
                tgt = e + sgn * runner_target * risk
            if mode != "runner":
                hit_t = (hi >= tgt) if sgn == 1 else (lo <= tgt)
                if hit_t:
                    tR = ((tgt - e) if sgn == 1 else (e - tgt)) / risk
                    R = realized + tR * (0.5 if half_done else 1.0)
                    break
            if mode == "time_stop" and k >= time_bars:
                cl = p["C"][k]
                r = ((cl - e) if sgn == 1 else (e - cl)) / risk
                R = realized + r * (0.5 if half_done else 1.0)
                break
        if R is None:
            cl = p["C"][-1]
            r = ((cl - e) if sgn == 1 else (e - cl)) / risk
            R = realized + r * (0.5 if half_done else 1.0)
        out.append(R)
    return np.array(out)


def build_paths(df: pd.DataFrame, signals, horizon=60):
    """signals: iterable of (i, side, entry, stop, risk). Captures the price
    path AFTER the fill so any exit rule can be replayed on it."""
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().to_numpy()
    n = len(df)
    paths = []
    busy = -1
    for (i, side, entry, stop, risk) in signals:
        if i <= busy:
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        sgn = 1 if side == "long" else -1
        fill = None
        for j in range(i + 1, min(i + 20, n)):
            if l[j] <= entry <= h[j]:
                fill = j
                break
        if fill is None:
            continue
        end = min(fill + 1 + horizon, n)
        if end - (fill + 1) < 5:
            continue
        paths.append({"entry": entry, "stop": stop, "risk": risk, "sgn": sgn,
                      "atr": a, "ts": df.index[i],
                      "H": h[fill + 1:end], "L": l[fill + 1:end],
                      "C": c[fill + 1:end]})
        busy = fill
    return paths
