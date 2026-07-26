"""Liquidity-pool targeting — the exit rule we got wrong.

Our backtests used FIXED targets (1R, 2R). TJR doesn't trade fixed R: he
targets the next pool of liquidity — old highs/lows, equal highs/lows, the
prior day's extremes, session extremes. Winners run to wherever liquidity
sits, which might be 0.8R or 5R.

If the edge lives anywhere in his approach, this is a strong candidate,
because it changes the entire return distribution — a few large winners
instead of many capped ones.

NO LOOKAHEAD: candidate levels at entry are only
  - prior session high/low        (known at the open)
  - today's high/low SO FAR       (known)
  - swing highs/lows confirmed >= swing_lb bars ago
"""
from __future__ import annotations
import numpy as np
import pandas as pd

STOP_SLIP = 1.5


def build_signals(df: pd.DataFrame, swing_lb=2, fvg_window=12,
                  min_risk_pct=0.0005, kz=(510, 660)):
    """Yield (i, side, entry, stop) for each sweep+FVG signal. Same detection
    logic as strategy.py, no lookahead."""
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    n = len(df)
    w = 2 * swing_lb + 1
    is_h = (df["high"].rolling(w, center=True).max() == df["high"]).to_numpy()
    is_l = (df["low"].rolling(w, center=True).min() == df["low"]).to_numpy()
    is_h[:swing_lb] = is_h[n - swing_lb:] = False
    is_l[:swing_lb] = is_l[n - swing_lb:] = False
    mins = np.asarray(df.index.hour) * 60 + np.asarray(df.index.minute)
    inkz = (mins >= kz[0]) & (mins <= kz[1])

    lsh = lsl = np.nan
    armed, sweep_i, expiry = 0, -1, -1
    out = []
    for i in range(swing_lb, n):
        conf = i - swing_lb
        if is_h[conf]:
            lsh = h[conf]
        if is_l[conf]:
            lsl = l[conf]
        if not np.isnan(lsl) and l[i] < lsl:
            armed, sweep_i, expiry = 1, i, i + fvg_window
        elif not np.isnan(lsh) and h[i] > lsh:
            armed, sweep_i, expiry = -1, i, i + fvg_window
        if armed != 0 and i > expiry:
            armed = 0
        if armed == 0 or not inkz[i]:
            continue
        if armed == 1:
            if not (h[i - 2] < l[i]):
                continue
            entry, stop = l[i], l[sweep_i]
        else:
            if not (l[i - 2] > h[i]):
                continue
            entry, stop = h[i], h[sweep_i]
        risk = abs(entry - stop)
        if risk <= max(1e-6, min_risk_pct * entry):
            continue
        out.append((i, armed, entry, stop))
        armed = 0
    return out, is_h, is_l


def liquidity_levels(df, i, side, entry, is_h, is_l, swing_lb, lookback=500):
    """Candidate target levels ABOVE entry (long) / BELOW entry (short),
    using only information available at bar i."""
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    idx = df.index
    day = idx[i].normalize()
    levels = []

    # prior session extremes
    prior = df[idx.normalize() < day]
    if len(prior):
        pday = prior.index.normalize()[-1]
        pmask = pday == prior.index.normalize()
        levels += [float(prior["high"][pmask].max()), float(prior["low"][pmask].min())]

    # today's extremes so far (strictly before i)
    todays = df.iloc[max(0, i - 200):i]
    todays = todays[todays.index.normalize() == day]
    if len(todays):
        levels += [float(todays["high"].max()), float(todays["low"].min())]

    # confirmed swing points (confirmation lag respected)
    lo_i = max(swing_lb, i - lookback)
    for j in range(lo_i, i - swing_lb):
        if is_h[j]:
            levels.append(float(h[j]))
        if is_l[j]:
            levels.append(float(l[j]))

    if side == 1:
        cand = sorted(set(x for x in levels if x > entry))
    else:
        cand = sorted(set(x for x in levels if x < entry), reverse=True)
    return cand


def simulate(df, mode="liq", rr=1.0, swing_lb=2, min_rr=0.5, max_rr=10.0,
             pool_index=0, max_bars=60):
    """mode: 'fixed' (rr multiple) or 'liq' (nearest liquidity pool)."""
    sigs, is_h, is_l = build_signals(df, swing_lb=swing_lb)
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(df)
    rows = []
    for (i, side, entry, stop) in sigs:
        risk = abs(entry - stop)
        if mode == "fixed":
            target = entry + rr * risk if side == 1 else entry - rr * risk
        else:
            cand = liquidity_levels(df, i, side, entry, is_h, is_l, swing_lb)
            # keep pools at least min_rr away, take the pool_index-th
            ok = [x for x in cand if abs(x - entry) >= min_rr * risk
                  and abs(x - entry) <= max_rr * risk]
            if not ok:
                continue
            target = ok[min(pool_index, len(ok) - 1)]
        tR = abs(target - entry) / risk
        filled = False
        R = np.nan
        for j in range(i + 1, min(i + 1 + max_bars, n)):
            if not filled:
                if l[j] <= entry <= h[j]:
                    filled = True
                else:
                    continue
            if side == 1:
                if l[j] <= stop:
                    R = (stop - STOP_SLIP - entry) / risk
                    break
                if h[j] >= target:
                    R = tR
                    break
            else:
                if h[j] >= stop:
                    R = (entry - stop - STOP_SLIP) / risk
                    break
                if l[j] <= target:
                    R = tR
                    break
        if np.isfinite(R):
            rows.append({"R": R, "target_R": tR, "side": side})
    return pd.DataFrame(rows)


def stats(t: pd.DataFrame) -> dict:
    if t.empty or len(t) < 30:
        return {"n": len(t)}
    R = t.R.values
    return {"n": len(R), "expR": R.mean(), "win%": (R > 0).mean() * 100,
            "avg_tgtR": t.target_R.mean(),
            "t": R.mean() / (R.std(ddof=1) / np.sqrt(len(R)))}
