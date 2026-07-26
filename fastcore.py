"""Numpy-array reimplementation of the strategy + trade simulation.

Same logic as strategy.py / backtest.py, ~50x faster, so we can search
hundreds of configs. `verify.py` asserts this produces IDENTICAL trades to
the readable pandas version -- speed must never buy us a silent behavior
change.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from strategy import KILLZONES
from backtest import STOP_SLIP, ENTRY_SLIP

NS_2H = np.int64(2 * 3600 * 1_000_000_000)


def prep(df: pd.DataFrame, swing_lb: int, zones=("ny_am",)) -> dict:
    """Precompute everything that doesn't depend on the searched params."""
    n = len(df)
    o = {"high": df["high"].to_numpy(float), "low": df["low"].to_numpy(float),
         "close": df["close"].to_numpy(float), "n": n,
         "ts": df.index.view("int64") if hasattr(df.index, "view")
               else df.index.astype("int64").to_numpy()}
    w = 2 * swing_lb + 1
    o["is_high"] = (df["high"].rolling(w, center=True).max().to_numpy() == o["high"])
    o["is_low"] = (df["low"].rolling(w, center=True).min().to_numpy() == o["low"])
    o["is_high"][:swing_lb] = False
    o["is_high"][n - swing_lb:] = False
    o["is_low"][:swing_lb] = False
    o["is_low"][n - swing_lb:] = False
    # killzone mask
    mins = np.asarray(df.index.hour) * 60 + np.asarray(df.index.minute)
    mask = np.zeros(n, bool)
    for z in zones:
        (sh, sm), (eh, em) = KILLZONES[z]
        mask |= (mins >= sh * 60 + sm) & (mins <= eh * 60 + em)
    o["kz"] = mask
    for col in ("atr", "ema", "atr_rank"):
        o[col] = df[col].to_numpy(float) if col in df.columns else np.full(n, np.nan)
    return o


def signals_fast(P: dict, rr=2.0, swing_lb=2, fvg_window=12,
                 min_risk_pct=0.0005, trend_filter=False, min_gap_atr=0.0,
                 atr_rank_min=0.0, atr_rank_max=1.0, max_risk_atr=0.0):
    high, low, close = P["high"], P["low"], P["close"]
    is_high, is_low, kz, ts = P["is_high"], P["is_low"], P["kz"], P["ts"]
    atr, ema, arank = P["atr"], P["ema"], P["atr_rank"]
    n = P["n"]
    out_i, out_side, out_entry, out_stop, out_tgt = [], [], [], [], []
    lsh = lsl = np.nan
    armed = 0            # 0 none, 1 long, -1 short
    sweep = np.nan
    expiry = -1
    busy_until = np.int64(-1)

    for i in range(swing_lb, n):
        conf = i - swing_lb
        if is_high[conf]:
            lsh = high[conf]
        if is_low[conf]:
            lsl = low[conf]

        if not np.isnan(lsl) and low[i] < lsl:
            armed, sweep, expiry = 1, low[i], i + fvg_window
        elif not np.isnan(lsh) and high[i] > lsh:
            armed, sweep, expiry = -1, high[i], i + fvg_window

        if armed != 0 and i > expiry:
            armed = 0
        if armed == 0 or not kz[i] or ts[i] <= busy_until:
            continue

        # fair value gap on (i-2, i)
        if armed == 1:
            if not (high[i - 2] < low[i]):
                continue
            g_lo, g_hi = high[i - 2], low[i]
        else:
            if not (low[i - 2] > high[i]):
                continue
            g_lo, g_hi = high[i], low[i - 2]

        a = atr[i]
        if trend_filter and not np.isnan(ema[i]):
            if armed == 1 and close[i] < ema[i]:
                continue
            if armed == -1 and close[i] > ema[i]:
                continue
        if min_gap_atr > 0 and not np.isnan(a) and a > 0:
            if (g_hi - g_lo) < min_gap_atr * a:
                continue
        if atr_rank_min > 0 or atr_rank_max < 1:
            r = arank[i]
            if np.isnan(r) or not (atr_rank_min <= r <= atr_rank_max):
                continue

        if armed == 1:
            entry, stop = g_hi, sweep - 1e-9
            risk = entry - stop
            tgt = entry + rr * risk
        else:
            entry, stop = g_lo, sweep + 1e-9
            risk = stop - entry
            tgt = entry - rr * risk
        if risk <= max(1e-6, min_risk_pct * entry):
            continue
        if max_risk_atr > 0 and not np.isnan(a) and a > 0 and risk > max_risk_atr * a:
            continue

        out_i.append(i); out_side.append(armed)
        out_entry.append(entry); out_stop.append(stop); out_tgt.append(tgt)
        busy_until = ts[i] + NS_2H
        armed = 0
    return (np.array(out_i, int), np.array(out_side, int),
            np.array(out_entry), np.array(out_stop), np.array(out_tgt))


def simulate_fast(P: dict, sig, max_bars=60) -> np.ndarray:
    """Returns array of R multiples for resolved trades."""
    high, low = P["high"], P["low"]
    idx, side, entry, stop, tgt = sig
    n = P["n"]
    Rs = []
    for k in range(len(idx)):
        i, sd, e0, st, tg = idx[k], side[k], entry[k], stop[k], tgt[k]
        risk = abs(e0 - st)
        if sd == 1:
            fill, stop_fill = e0 + ENTRY_SLIP, st - STOP_SLIP
        else:
            fill, stop_fill = e0 - ENTRY_SLIP, st + STOP_SLIP
        filled = False
        for j in range(i + 1, min(i + 1 + max_bars, n)):
            if not filled:
                if low[j] <= e0 <= high[j]:
                    filled = True
                else:
                    continue
            if sd == 1:
                if low[j] <= st:
                    Rs.append((stop_fill - fill) / risk); break
                if high[j] >= tg:
                    Rs.append((tg - fill) / risk); break
            else:
                if high[j] >= st:
                    Rs.append((fill - stop_fill) / risk); break
                if low[j] <= tg:
                    Rs.append((fill - tg) / risk); break
    return np.array(Rs)


def stats(R: np.ndarray, min_n: int = 30) -> dict:
    n = len(R)
    if n < min_n:
        return {"n": n, "win": np.nan, "expR": np.nan, "pf": np.nan,
                "ddR": np.nan, "t": np.nan, "totR": np.nan}
    eq = R.cumsum()
    w, l = R[R > 0].sum(), abs(R[R <= 0].sum())
    return {"n": n, "win": (R > 0).mean() * 100, "expR": R.mean(),
            "pf": w / l if l else np.inf,
            "ddR": float((np.maximum.accumulate(eq) - eq).max()),
            "t": R.mean() / (R.std(ddof=1) / np.sqrt(n)), "totR": R.sum()}
