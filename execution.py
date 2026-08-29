"""EXECUTION MODELS — recreating how a disciplined trader actually exits.

Our bot does the naive thing: a hard market stop resting in the book. Any
wick through it takes you out, at market, with slippage. That is the single
most expensive habit in retail trading and it is what every backtest we ran
assumed.

A disciplined discretionary trader does something different. The variants
here are all things real traders actually do:

  market_stop    baseline -- hard stop, intrabar touch, market fill (worst)
  close_beyond   exit only if a bar CLOSES beyond the level. Wicks do not
                 take you out. You accept a worse fill when price runs, in
                 exchange for not being stopped by noise.
  close_confirm  same, but require the close beyond AND exit at next open
                 (realistic: you see the close, then act)
  limit_target   target filled as a resting LIMIT (no slippage) while the
                 stop is close_beyond -- the asymmetric setup traders use
  liquid_hours   only trade the tightest-spread window (NY 09:30-11:30)

Our measured break-even is ~0.69 pts of slippage. The question is whether
execution discipline alone gets under it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def simulate(df: pd.DataFrame, signals, mode="market_stop", tmult=1.0,
             slip_pts=1.5, close_slip=0.5, horizon=60, hours=None):
    """signals: (i, side, entry, stop, risk). Returns DataFrame of outcomes."""
    h, l, c, o = (df[x].to_numpy(float) for x in ("high", "low", "close", "open"))
    n = len(df)
    hrs = np.asarray(df.index.hour) + np.asarray(df.index.minute) / 60
    rows, busy = [], -1
    for (i, side, entry, stop, risk) in signals:
        if i <= busy:
            continue
        if hours is not None and not (hours[0] <= hrs[i] <= hours[1]):
            continue
        sgn = 1 if side == "long" else -1
        tgt = entry + sgn * tmult * risk
        filled, R, last = False, None, i
        for j in range(i + 1, min(i + 1 + horizon, n)):
            if not filled:
                if l[j] <= entry <= h[j]:
                    filled, last = True, j
                continue

            if mode == "market_stop":
                hit_stop = (l[j] <= stop) if sgn == 1 else (h[j] >= stop)
                stop_px = stop - sgn * slip_pts
            else:
                # only a CLOSE beyond the level counts -- wicks are ignored
                hit_stop = (c[j] <= stop) if sgn == 1 else (c[j] >= stop)
                if mode == "close_confirm" and hit_stop and j + 1 < n:
                    stop_px = o[j + 1] - sgn * close_slip
                else:
                    stop_px = c[j] - sgn * close_slip

            hit_tgt = (h[j] >= tgt) if sgn == 1 else (l[j] <= tgt)

            if hit_stop:
                R = ((stop_px - entry) if sgn == 1 else (entry - stop_px)) / risk
            elif hit_tgt:
                R = tmult                      # resting limit: no slippage
            if R is not None:
                last = j
                break
        busy = last
        if R is not None:
            rows.append({"ts": df.index[i], "R": R, "side": side})
    return pd.DataFrame(rows)


def gen_atr_signals(df: pd.DataFrame, swing_lb=2, fvg_window=12):
    """Sweep + FVG with an ATR stop (the best structure we found)."""
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(df)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().to_numpy()
    w = 2 * swing_lb + 1
    is_h = (df["high"].rolling(w, center=True).max() == df["high"]).to_numpy()
    is_l = (df["low"].rolling(w, center=True).min() == df["low"]).to_numpy()
    is_h[:swing_lb] = is_h[n - swing_lb:] = False
    is_l[:swing_lb] = is_l[n - swing_lb:] = False
    lsh = lsl = np.nan
    armed, expiry = 0, -1
    out = []
    for i in range(swing_lb, n):
        conf = i - swing_lb
        if is_h[conf]:
            lsh = h[conf]
        if is_l[conf]:
            lsl = l[conf]
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if not np.isnan(lsl) and l[i] < lsl:
            armed, expiry = 1, i + fvg_window
        elif not np.isnan(lsh) and h[i] > lsh:
            armed, expiry = -1, i + fvg_window
        if armed != 0 and i > expiry:
            armed = 0
        if armed == 0:
            continue
        if armed == 1 and h[i - 2] < l[i]:
            entry = l[i]
        elif armed == -1 and l[i - 2] > h[i]:
            entry = h[i]
        else:
            continue
        sgn = armed
        stop = entry - sgn * a
        risk = abs(entry - stop)
        if risk >= max(1e-6, 0.0005 * entry):
            out.append((i, "long" if sgn == 1 else "short", entry, stop, risk))
        armed = 0
    return out
