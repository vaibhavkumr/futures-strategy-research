"""TJR's actual 4-step method, implemented from the strategy video.

Every previous version of this project got the entry wrong. The video is
explicit that entering at the reversal signal gives a bad entry -- you wait
for a retrace and trigger on the 1-minute. We were entering at the exact
point he warns against, on every single trade.

THE FOUR STEPS
  1 MANIPULATION   price sweeps a draw on liquidity: session high/low,
                   1H high/low, 4H high/low. The same levels are targets.
  2 REVERSAL       on the 5-minute: a BREAK OF STRUCTURE (candle CLOSING
                   beyond the most recent swing) or an INVERSE FVG.
  3 RETRACE        do NOT enter yet. Wait for a 5m retrace, identified as a
                   1-minute break of structure in the OPPOSITE direction.
  4 ENTRY          1-minute break of structure back in the trade direction.

  STOP    beyond the swing that formed during the retrace (the "second high")
  TARGET  the next opposing draw on liquidity

TWO DEFINITIONS WE HAD BACKWARDS
  BREAK OF STRUCTURE is a CLOSE beyond a swing, not a wick through it.
  INVERSE FVG is a fair value gap price CLOSES THROUGH -- a failure of the
  gap, the opposite of the gap forming. We were trading gap formation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------- structure
def swing_points(df: pd.DataFrame, lb: int = 2):
    """Fractal swings. Confirmed lb bars late -- callers must respect that."""
    w = 2 * lb + 1
    hi = (df["high"].rolling(w, center=True).max() == df["high"]).to_numpy()
    lo = (df["low"].rolling(w, center=True).min() == df["low"]).to_numpy()
    n = len(df)
    hi[:lb] = hi[n - lb:] = False
    lo[:lb] = lo[n - lb:] = False
    return hi, lo


def break_of_structure(c, swing_level, direction) -> bool:
    """A CLOSE beyond the most recent swing. Not a wick."""
    if not np.isfinite(swing_level):
        return False
    return c > swing_level if direction > 0 else c < swing_level


def find_inverse_fvg(h, l, c, i, direction) -> bool:
    """An inverse FVG: a gap that price CLOSES THROUGH.

    A bullish gap sits between high[k-2] and low[k]. If a later candle closes
    BELOW that gap, the gap failed -- bullish order flow was rejected, which
    is a bearish signal. Mirror for bearish gaps.
    """
    for k in range(max(i - 12, 2), i):
        if direction < 0:                      # looking for a bearish signal
            if h[k - 2] < l[k]:                # a bullish gap existed
                if c[i] < h[k - 2]:            # closed below it -> inverted
                    return True
        else:                                  # bullish signal
            if l[k - 2] > h[k]:                # a bearish gap existed
                if c[i] > l[k - 2]:            # closed above it -> inverted
                    return True
    return False


# ---------------------------------------------------------- liquidity draws
def draws_on_liquidity(df5: pd.DataFrame) -> pd.DataFrame:
    """Session, 1H and 4H highs/lows -- the levels the video names.
    Each becomes available only once the period that formed it has closed."""
    out = pd.DataFrame(index=df5.index)
    day = df5.index.normalize()
    hh = np.asarray(df5.index.hour)
    mins = hh * 60 + np.asarray(df5.index.minute)

    d = df5.groupby(day)
    out["pdh"] = day.map(d["high"].max().shift(1))
    out["pdl"] = day.map(d["low"].min().shift(1))

    # Asia 20:00-02:00 (spans midnight -> belongs to the NEXT session)
    sess = pd.DatetimeIndex(np.where(hh >= 20, day + pd.Timedelta(days=1), day))
    m = (hh >= 20) | (hh < 2)
    a = df5[m]
    out["asia_h"] = day.map(a.groupby(sess[m])["high"].max())
    out["asia_l"] = day.map(a.groupby(sess[m])["low"].min())
    # London 02:00-05:00
    m2 = (hh >= 2) & (hh < 5)
    lo = df5[m2]
    out["lon_h"] = day.map(lo.groupby(lo.index.normalize())["high"].max())
    out["lon_l"] = day.map(lo.groupby(lo.index.normalize())["low"].min())

    # a range is unknown until its session closes
    for col in ("asia_h", "asia_l"):
        out.loc[mins < 120, col] = np.nan
    for col in ("lon_h", "lon_l"):
        out.loc[mins < 300, col] = np.nan

    # 1H and 4H swing highs/lows, shifted so only closed candles are used
    for rule, tag in (("1h", "h1"), ("4h", "h4")):
        r = df5.resample(rule).agg({"high": "max", "low": "min"}).dropna()
        sh, sl = swing_points(r, 1)
        hi = pd.Series(np.where(sh, r["high"], np.nan), index=r.index).ffill().shift(2)
        lw = pd.Series(np.where(sl, r["low"], np.nan), index=r.index).ffill().shift(2)
        out[f"{tag}_h"] = hi.reindex(df5.index, method="ffill")
        out[f"{tag}_l"] = lw.reindex(df5.index, method="ffill")
    return out


# ------------------------------------------------------------------- engine
def generate(df5: pd.DataFrame, df1: pd.DataFrame, *,
             sweep_window=48, retrace_window=36, entry_window=24,
             max_bars=120, slip_frac=0.05, swing_lb=2,
             require_retrace=True, use_ifvg=True):
    """df5 = 5-minute bars, df1 = 1-minute bars (same instrument/period)."""
    P = draws_on_liquidity(df5)
    h5, l5, c5 = (df5[x].to_numpy(float) for x in ("high", "low", "close"))
    n5 = len(df5)
    tr = pd.concat([df5["high"] - df5["low"],
                    (df5["high"] - df5["close"].shift()).abs(),
                    (df5["low"] - df5["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().to_numpy()

    sh5, sl5 = swing_points(df5, swing_lb)
    last_h5 = pd.Series(np.where(sh5, h5, np.nan)).ffill().shift(swing_lb).to_numpy()
    last_l5 = pd.Series(np.where(sl5, l5, np.nan)).ffill().shift(swing_lb).to_numpy()

    h1, l1, c1 = (df1[x].to_numpy(float) for x in ("high", "low", "close"))
    n1 = len(df1)
    sh1, sl1 = swing_points(df1, 1)
    last_h1 = pd.Series(np.where(sh1, h1, np.nan)).ffill().shift(1).to_numpy()
    last_l1 = pd.Series(np.where(sl1, l1, np.nan)).ffill().shift(1).to_numpy()
    idx1 = df1.index
    pos1 = {t: k for k, t in enumerate(idx1)}

    cols = {k: P[k].to_numpy(float) for k in P.columns}
    LEVELS_H = ("pdh", "asia_h", "lon_h", "h1_h", "h4_h")
    LEVELS_L = ("pdl", "asia_l", "lon_l", "h1_l", "h4_l")

    rows = []
    busy_ts = None

    for i in range(30, n5 - 1):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        ts = df5.index[i]
        if busy_ts is not None and ts <= busy_ts:
            continue

        highs = sorted({cols[k][i] for k in LEVELS_H if np.isfinite(cols[k][i])})
        lows = sorted({cols[k][i] for k in LEVELS_L if np.isfinite(cols[k][i])})

        # ---- STEP 1: manipulation (sweep of a real draw) ----
        swept_hi = [x for x in highs if h5[i] > x >= h5[i] - 2 * a]
        swept_lo = [x for x in lows if l5[i] < x <= l5[i] + 2 * a]
        if swept_hi:
            sgn = -1
            level = min(swept_hi)
        elif swept_lo:
            sgn = 1
            level = max(swept_lo)
        else:
            continue

        # ---- STEP 2: 5m break of structure OR inverse FVG ----
        conf_i = None
        for j in range(i + 1, min(i + 1 + sweep_window, n5)):
            ref = last_h5[j] if sgn > 0 else last_l5[j]
            bos = break_of_structure(c5[j], ref, sgn)
            ifvg = use_ifvg and find_inverse_fvg(h5, l5, c5, j, sgn)
            if bos or ifvg:
                conf_i = j
                break
        if conf_i is None:
            continue

        # ---- STEPS 3 & 4 on the 1-minute chart ----
        t_conf = df5.index[conf_i]
        k0 = pos1.get(t_conf)
        if k0 is None:
            nxt = idx1.searchsorted(t_conf)
            if nxt >= n1:
                continue
            k0 = int(nxt)

        # STEP 3: retrace = 1m break of structure AGAINST the trade
        r_k = None
        for k in range(k0 + 1, min(k0 + 1 + retrace_window, n1)):
            ref = last_h1[k] if sgn < 0 else last_l1[k]
            if break_of_structure(c1[k], ref, -sgn):
                r_k = k
                break
        if require_retrace and r_k is None:
            continue
        start_k = r_k if r_k is not None else k0

        # STEP 4: 1m break of structure back IN the trade direction
        e_k = None
        for k in range(start_k + 1, min(start_k + 1 + entry_window, n1)):
            ref = last_h1[k] if sgn > 0 else last_l1[k]
            if break_of_structure(c1[k], ref, sgn):
                e_k = k
                break
        if e_k is None:
            continue

        entry = c1[e_k]
        # stop beyond the swing formed during the retrace ("the second high")
        seg = slice(start_k, e_k + 1)
        stop = (np.min(l1[seg]) - 0.1 * a) if sgn > 0 else (np.max(h1[seg]) + 0.1 * a)
        risk = abs(entry - stop)
        if risk < 0.15 * a or risk > 3 * a:
            continue

        # target = next opposing draw on liquidity
        cand = [x for x in (highs if sgn > 0 else lows) if (x - entry) * sgn > 0.5 * risk]
        if not cand:
            continue
        tgt = min(cand) if sgn > 0 else max(cand)
        if abs(tgt - entry) / risk > 10:
            tgt = entry + sgn * 10 * risk

        slip = slip_frac * a
        R = None
        for k in range(e_k + 1, min(e_k + 1 + max_bars * 5, n1)):
            if sgn > 0:
                if l1[k] <= stop:
                    R = (stop - slip - entry) / risk
                elif h1[k] >= tgt:
                    R = (tgt - entry) / risk
            else:
                if h1[k] >= stop:
                    R = (entry - stop - slip) / risk
                elif l1[k] <= tgt:
                    R = (entry - tgt) / risk
            if R is not None:
                busy_ts = idx1[k]
                break
        if R is None:
            continue
        rows.append({"ts": idx1[e_k], "side": "long" if sgn > 0 else "short",
                     "R": R, "tgt_R": abs(tgt - entry) / risk,
                     "retraced": r_k is not None})
    return pd.DataFrame(rows)
