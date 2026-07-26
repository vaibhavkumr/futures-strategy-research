"""Better liquidity-sweep detection.

Our current rule is crude: "price traded 1 tick past the last swing".
That treats a meaningless poke and a violent stop-run as the same event.
Real sweep quality has structure. This module measures it.

Features (all causal):
  depth_atr      how far past the level, in ATR. A 0.05-ATR poke is noise;
                 a 0.5-ATR run is stops being taken.
  rejection      wick beyond the level / total bar range. High = price was
                 rejected hard = the sweep failed = reversal likely.
  close_back     did the bar CLOSE back inside the level? (ICT calls the
                 close the confirmation; the wick alone is not enough)
  equal_touches  how many prior swings sat at this level (+/- tolerance).
                 Equal highs/lows = stacked stops = the liquidity TJR hunts.
  displacement   size of the move AWAY from the sweep in the next few bars,
                 in ATR. ICT "displacement" -- a real reversal moves fast.
  level_age      bars since the level formed. Older levels hold more stops.
  session_level  was the swept level a session high/low (more significant)
                 rather than a minor intraday swing?

METHODOLOGY NOTE: the Nasdaq dataset has driven hundreds of decisions and is
contaminated. These features are developed as HYPOTHESES there, then tested
on S&P / DAX / Dow, which have seen only one locked config each.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame, swing_lb: int = 2,
                     fvg_window: int = 12, eq_tol_atr: float = 0.15,
                     disp_bars: int = 3) -> pd.DataFrame:
    """Return one row per detected sweep+FVG signal, with quality features
    and the realised outcome (R) so we can measure what actually predicts."""
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    atr = df["atr"].to_numpy(float) if "atr" in df else np.full(len(df), np.nan)
    n = len(df)
    w = 2 * swing_lb + 1
    is_h = (df["high"].rolling(w, center=True).max() == df["high"]).to_numpy()
    is_l = (df["low"].rolling(w, center=True).min() == df["low"]).to_numpy()
    is_h[:swing_lb] = is_h[n - swing_lb:] = False
    is_l[:swing_lb] = is_l[n - swing_lb:] = False

    mins = np.asarray(df.index.hour) * 60 + np.asarray(df.index.minute)
    kz = ((mins >= 510) & (mins <= 660))       # NY am

    swing_highs: list[tuple[int, float]] = []   # (bar_index, price)
    swing_lows: list[tuple[int, float]] = []
    lsh = lsl = np.nan
    lsh_i = lsl_i = -1
    armed = 0
    sweep_px = np.nan
    sweep_i = -1
    expiry = -1
    rows = []

    for i in range(swing_lb, n):
        conf = i - swing_lb
        if is_h[conf]:
            lsh, lsh_i = h[conf], conf
            swing_highs.append((conf, h[conf]))
        if is_l[conf]:
            lsl, lsl_i = l[conf], conf
            swing_lows.append((conf, l[conf]))

        if not np.isnan(lsl) and l[i] < lsl:
            armed, sweep_px, sweep_i, expiry = 1, lsl, i, i + fvg_window
            lvl_i = lsl_i
        elif not np.isnan(lsh) and h[i] > lsh:
            armed, sweep_px, sweep_i, expiry = -1, lsh, i, i + fvg_window
            lvl_i = lsh_i
        if armed != 0 and i > expiry:
            armed = 0
        if armed == 0 or not kz[i]:
            continue

        # FVG confirmation
        if armed == 1:
            if not (h[i - 2] < l[i]):
                continue
            entry, stop = l[i], l[sweep_i]
            risk = entry - stop
            target = entry + risk
        else:
            if not (l[i - 2] > h[i]):
                continue
            entry, stop = h[i], h[sweep_i]
            risk = stop - entry
            target = entry - risk
        a = atr[i]
        if risk <= max(1e-6, 0.0005 * entry) or not np.isfinite(a) or a <= 0:
            continue

        # ---- quality features, measured at the sweep bar ----
        sb = sweep_i
        if armed == 1:
            depth = (sweep_px - l[sb]) / a
            beyond = max(sweep_px - l[sb], 0.0)
            rng = max(h[sb] - l[sb], 1e-9)
            rejection = beyond / rng
            close_back = 1.0 if c[sb] > sweep_px else 0.0
            levels = [p for (_, p) in swing_lows]
        else:
            depth = (h[sb] - sweep_px) / a
            beyond = max(h[sb] - sweep_px, 0.0)
            rng = max(h[sb] - l[sb], 1e-9)
            rejection = beyond / rng
            close_back = 1.0 if c[sb] < sweep_px else 0.0
            levels = [p for (_, p) in swing_highs]
        eq = sum(1 for p in levels if abs(p - sweep_px) <= eq_tol_atr * a)
        disp_end = min(sb + disp_bars, n - 1)
        disp = (abs(c[disp_end] - c[sb]) / a) if disp_end > sb else 0.0
        age = sb - lvl_i

        # ---- realised outcome (conservative intrabar: stop wins ties) ----
        R = np.nan
        filled = False
        fill = entry
        for j in range(i + 1, min(i + 61, n)):
            if not filled:
                if l[j] <= entry <= h[j]:
                    filled = True
                else:
                    continue
            if armed == 1:
                if l[j] <= stop:
                    R = (stop - 1.5 - fill) / risk
                    break
                if h[j] >= target:
                    R = (target - fill) / risk
                    break
            else:
                if h[j] >= stop:
                    R = (fill - stop - 1.5) / risk
                    break
                if l[j] <= target:
                    R = (fill - target) / risk
                    break
        sd_save = armed
        armed = 0
        if np.isfinite(R):
            rows.append({"ts": df.index[i], "side_dir": sd_save, "R": R,
                         "depth_atr": depth, "rejection": rejection,
                         "close_back": close_back, "equal_touches": eq,
                         "displacement": disp, "level_age": age,
                         "risk_atr": risk / a})
    return pd.DataFrame(rows)
