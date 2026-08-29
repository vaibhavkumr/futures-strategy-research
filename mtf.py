"""Multi-timeframe strategy — rebuilt from what the course transcripts describe.

Everything we tested before shared one flaw: it was a SINGLE 5-minute system.
The transcripts weight hourly (225 mentions) and 4-hour (177) far above
5-minute (59), and describe reading four timeframes with three confluences.
So this is not another filter on the old bot -- it is a different structure.

  4H / 1H   directional bias + the liquidity pools that matter
  15m       market structure shift (the confirmation)
  5m        entry timing only

Other gaps this closes:
  ORDER BLOCKS   201 transcript mentions, never implemented. The last
                 opposing candle before the impulse. Offered alongside FVG
                 because the course presents them as alternative entry models.
  LIQUIDITY TGT  exits at the next opposing pool, not a fixed R multiple.
                 Our earlier test used generic swings; this uses real pools
                 (prior day H/L, Asia range, London range, equal highs/lows).
  ALL SESSIONS   London (203) and Asia (180) outrank New York (129) in the
                 transcripts. We had prioritised NY.

Everything is causal: HTF values are shifted so a bar never sees its own
timeframe's unclosed candle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- resampling
def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df.resample(rule).agg({"open": "first", "high": "max",
                               "low": "min", "close": "last"}).dropna()
    return o


def htf_bias(df5: pd.DataFrame) -> pd.DataFrame:
    """4H and 1H directional bias, shifted so only CLOSED candles are used."""
    out = pd.DataFrame(index=df5.index)
    for rule, tag in (("4h", "h4"), ("1h", "h1")):
        r = resample(df5, rule)
        # structure: higher-highs/higher-lows vs lower-highs/lower-lows
        hh = r["high"] > r["high"].shift(1)
        hl = r["low"] > r["low"].shift(1)
        lh = r["high"] < r["high"].shift(1)
        ll = r["low"] < r["low"].shift(1)
        bias = pd.Series(0, index=r.index, dtype=float)
        bias[hh & hl] = 1
        bias[lh & ll] = -1
        bias = bias.replace(0, np.nan).ffill().fillna(0)
        # shift by one full candle so we never use an unclosed one
        out[tag] = bias.shift(1).reindex(df5.index, method="ffill")
    out["bias"] = np.where((out.h4 > 0) & (out.h1 > 0), 1,
                           np.where((out.h4 < 0) & (out.h1 < 0), -1, 0))
    return out


# ------------------------------------------------------------ liquidity pools
def liquidity_pools(df5: pd.DataFrame) -> pd.DataFrame:
    """The levels the course actually names: prior day high/low, Asia range,
    London range, and equal highs/lows. All known before the NY session."""
    out = pd.DataFrame(index=df5.index)
    day = df5.index.normalize()
    hh = df5.index.hour

    d = df5.groupby(day)
    out["pdh"] = day.map(d["high"].max().shift(1))
    out["pdl"] = day.map(d["low"].min().shift(1))

    # Asia 20:00-02:00 ET spans midnight -> assign to the NEXT session date
    sess = pd.DatetimeIndex(np.where(hh >= 20, day + pd.Timedelta(days=1), day))
    a = df5[(hh >= 20) | (hh < 2)]
    asd = sess[(hh >= 20) | (hh < 2)]
    out["asia_h"] = day.map(a.groupby(asd)["high"].max())
    out["asia_l"] = day.map(a.groupby(asd)["low"].min())

    lo = df5[(hh >= 2) & (hh < 5)]
    out["lon_h"] = day.map(lo.groupby(lo.index.normalize())["high"].max())
    out["lon_l"] = day.map(lo.groupby(lo.index.normalize())["low"].min())

    # A session range is not KNOWN until that session has closed. Mapping the
    # value across the whole day let bars at 00:00 read a London high that had
    # not formed yet -- five hours of pure future information. Mask each pool
    # until its session is complete.
    mins = hh * 60 + np.asarray(df5.index.minute)
    for col in ("asia_h", "asia_l"):
        out.loc[mins < 2 * 60, col] = np.nan          # Asia closes 02:00
    for col in ("lon_h", "lon_l"):
        out.loc[mins < 5 * 60, col] = np.nan          # London closes 05:00
    return out


def swings(df: pd.DataFrame, lb: int = 2):
    w = 2 * lb + 1
    hi = (df["high"].rolling(w, center=True).max() == df["high"]).to_numpy()
    lo = (df["low"].rolling(w, center=True).min() == df["low"]).to_numpy()
    n = len(df)
    hi[:lb] = hi[n - lb:] = False
    lo[:lb] = lo[n - lb:] = False
    return hi, lo


# ------------------------------------------------------------------ the model
def generate(df5: pd.DataFrame, *, use_bias=True, use_mss=True,
             entry_mode="both", target_mode="liquidity", tmult=2.0,
             eq_tol_atr=0.20, sweep_window=36, max_bars=120,
             slip_frac=0.05, mss_shift=2):
    """entry_mode: 'fvg' | 'ob' | 'both'
    target_mode: 'liquidity' (next opposing pool) | 'fixed' (tmult x R)"""
    ctx = htf_bias(df5)
    pools = liquidity_pools(df5)
    h, l, c, o = (df5[x].to_numpy(float) for x in ("high", "low", "close", "open"))
    n = len(df5)
    tr = pd.concat([df5["high"] - df5["low"],
                    (df5["high"] - df5["close"].shift()).abs(),
                    (df5["low"] - df5["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().to_numpy()
    bias = ctx["bias"].to_numpy()

    # 15-minute structure, mapped back onto 5-minute bars (shifted = causal)
    r15 = resample(df5, "15min")
    h15, l15 = r15["high"].to_numpy(), r15["low"].to_numpy()
    sh15, sl15 = swings(r15, 1)
    # A centered swing at 15m bar k needs bar k+1, so it is only KNOWN once
    # bar k+1 has closed. shift(1) still leaks that; shift(2) is honest.
    last_sh = pd.Series(np.where(sh15, h15, np.nan), index=r15.index).ffill().shift(mss_shift)
    last_sl = pd.Series(np.where(sl15, l15, np.nan), index=r15.index).ffill().shift(mss_shift)
    sh_on5 = last_sh.reindex(df5.index, method="ffill").to_numpy()
    sl_on5 = last_sl.reindex(df5.index, method="ffill").to_numpy()

    P = {k: pools[k].to_numpy(float) for k in pools.columns}
    is_h5, is_l5 = swings(df5, 2)
    sw_hi: list[float] = []
    sw_lo: list[float] = []

    rows = []
    armed = 0
    swept = np.nan
    armed_i = -1
    mss = False
    busy = -1

    for i in range(30, n):
        conf = i - 2
        if is_h5[conf]:
            sw_hi.append(h[conf])
        if is_l5[conf]:
            sw_lo.append(l[conf])
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue

        # ---- pools available right now ----
        highs = [P[k][i] for k in ("pdh", "asia_h", "lon_h")]
        lows = [P[k][i] for k in ("pdl", "asia_l", "lon_l")]
        for arr, dst in ((sw_hi, highs), (sw_lo, lows)):
            rec = arr[-40:]
            for p in set(rec):
                if sum(1 for q in rec if abs(q - p) <= eq_tol_atr * a) >= 2:
                    dst.append(p)                      # equal highs/lows
        highs = sorted({x for x in highs if np.isfinite(x)})
        lows = sorted({x for x in lows if np.isfinite(x)})

        # ---- sweep of a real pool ----
        hit_lo = [x for x in lows if l[i] < x <= l[i] + 2 * a]
        hit_hi = [x for x in highs if h[i] > x >= h[i] - 2 * a]
        if hit_lo:
            armed, swept, armed_i, mss = 1, max(hit_lo), i, False
        elif hit_hi:
            armed, swept, armed_i, mss = -1, min(hit_hi), i, False
        if armed != 0 and i - armed_i > sweep_window:
            armed = 0
        if armed == 0 or i <= busy:
            continue

        # ---- HTF bias confluence ----
        if use_bias and bias[i] != 0 and bias[i] != armed:
            continue

        # ---- 15m market structure shift ----
        if use_mss and not mss:
            ref = sh_on5[i] if armed == 1 else sl_on5[i]
            if not np.isfinite(ref):
                continue
            if (armed == 1 and c[i] > ref) or (armed == -1 and c[i] < ref):
                mss = True
            else:
                continue

        # ---- entry: FVG or ORDER BLOCK ----
        entry = np.nan
        kind = ""
        if entry_mode in ("fvg", "both"):
            if armed == 1 and h[i - 2] < l[i]:
                entry, kind = l[i], "fvg"
            elif armed == -1 and l[i - 2] > h[i]:
                entry, kind = h[i], "fvg"
        if not np.isfinite(entry) and entry_mode in ("ob", "both"):
            # order block = last opposing candle before the impulse leg
            for k in range(i - 1, max(i - 8, armed_i - 1), -1):
                if armed == 1 and c[k] < o[k]:
                    entry, kind = h[k], "ob"
                    break
                if armed == -1 and c[k] > o[k]:
                    entry, kind = l[k], "ob"
                    break
        if not np.isfinite(entry):
            continue

        sgn = armed
        stop = (l[armed_i] - 0.1 * a) if sgn == 1 else (h[armed_i] + 0.1 * a)
        risk = abs(entry - stop)
        if risk < 0.2 * a or risk > 4 * a:
            continue

        # ---- target: next opposing liquidity pool ----
        if target_mode == "liquidity":
            cand = [x for x in (highs if sgn == 1 else lows)
                    if (x - entry) * sgn > 0.5 * risk]
            if not cand:
                continue
            tgt = min(cand) if sgn == 1 else max(cand)
            if abs(tgt - entry) / risk > 8:
                tgt = entry + sgn * 8 * risk
        else:
            tgt = entry + sgn * tmult * risk

        slip = slip_frac * a
        filled = False
        R = None
        last = i
        for j in range(i + 3, min(i + 3 + max_bars, n)):   # 15-min feed lag
            if not filled:
                if l[j] <= entry <= h[j]:
                    filled, last = True, j
                continue
            if sgn == 1:
                if l[j] <= stop:
                    R = (stop - slip - entry) / risk
                elif h[j] >= tgt:
                    R = (tgt - entry) / risk
            else:
                if h[j] >= stop:
                    R = (entry - stop - slip) / risk
                elif l[j] <= tgt:
                    R = (entry - tgt) / risk
            if R is not None:
                last = j
                break
        busy = last
        armed = 0
        timed_out = False
        if R is None and filled:
            # Discarding unresolved trades is NOT neutral: if slow trades skew
            # to losers, dropping them flatters the result. Mark them to close
            # at market so they can be counted either way.
            cl = c[min(last, n - 1)]
            R = ((cl - entry) if sgn == 1 else (entry - cl)) / risk
            timed_out = True
        if R is not None:
            rows.append({"ts": df5.index[i], "side": "long" if sgn == 1 else "short",
                         "kind": kind, "R": R, "tgt_R": abs(tgt - entry) / risk,
                         "bias": bias[i], "timeout": timed_out,
                         "filled": bool(filled)})
    return pd.DataFrame(rows)
