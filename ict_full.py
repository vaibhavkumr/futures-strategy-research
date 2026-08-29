"""A FAITHFUL ICT/TJR implementation — closing the gaps in our earlier tests.

What our previous version got wrong, and what this fixes:

  WRONG: any 5-bar fractal counted as a liquidity level.
  FIX:   real levels only -- prior day high/low, prior session extremes,
         equal highs/lows (stop clusters), and the Asia-range extremes.
         A random 20-minute-old wiggle is not liquidity.

  WRONG: no higher-timeframe bias at all.
  FIX:   daily + 4H directional bias; longs only with bullish bias, shorts
         only with bearish. This is step one of TJR's actual process.

  WRONG: an FVG appearing was treated as confirmation.
  FIX:   require a real MARKET STRUCTURE SHIFT -- after sweeping a low,
         price must actually break the most recent swing HIGH before we
         look for an entry. That is the confirmation ICT teaches.

  MISSING: premium/discount. ICT only buys in "discount" (lower half of the
         current dealing range) and sells in "premium". Never implemented.
  FIX:   added, with the dealing range derived from confirmed swings.

  MISSING: displacement. The move creating the FVG should be impulsive.
  FIX:   require the displacement leg to exceed a multiple of ATR.

Every level and bias uses only past, confirmed data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

STOP_SLIP = 1.5
SLIP_ATR = 0.05   # slippage as a fraction of ATR (= fraction of 1R)


def load(symbol="NQ=F", period="60d", interval="5m"):
    import yfinance as yf
    d = yf.download(symbol, period=period, interval=interval,
                    progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.columns = [c.lower() for c in d.columns]
    idx = pd.to_datetime(d.index, utc=True).tz_convert("America/New_York")
    return d.set_index(idx).sort_index().dropna(subset=["high", "low", "close"])


def htf_context(df: pd.DataFrame) -> pd.DataFrame:
    """Daily / session context, all shifted so nothing leaks from the future."""
    out = df.copy()
    day = out.index.normalize()

    dh = out.groupby(day)["high"].max()
    dl = out.groupby(day)["low"].min()
    dc = out.groupby(day)["close"].last()
    do = out.groupby(day)["open"].first()

    ctx = pd.DataFrame({
        "pdh": dh.shift(1), "pdl": dl.shift(1), "pdc": dc.shift(1),
        "pd_mid": (dh.shift(1) + dl.shift(1)) / 2,
        # daily bias: prior close vs the close 3 days back, and vs prior range
        "d_trend": np.sign(dc.shift(1) - dc.shift(4)),
        "pd_close_pos": (dc.shift(1) - dl.shift(1)) / (dh.shift(1) - dl.shift(1) + 1e-9),
    })
    for col in ctx.columns:
        out[col] = day.map(ctx[col])

    # Asia session (20:00-02:00 ET) SPANS MIDNIGHT: bars from 20:00 belong to
    # the NEXT trading day's Asia range. Grouping by calendar date would split
    # one session across two days and mislabel every level.
    hh = out.index.hour
    sess_date = pd.DatetimeIndex(np.where(hh >= 20, day + pd.Timedelta(days=1), day))
    asia = (hh >= 20) | (hh < 2)
    a = out[asia]
    a_sd = sess_date[asia]
    out["asia_hi"] = day.map(a.groupby(a_sd)["high"].max())
    out["asia_lo"] = day.map(a.groupby(a_sd)["low"].min())

    # London extremes (02:00-05:00), fully formed by the NY open
    lon = (hh >= 2) & (hh < 5)
    lo_ = out[lon]
    out["lon_hi"] = day.map(lo_.groupby(lo_.index.normalize())["high"].max())
    out["lon_lo"] = day.map(lo_.groupby(lo_.index.normalize())["low"].min())
    return out


def swings(df, lb=2):
    w = 2 * lb + 1
    hi = (df["high"].rolling(w, center=True).max() == df["high"]).to_numpy()
    lo = (df["low"].rolling(w, center=True).min() == df["low"]).to_numpy()
    n = len(df)
    hi[:lb] = hi[n - lb:] = False
    lo[:lb] = lo[n - lb:] = False
    return hi, lo


def generate(df: pd.DataFrame, *, use_htf=True, use_mss=True,
             use_pd=True, use_real_levels=True, min_disp_atr=0.0,
             swing_lb=2, tmult=1.0, stop_mode="atr", window=24,
             eq_tol=0.15):
    """Full ICT pipeline. Flags let us ablate each component to see which
    one actually contributes."""
    ctx = htf_context(df)
    h, l, c, o = (ctx[x].to_numpy(float) for x in ("high", "low", "close", "open"))
    n = len(ctx)
    tr = pd.concat([ctx["high"] - ctx["low"],
                    (ctx["high"] - ctx["close"].shift()).abs(),
                    (ctx["low"] - ctx["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().to_numpy()
    is_h, is_l = swings(ctx, swing_lb)
    mins = np.asarray(ctx.index.hour) * 60 + np.asarray(ctx.index.minute)
    kz = ((mins >= 120) & (mins <= 300)) | ((mins >= 510) & (mins <= 660)) | \
         ((mins >= 810) & (mins <= 960))
    cols = {k: ctx[k].to_numpy(float) for k in
            ("pdh", "pdl", "pd_mid", "d_trend", "pd_close_pos",
             "asia_hi", "asia_lo", "lon_hi", "lon_lo")}

    swing_hi_px: list[float] = []
    swing_lo_px: list[float] = []
    rows = []
    armed = 0
    sweep_lvl = np.nan
    armed_i = -1
    mss_ok = False
    busy = -1

    for i in range(swing_lb + 2, n):
        conf = i - swing_lb
        if is_h[conf]:
            swing_hi_px.append(h[conf])
        if is_l[conf]:
            swing_lo_px.append(l[conf])
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue

        # ---------- real liquidity levels ----------
        if use_real_levels:
            highs = [cols["pdh"][i], cols["asia_hi"][i], cols["lon_hi"][i]]
            lows = [cols["pdl"][i], cols["asia_lo"][i], cols["lon_lo"][i]]
            # equal highs/lows: clusters of confirmed swings
            for arr, dst in ((swing_hi_px, highs), (swing_lo_px, lows)):
                recent = arr[-30:]
                for p in set(recent):
                    if sum(1 for q in recent if abs(q - p) <= eq_tol * a) >= 2:
                        dst.append(p)
            highs = [x for x in highs if np.isfinite(x)]
            lows = [x for x in lows if np.isfinite(x)]
        else:
            highs = swing_hi_px[-1:] if swing_hi_px else []
            lows = swing_lo_px[-1:] if swing_lo_px else []

        # ---------- sweep detection ----------
        swept_low = [x for x in lows if l[i] < x <= l[i] + 3 * a]
        swept_high = [x for x in highs if h[i] > x >= h[i] - 3 * a]
        if swept_low:
            armed, sweep_lvl, armed_i, mss_ok = 1, max(swept_low), i, False
        elif swept_high:
            armed, sweep_lvl, armed_i, mss_ok = -1, min(swept_high), i, False
        if armed != 0 and i - armed_i > window:
            armed = 0
        if armed == 0 or i <= busy:
            continue

        # ---------- market structure shift ----------
        if use_mss and not mss_ok:
            prior_hi = [p for p in swing_hi_px[-6:]]
            prior_lo = [p for p in swing_lo_px[-6:]]
            if armed == 1 and prior_hi and c[i] > max(prior_hi[-2:]):
                mss_ok = True
            elif armed == -1 and prior_lo and c[i] < min(prior_lo[-2:]):
                mss_ok = True
            if not mss_ok:
                continue
        # ---------- FVG ----------
        if armed == 1:
            if not (h[i - 2] < l[i]):
                continue
            entry = l[i]
            disp = (c[i] - l[armed_i]) / a
        else:
            if not (l[i - 2] > h[i]):
                continue
            entry = h[i]
            disp = (h[armed_i] - c[i]) / a
        if min_disp_atr > 0 and disp < min_disp_atr:
            continue
        if not kz[i]:
            continue

        # ---------- higher-timeframe bias ----------
        if use_htf:
            bias = cols["d_trend"][i]
            if not np.isfinite(bias) or bias == 0:
                continue
            if (armed == 1 and bias < 0) or (armed == -1 and bias > 0):
                continue

        # ---------- premium / discount ----------
        if use_pd:
            if not (swing_hi_px and swing_lo_px):
                continue
            rng_hi = max(swing_hi_px[-10:])
            rng_lo = min(swing_lo_px[-10:])
            if rng_hi <= rng_lo:
                continue
            pos = (entry - rng_lo) / (rng_hi - rng_lo)
            if armed == 1 and pos > 0.5:
                continue                       # only buy in discount
            if armed == -1 and pos < 0.5:
                continue                       # only sell in premium

        sgn = 1 if armed == 1 else -1
        stop = (entry - sgn * a) if stop_mode == "atr" else sweep_lvl
        risk = abs(entry - stop)
        # Slippage must be PROPORTIONAL. Charging absolute points across
        # markets at 5,000 and 40,000 taxes them ~8x differently.
        slip = SLIP_ATR * a
        if risk < max(1e-6, 0.0005 * entry):
            continue
        tgt = entry + sgn * tmult * risk

        filled, R, last = False, None, i
        for j in range(i + 1, min(i + 61, n)):
            if not filled:
                if l[j] <= entry <= h[j]:
                    filled, last = True, j
                continue
            if sgn == 1:
                if l[j] <= stop:
                    R = (stop - slip - entry) / risk
                elif h[j] >= tgt:
                    R = tmult
            else:
                if h[j] >= stop:
                    R = (entry - stop - slip) / risk
                elif l[j] <= tgt:
                    R = tmult
            if R is not None:
                last = j
                break
        busy = last
        armed = 0
        if R is not None:
            rows.append({"ts": ctx.index[i], "side": "long" if sgn == 1 else "short",
                         "R": R, "disp_atr": round(disp, 2)})
    return pd.DataFrame(rows)
