"""Market-context features — 'what conditions make this setup work?'

Not price prediction (nobody has that). Conditioning: given a sweep signal
has fired, does the surrounding market state tell us whether to take it?

Candidates, all causal and free to compute:
  tod_min        minutes since the RTH open (intraday seasonality is real)
  dow            day of week
  gap_atr        overnight gap vs prior close, in ATR
  prior_ret      prior session's return
  prior_range    prior session range / ATR (was yesterday wild?)
  dist_open_atr  distance from today's open, in ATR
  dist_vwap_atr  distance from session VWAP, in ATR
  atr_regime     current ATR vs its 100-bar average (vol expansion/contraction)
  or_position    position within the opening-range (first 30 min) high/low
  trend_align    is the trade with or against the 200-bar trend
  range_used     fraction of a typical daily range already travelled

Same discipline: developed as hypotheses, judged on S&P/Dow/DAX which have
not been mined. A feature ships only if it works across markets.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def session_context(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-bar session context columns. df must be RTH-filtered."""
    out = df.copy()
    day = out.index.normalize()
    g = out.groupby(day)

    out["day_open"] = g["open"].transform("first")
    out["bar_of_day"] = g.cumcount()
    out["tod_min"] = out["bar_of_day"] * 5
    out["dow"] = out.index.dayofweek

    # session VWAP (no volume on index CFDs -> use typical price mean)
    tp = (out["high"] + out["low"] + out["close"]) / 3
    out["vwap"] = tp.groupby(day).expanding().mean().reset_index(level=0, drop=True)

    # opening range = first 6 bars (30 min)
    or_hi = out["high"].where(out["bar_of_day"] < 6).groupby(day).transform("max")
    or_lo = out["low"].where(out["bar_of_day"] < 6).groupby(day).transform("min")
    out["or_hi"], out["or_lo"] = or_hi, or_lo

    # prior session stats
    dclose = out.groupby(day)["close"].last()
    dopen = out.groupby(day)["open"].first()
    dhi = out.groupby(day)["high"].max()
    dlo = out.groupby(day)["low"].min()
    prior_close = dclose.shift(1)
    prior_ret = (dclose / dclose.shift(1) - 1).shift(1)
    prior_range = (dhi - dlo).shift(1)
    dmap = pd.DataFrame({"prior_close": prior_close, "prior_ret": prior_ret,
                         "prior_range": prior_range, "day_hi": dhi, "day_lo": dlo})
    out = out.join(dmap, on=day)

    out["atr_avg"] = out["atr"].rolling(100).mean()
    out["trend_ma"] = out["close"].rolling(200).mean().shift(1)
    return out


def add_context_features(sig_rows: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Attach context at each signal's timestamp."""
    if sig_rows.empty:
        return sig_rows
    # NOTE: .values on a tz-aware series drops the tz and silently reindexes
    # to all-NaN. Use the DatetimeIndex directly.
    c = ctx.reindex(pd.DatetimeIndex(sig_rows["ts"]))
    a = c["atr"].to_numpy()
    f = pd.DataFrame(index=sig_rows.index)
    f["tod_min"] = c["tod_min"].to_numpy()
    f["dow"] = c["dow"].to_numpy()
    f["gap_atr"] = ((c["day_open"] - c["prior_close"]) / a).to_numpy()
    f["prior_ret"] = c["prior_ret"].to_numpy()
    f["prior_range"] = (c["prior_range"] / a).to_numpy()
    f["dist_open_atr"] = ((c["close"] - c["day_open"]) / a).to_numpy()
    f["dist_vwap_atr"] = ((c["close"] - c["vwap"]) / a).to_numpy()
    f["atr_regime"] = (c["atr"] / c["atr_avg"]).to_numpy()
    rng = (c["or_hi"] - c["or_lo"]).replace(0, np.nan)
    f["or_position"] = ((c["close"] - c["or_lo"]) / rng).to_numpy()
    f["range_used"] = ((c["day_hi"] - c["day_lo"]) / a).to_numpy()
    above = (c["close"] > c["trend_ma"]).to_numpy()
    # side: +1 long, -1 short (stored in sig_rows as 'side_dir')
    sd = sig_rows["side_dir"].to_numpy() if "side_dir" in sig_rows else np.zeros(len(f))
    f["trend_align"] = np.where((above & (sd > 0)) | (~above & (sd < 0)), 1.0, 0.0)
    return pd.concat([sig_rows.reset_index(drop=True), f.reset_index(drop=True)], axis=1)
