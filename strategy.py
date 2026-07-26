"""TJR / ICT-style strategy engine.

This codifies the MECHANICAL parts of TJR's playbook. The pieces:

  1. Killzone filter      - only trade his session windows (ICT killzones).
  2. Market structure     - track swing highs/lows; detect a break of
                            structure (BOS) = trend intent.
  3. Liquidity sweep      - price runs a prior swing (stops) then reverses.
  4. Fair Value Gap (FVG) - the 3-candle imbalance TJR enters on the retrace.
  5. Entry / stop / target- enter on FVG retrace after a sweep+BOS, stop past
                            the sweep, target a fixed R multiple.

What this does NOT capture: the discretionary "does this setup look clean"
judgment TJR applies live. That's the honest limit of automating a manual
strategy — the backtest tells you how much the mechanical core is worth.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


# --- ICT killzones in America/New_York local time (hh, mm) ---
KILLZONES = {
    "london":     ((2, 0), (5, 0)),
    "ny_am":      ((8, 30), (11, 0)),   # the main TJR window
    "ny_pm":      ((13, 30), (16, 0)),
}


def in_killzone(ts: pd.Timestamp, zones=("ny_am",)) -> bool:
    for z in zones:
        (sh, sm), (eh, em) = KILLZONES[z]
        start = ts.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end = ts.replace(hour=eh, minute=em, second=0, microsecond=0)
        if start <= ts <= end:
            return True
    return False


def swing_points(df: pd.DataFrame, lookback: int = 2):
    """Fractal swing highs/lows: a high with `lookback` lower highs on each
    side (and vice-versa). Returns two boolean Series aligned to df.index."""
    highs, lows = df["high"], df["low"]
    is_high = pd.Series(False, index=df.index)
    is_low = pd.Series(False, index=df.index)
    for i in range(lookback, len(df) - lookback):
        window_h = highs.iloc[i - lookback:i + lookback + 1]
        window_l = lows.iloc[i - lookback:i + lookback + 1]
        if highs.iloc[i] == window_h.max():
            is_high.iloc[i] = True
        if lows.iloc[i] == window_l.min():
            is_low.iloc[i] = True
    return is_high, is_low


def find_fvg(df: pd.DataFrame, i: int, direction: str):
    """Bullish FVG: candle[i-2].high < candle[i].low (gap the middle candle
    left). Bearish: candle[i-2].low > candle[i].high. Returns (lo, hi) of the
    gap zone or None. `i` is a positional index."""
    if i < 2:
        return None
    a, c = df.iloc[i - 2], df.iloc[i]
    if direction == "long" and a["high"] < c["low"]:
        return (a["high"], c["low"])
    if direction == "short" and a["low"] > c["high"]:
        return (c["high"], a["low"])
    return None


@dataclass
class Signal:
    ts: pd.Timestamp
    side: str          # "long" / "short"
    entry: float
    stop: float
    target: float
    reason: str


def add_indicators(df: pd.DataFrame, ema_len: int = 50, atr_len: int = 14) -> pd.DataFrame:
    """Attach trend/volatility context columns used by the optional filters.
    All are causal (shifted) so no future information leaks in."""
    out = df.copy()
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - out["close"].shift()).abs(),
        (out["low"] - out["close"].shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(atr_len).mean().shift(1)
    out["ema"] = out["close"].ewm(span=ema_len, adjust=False).mean().shift(1)
    out["atr_pct"] = out["atr"] / out["close"].shift(1)
    # rolling percentile of volatility (regime gate), causal
    out["atr_rank"] = out["atr_pct"].rolling(500).rank(pct=True)
    return out


def generate_signals(df: pd.DataFrame, zones=("ny_am",), rr: float = 2.0,
                     swing_lb: int = 2, fvg_window: int = 12,
                     min_risk_pct: float = 0.0005,
                     trend_filter: bool = False,
                     min_gap_atr: float = 0.0,
                     atr_rank_min: float = 0.0,
                     atr_rank_max: float = 1.0,
                     max_risk_atr: float = 0.0) -> list[Signal]:
    """Walk the candles and emit TJR-style signals.

    Logic per bar (once we're past warmup):
      - must be inside a killzone
      - detect the most recent confirmed swing high & low
      - LONG when: price sweeps below the last swing low (liquidity grab)
        then a bullish FVG forms -> enter at FVG top, stop below sweep low,
        target = entry + rr * risk. SHORT is the mirror.
    """
    is_high, is_low = swing_points(df, swing_lb)
    signals: list[Signal] = []
    last_swing_high = last_swing_low = None
    in_trade_until = None
    # after a sweep we "arm" a direction and hunt for the reversal FVG
    armed_side = None          # "long" / "short"
    armed_sweep_px = None      # the swept extreme (used for the stop)
    armed_expiry = None        # bar index the arming window closes

    for i in range(swing_lb, len(df)):
        ts = df.index[i]
        low_i, high_i = df["low"].iloc[i], df["high"].iloc[i]

        # NO LOOKAHEAD: a swing at bar `c` needs `swing_lb` bars on BOTH sides
        # to confirm, so it only becomes *known* at bar c+swing_lb. At bar i the
        # newest swing we could actually have seen formed at i-swing_lb.
        conf = i - swing_lb
        if conf >= 0:
            if is_high.iloc[conf]:
                last_swing_high = df["high"].iloc[conf]
            if is_low.iloc[conf]:
                last_swing_low = df["low"].iloc[conf]

        # --- detect liquidity sweeps against the last CONFIRMED swing ---
        if last_swing_low is not None and low_i < last_swing_low:
            armed_side, armed_sweep_px, armed_expiry = "long", low_i, i + fvg_window
        elif last_swing_high is not None and high_i > last_swing_high:
            armed_side, armed_sweep_px, armed_expiry = "short", high_i, i + fvg_window

        if armed_expiry is not None and i > armed_expiry:
            armed_side = None  # window closed, no reversal came

        if not in_killzone(ts, zones) or armed_side is None:
            continue
        if in_trade_until is not None and ts <= in_trade_until:
            continue

        # --- look for the reversal FVG that confirms the entry ---
        fvg = find_fvg(df, i, armed_side)
        if not fvg:
            continue

        # ---- optional quality filters (all causal / no lookahead) ----
        atr = df["atr"].iloc[i] if "atr" in df.columns else None
        if trend_filter and "ema" in df.columns:
            ema = df["ema"].iloc[i]
            if pd.notna(ema):
                # only take longs above the trend line, shorts below it
                if armed_side == "long" and df["close"].iloc[i] < ema:
                    continue
                if armed_side == "short" and df["close"].iloc[i] > ema:
                    continue
        if min_gap_atr > 0 and atr and pd.notna(atr) and atr > 0:
            if (fvg[1] - fvg[0]) < min_gap_atr * atr:
                continue  # imbalance too small to be meaningful
        if (atr_rank_min > 0 or atr_rank_max < 1) and "atr_rank" in df.columns:
            r = df["atr_rank"].iloc[i]
            if pd.isna(r) or not (atr_rank_min <= r <= atr_rank_max):
                continue  # wrong volatility regime
        if armed_side == "long":
            entry, stop = fvg[1], armed_sweep_px - 1e-9
            risk = entry - stop
            target = entry + rr * risk
        else:
            entry, stop = fvg[0], armed_sweep_px + 1e-9
            risk = stop - entry
            target = entry - rr * risk
        # Reject degenerate stops: when the sweep price sits right on the FVG
        # edge the risk distance collapses to ~0, which is untradeable (real
        # stops need room) and produces absurd R-multiples. Require the stop to
        # be at least min_risk_pct of price away (0.05% => ~11pts on NQ@22k).
        if risk <= max(1e-6, min_risk_pct * entry):
            continue
        # reject trades whose stop is absurdly wide vs current volatility
        if max_risk_atr > 0 and atr and pd.notna(atr) and atr > 0:
            if risk > max_risk_atr * atr:
                continue
        signals.append(Signal(ts, armed_side, entry, stop, target,
                              f"sweep {'low' if armed_side=='long' else 'high'} + {armed_side} FVG"))
        in_trade_until = ts + pd.Timedelta(hours=2)
        armed_side = None

    return signals
