"""LIVE PAPER TRADER — the TJR strategy, running for real, on paper.

Runs two accounts side by side on identical signals:
  CONSERVATIVE  1.0% risk/trade  -> what the math says is survivable
  AGGRESSIVE    6.5% risk/trade  -> sized for the $200/day target

Both start at $10,000. Same trades, different size. After a few weeks the
market tells us which sizing was right -- no arguing required.

Data: Yahoo NQ=F 5-minute bars. NOTE these can lag ~15 min, so fills are
approximate. Directionally honest, not tick-accurate.

Run:  python live_paper.py          (loops until stopped)
      python live_paper.py --once   (single check, for scheduling)
"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

STATE = "live_paper_state.json"
LOG = "live_paper_trades.csv"
SYMBOL = "NQ=F"
POINT = 20.0          # $/point for NQ (MNQ would be 2.0)
# Two exit rules run in parallel on IDENTICAL entries/stops. Backtests on
# S&P/Dow/DAX favour 0.5R, but those datasets have been mined four times --
# so we let live forward data arbitrate instead of trusting the backtest.
TARGETS = {"r05": 0.5, "r10": 1.0}
SWING_LB = 2
FVG_WINDOW = 12
COOLDOWN_MIN = 15
STOP_SLIP = 1.5
ACCOUNTS = {"conservative": 0.010, "aggressive": 0.065}
ET = timezone(timedelta(hours=-4))

# --- risk controls -------------------------------------------------------
# These need no predictive edge to justify: they cap the left tail. Applied
# to BOTH accounts so the sizing comparison stays apples-to-apples.
MAX_LOSSES_PER_DAY = 3      # stop trading after 3 losers in a session
DERISK_DD = 0.20            # halve size once >20% below peak equity
HALT_DD = 0.40              # stop entirely at -40% (account is broken)


def fetch():
    import yfinance as yf
    df = yf.download(SYMBOL, period="5d", interval="5m",
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    idx = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    return df.set_index(idx).sort_index()[["open", "high", "low", "close"]].astype(float)


def in_session(ts) -> bool:
    """ICT killzones: London 02-05, NY am 08:30-11, NY pm 13:30-16 (ET)."""
    m = ts.hour * 60 + ts.minute
    return (120 <= m <= 300) or (510 <= m <= 660) or (810 <= m <= 960)


def swings(df, lb):
    w = 2 * lb + 1
    hi = df["high"].rolling(w, center=True).max() == df["high"]
    lo = df["low"].rolling(w, center=True).min() == df["low"]
    return hi.values, lo.values


def find_signal(df):
    """Return the most recent unfired signal, or None. Mirrors strategy.py:
    liquidity sweep -> reversal FVG, no lookahead (swings confirmed late)."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    is_h, is_l = swings(df, SWING_LB)
    n = len(df)
    lsh = lsl = np.nan
    armed, sweep, expiry = 0, np.nan, -1
    last = None
    for i in range(SWING_LB, n):
        conf = i - SWING_LB
        if is_h[conf]:
            lsh = h[conf]
        if is_l[conf]:
            lsl = l[conf]
        if not np.isnan(lsl) and l[i] < lsl:
            armed, sweep, expiry = 1, l[i], i + FVG_WINDOW
        elif not np.isnan(lsh) and h[i] > lsh:
            armed, sweep, expiry = -1, h[i], i + FVG_WINDOW
        if armed != 0 and i > expiry:
            armed = 0
        if armed == 0 or not in_session(df.index[i]):
            continue
        if armed == 1 and h[i - 2] < l[i]:
            entry, stop = l[i], sweep
            risk = entry - stop
            if risk > max(1e-6, 0.0005 * entry):
                last = (df.index[i], "long", entry, stop, risk)
                armed = 0
        elif armed == -1 and l[i - 2] > h[i]:
            entry, stop = h[i], sweep
            risk = stop - entry
            if risk > max(1e-6, 0.0005 * entry):
                last = (df.index[i], "short", entry, stop, risk)
                armed = 0
    return last


def acct_names():
    return [f"{v}_{s}" for v in TARGETS for s in ACCOUNTS]


def load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"accounts": {k: 10000.0 for k in acct_names()},
            "peak": {k: 10000.0 for k in acct_names()},
            "open": None, "fired": [], "day": None, "losses_today": 0,
            "started": datetime.now(ET).isoformat()}


def risk_scale(state, name):
    """Size multiplier for an account given its drawdown. 1.0 = normal."""
    eq = state["accounts"][name]
    peak = max(state.get("peak", {}).get(name, eq), eq)
    dd = 1 - eq / peak
    if dd >= HALT_DD:
        return 0.0
    if dd >= DERISK_DD:
        return 0.5
    return 1.0


def sweep_features(df, i, sweep_px, side, atr):
    """Log-only quality metrics, so we can test them on FRESH forward data
    instead of on the datasets we already mined."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    rng = max(h[i] - l[i], 1e-9)
    if side == "long":
        depth = (sweep_px - l[i]) / atr if atr else np.nan
        rejection = max(sweep_px - l[i], 0.0) / rng
    else:
        depth = (h[i] - sweep_px) / atr if atr else np.nan
        rejection = max(h[i] - sweep_px, 0.0) / rng
    return round(float(depth), 3), round(float(rejection), 3)


def save(s):
    json.dump(s, open(STATE, "w"), indent=2)


def log_row(row):
    hdr = not os.path.exists(LOG)
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=hdr, index=False)


def check(state, df):
    now = df.index[-1]
    px = df["close"].iloc[-1]
    op = state.get("open")

    # --- manage open position (each target rule exits independently) -----
    if op:
        hi, lo = df["high"].iloc[-1], df["low"].iloc[-1]
        for var, mult in TARGETS.items():
            if op["done"].get(var):
                continue
            tgt = op["targets"][var]
            hit = exit_px = None
            if op["side"] == "long":
                if lo <= op["stop"]:
                    hit, exit_px = "stop", op["stop"] - STOP_SLIP
                elif hi >= tgt:
                    hit, exit_px = "target", tgt
            else:
                if hi >= op["stop"]:
                    hit, exit_px = "stop", op["stop"] + STOP_SLIP
                elif lo <= tgt:
                    hit, exit_px = "target", tgt
            if not hit:
                continue
            pts = (exit_px - op["entry"]) if op["side"] == "long" else (op["entry"] - exit_px)
            R = pts / op["risk_pts"]
            for sz, frac in ACCOUNTS.items():
                name = f"{var}_{sz}"
                state["accounts"][name] *= (1 + R * frac * op["scale"].get(name, 1.0))
                state["peak"][name] = max(state["peak"].get(name, 10000.0),
                                          state["accounts"][name])
            op["done"][var] = True
            if R < 0 and var == "r10":       # count losses once, on the 1R rule
                state["losses_today"] = state.get("losses_today", 0) + 1
            log_row({"time": str(now), "event": "EXIT", "variant": var,
                     "side": op["side"], "entry": round(op["entry"], 2),
                     "exit": round(exit_px, 2), "result": hit, "R": round(R, 3),
                     **{k: round(v, 2) for k, v in state["accounts"].items()}})
            print(f"  EXIT[{var}] {hit.upper()} {op['side']} {R:+.2f}R")
        if all(op["done"].get(v) for v in TARGETS):
            state["open"] = None
            a = state["accounts"]
            print(f"    -> 0.5R ${a['r05_conservative']:,.0f} / 1R ${a['r10_conservative']:,.0f} (cons)")
        return

    # --- daily reset + loss limit ---------------------------------------
    today = str(now.date())
    if state.get("day") != today:
        state["day"], state["losses_today"] = today, 0
    if state.get("losses_today", 0) >= MAX_LOSSES_PER_DAY:
        print(f"  [daily loss limit hit: {state['losses_today']} losers -- done for today]")
        return

    # --- look for a new signal ------------------------------------------
    if state.get("open"):
        return
    sig = find_signal(df)
    if not sig:
        return
    ts, side, entry, stop, risk = sig
    key = str(ts)
    if key in state["fired"]:
        return
    if (now - ts).total_seconds() > 60 * 60:
        return                      # stale, don't chase

    scale = {k: risk_scale(state, k) for k in acct_names()}
    if all(v == 0 for v in scale.values()):
        print("  [all accounts halted on drawdown]")
        return

    # ATR for the quality metrics (log only -- not used to filter)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    depth, rejection = sweep_features(df, len(df) - 1, stop, side, atr)

    sgn = 1 if side == "long" else -1
    targets = {v: float(entry + sgn * m * risk) for v, m in TARGETS.items()}
    state["fired"] = (state["fired"] + [key])[-200:]
    state["open"] = {"ts": key, "side": side, "entry": float(entry),
                     "stop": float(stop), "targets": targets,
                     "done": {v: False for v in TARGETS},
                     "risk_pts": float(risk), "scale": scale}
    log_row({"time": str(now), "event": "ENTRY", "side": side,
             "entry": round(entry, 2), "stop": round(stop, 2),
             "tgt_r05": round(targets["r05"], 2), "tgt_r10": round(targets["r10"], 2),
             "risk_pts": round(risk, 2),
             "depth_atr": depth, "rejection": rejection,
             "tod_min": now.hour * 60 + now.minute - 570,
             **{k: round(v, 2) for k, v in state["accounts"].items()}})
    print(f"  ENTRY {side.upper()} @ {entry:.2f}  stop {stop:.2f}  "
          f"tgts {targets['r05']:.2f}/{targets['r10']:.2f}  ({risk:.1f} pts risk)"
          + ("  [DE-RISKED]" if any(v < 1 for v in scale.values()) else ""))


def status(state):
    a = state["accounts"]
    start = pd.Timestamp(state["started"])
    days = max((pd.Timestamp.now(tz=ET) - start).days, 1)
    print(f"\n  {'account':<22}{'balance':>12}{'P&L':>12}{'$/day':>10}")
    for k in acct_names():
        pl = a[k] - 10000
        print(f"  {k:<22}{a[k]:>12,.0f}{pl:>+12,.0f}{pl/days:>10,.0f}")
    print(f"  (day {days}; target $200/day; r05 vs r10 = which exit rule wins)")


def main(once=False, interval=300):
    while True:
        try:
            df = fetch()
            state = load()
            now_et = datetime.now(ET)
            print(f"[{now_et:%Y-%m-%d %H:%M ET}] bars through {df.index[-1]:%m-%d %H:%M}"
                  f"  last {df['close'].iloc[-1]:,.2f}"
                  f"  {'IN SESSION' if in_session(df.index[-1]) else 'outside killzone'}")
            check(state, df)
            save(state)
            status(state)
        except Exception as e:
            print(f"  error: {e}")
        if once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    a = ap.parse_args()
    main(once=a.once, interval=a.interval)
