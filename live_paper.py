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

import econ_calendar as EC

# Instrument config -- overridable from the command line so the same code
# runs an NQ book and a Dow book side by side in separate windows.
STATE = "live_paper_state.json"
LOG = "live_paper_trades.csv"
SYMBOL = "NQ=F"
POINT = 2.0           # $/point for the MICRO (MNQ). MYM would be 0.50.
START_EQUITY = 10000.0
LABEL = "NQ"

# --- contract-minimum mode -------------------------------------------------
# Percentage sizing is a fiction on a small account: you cannot buy 0.3 of a
# contract. In this mode size is INTEGER contracts (min 1), so the risk taken
# is whatever the stop forces -- which is the real constraint at $100.
# --- realistic commission (IBKR micro futures) -----------------------------
# ~$0.25/side broker + $0.37 CME exchange + $0.02 NFA = ~$1.28 round trip.
# Charged per CONTRACT, so its cost as a fraction of R depends on how many
# contracts the risk budget buys -- which is why frequency is so expensive.
COMMISSION_RT = 1.28

# Stop placement. The sweep extreme is what ICT teaches, but we measured
# ATR stops beating it by ~7 points of win rate (67% vs 60% at a 0.5R
# target) across 4.5 years. Default to the better one.
USE_SPEC = False       # full spec build from the transcripts
USE_MTF = False        # multi-timeframe model from the transcripts
STOP_MODE = "atr"        # "atr" or "sweep"
ATR_STOP_MULT = 1.0

CONTRACT_MODE = False
POINT_VALUE = 0.50        # MYM. MNQ = 2.00, MES = 5.00
MARGIN_PER_CONTRACT = 50.0
MAX_RISK_FRAC = 0.50      # refuse a trade risking more than half the account


def commission_in_R(equity, risk_pts, want_frac):
    """Round-trip commission expressed as a fraction of 1R, so it can be
    subtracted directly from the trade's R outcome."""
    risk_usd = equity * want_frac
    if risk_usd <= 0 or risk_pts <= 0:
        return 0.0
    per_contract = risk_pts * POINT_VALUE
    n = max(1, round(risk_usd / per_contract))
    return (n * COMMISSION_RT) / risk_usd


def contracts_for(equity, risk_pts, want_frac):
    """(contracts, actual_risk_fraction, reason_skipped)"""
    if equity < MARGIN_PER_CONTRACT:
        return 0, 0.0, "below margin"
    per = risk_pts * POINT_VALUE
    if per <= 0:
        return 0, 0.0, "bad stop"
    n = int((equity * want_frac) // per)
    n = max(n, 1)                       # you cannot trade less than 1
    n = min(n, int(equity // MARGIN_PER_CONTRACT))
    if n < 1:
        return 0, 0.0, "cannot post margin"
    frac = n * per / equity
    if frac > MAX_RISK_FRAC:
        return 0, 0.0, f"1 contract = {frac*100:.0f}% risk, too big"
    return n, frac, ""
# Two exit rules run in parallel on IDENTICAL entries/stops. Backtests on
# S&P/Dow/DAX favour 0.5R, but those datasets have been mined four times --
# so we let live forward data arbitrate instead of trusting the backtest.
# Each strategy is an explicit config so we can vary the news filter
# independently of target and size. r05_aggr_NEWS is identical to
# r05_aggressive except it skips entries during scheduled releases -- that
# pairing is the A/B test of whether news protection is worth its cost.
STRATS = {
    "r05_conservative": dict(tmult=0.5, risk=0.010, news=False),
    "r05_aggressive":   dict(tmult=0.5, risk=0.065, news=False),
    "r10_conservative": dict(tmult=1.0, risk=0.010, news=False),
    "r10_aggressive":   dict(tmult=1.0, risk=0.065, news=False),
    "r05_aggr_NEWS":    dict(tmult=0.5, risk=0.065, news=True),
    # zero-lag twin of r05_aggressive: assumes you act the instant the signal
    # bar closes (i.e. a real-time broker feed). The gap between this and
    # r05_aggressive IS the cost of trading off a delayed feed.
    "r05_aggr_NOLAG":   dict(tmult=0.5, risk=0.065, news=False, lag=0),
}
for _c in STRATS.values():
    _c.setdefault("lag", None)          # None -> use global FEED_LAG_MIN
SWING_LB = 2
FVG_WINDOW = 12
COOLDOWN_MIN = 15
STOP_SLIP = 1.5
# Real NY zone, not a fixed offset -- a hardcoded -4 silently breaks when
# DST ends in November and would shift the daily reset by an hour.
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:                     # fallback if tzdata is unavailable
    ET = timezone(timedelta(hours=-4))

# --- risk controls -------------------------------------------------------
# These need no predictive edge to justify: they cap the left tail. Applied
# to BOTH accounts so the sizing comparison stays apples-to-apples.
# --- execution latency -----------------------------------------------------
# Yahoo's futures feed lags ~15 min. The bot therefore learns about a signal
# long after the bar that produced it closed. Without modelling that, fills
# get recorded on bars that had ALREADY closed before the order could exist --
# which turned a real loss into a logged win on 27 Jul. An order is only live
# from (signal bar close + FEED_LAG_MIN).
FEED_LAG_MIN = 15

MAX_LOSSES_PER_DAY = 3      # stop trading after 3 losers in a session
DERISK_DD = 0.20            # halve size once >20% below peak equity
HALT_DD = 0.40              # stop entirely at -40% (account is broken)



# --- news blackout ---------------------------------------------------------
# Scheduled US releases move NQ violently and stops GAP through rather than
# fill at your price. Backtests cannot see this: 5-min bars show a stop-out at
# the stop price, while reality fills you wherever the next tick prints.
# Historical expectancy in these windows is actually slightly ABOVE average,
# so this filter costs a little edge -- it buys protection from a tail that
# can remove 26% of an account at 6.5% risk in one print.
#   08:30 ET  CPI, PPI, NFP, jobless claims, retail sales, GDP
#   10:00 ET  ISM, consumer confidence, JOLTS
#   14:00 ET  FOMC (8x/year -- blocked every day, cheap insurance)
NEWS_BLACKOUT = [(8 * 60 + 25, 8 * 60 + 45),
                 (9 * 60 + 55, 10 * 60 + 15),
                 (13 * 60 + 55, 14 * 60 + 30)]
FLATTEN_BEFORE_NEWS = True     # close open positions entering a blackout


def in_blackout(ts) -> bool:
    m = ts.hour * 60 + ts.minute
    return any(a <= m <= b for a, b in NEWS_BLACKOUT)


def fetch():
    import yfinance as yf
    df = yf.download(SYMBOL, period="30d" if (USE_MTF or USE_SPEC) else "5d", interval="5m",
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
    _tr = pd.concat([df["high"] - df["low"],
                     (df["high"] - df["close"].shift()).abs(),
                     (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = _tr.rolling(14).mean().bfill().values
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
            entry = l[i]
            stop = (entry - ATR_STOP_MULT * atr[i]) if STOP_MODE == "atr" else sweep
            risk = entry - stop
            if risk > max(1e-6, 0.0005 * entry):
                last = (df.index[i], "long", entry, stop, risk)
                armed = 0
        elif armed == -1 and l[i - 2] > h[i]:
            entry = h[i]
            stop = (entry + ATR_STOP_MULT * atr[i]) if STOP_MODE == "atr" else sweep
            risk = stop - entry
            if risk > max(1e-6, 0.0005 * entry):
                last = (df.index[i], "short", entry, stop, risk)
                armed = 0
    return last


def acct_names():
    return list(STRATS)


def load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"accounts": {k: START_EQUITY for k in acct_names()},
            "peak": {k: START_EQUITY for k in acct_names()},
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


# One fixed schema for every row. ENTRY and EXIT rows previously wrote
# different column sets to the same file, so exit rows silently misaligned
# against the header written by the first entry.
LOG_COLS = (["time", "event", "variant", "side", "entry", "stop", "exit",
             "risk_pts", "result", "R", "depth_atr", "rejection", "tod_min"]
            + [f"risk$_{k}" for k in STRATS] + list(STRATS))


def log_row(row):
    """Append one row. If the strategy set has changed since the file was
    created the header would no longer match the rows (adding a 6th strategy
    silently made every new row 2 fields wider than the header, which made
    the log unparseable). Detect that and migrate the file."""
    cols = LOG_COLS
    if os.path.exists(LOG):
        with open(LOG) as f:
            existing = f.readline().strip().split(",")
        if existing != cols:
            try:
                old = pd.read_csv(LOG, header=0, names=existing,
                                  skiprows=1, on_bad_lines="skip")
            except Exception:
                old = pd.DataFrame(columns=existing)
            old.reindex(columns=cols).to_csv(LOG, index=False)
            print(f"  [log schema migrated: {len(existing)} -> {len(cols)} cols]")
    hdr = not os.path.exists(LOG)
    pd.DataFrame([row]).reindex(columns=cols).to_csv(
        LOG, mode="a", header=hdr, index=False)


def _process_bar(state, op, df, bar, bar_ts, blocked, reason):
    """Apply one bar to the open position. Returns True if it closed."""
    hi, lo = float(bar["high"]), float(bar["low"])

    # News-filtered strategies flatten before a scheduled release; the
    # unfiltered ones ride through. That contrast is the experiment.
    if blocked and op.get("filled"):
        ns = [k for k in op["parts"] if STRATS[k]["news"] and not op["done"].get(k)]
        if ns:
            px = float(bar["close"])
            pts = (px - op["entry"]) if op["side"] == "long" else (op["entry"] - px)
            R = pts / op["risk_pts"]
            for k in ns:
                state["accounts"][k] *= (1 + R * STRATS[k]["risk"] * op["scale"].get(k, 1.0))
                state["peak"][k] = max(state["peak"].get(k, START_EQUITY), state["accounts"][k])
                op["done"][k] = True
                log_row({"time": str(bar_ts), "event": "EXIT", "variant": k,
                         "side": op["side"], "entry": round(op["entry"], 2),
                         "exit": round(px, 2), "result": f"news_flat({reason})",
                         "R": round(R, 3),
                         **{a: round(v, 2) for a, v in state["accounts"].items()}})
            print(f"  FLATTENED (news: {reason}) @ {px:,.2f}")

    # A resting limit is not a position until price trades through it,
    # and it can never exit on the bar that filled it.
    # Fill is per-strategy because each has its own latency assumption.
    sig_ts = pd.Timestamp(op["ts"])
    fills = op.setdefault("fills", {})
    for k in op["parts"]:
        if fills.get(k) or op["done"].get(k):
            continue
        lag = STRATS[k].get("lag")
        lag = FEED_LAG_MIN if lag is None else lag
        if bar_ts < sig_ts + pd.Timedelta(minutes=lag):
            continue
        if lo <= op["entry"] <= hi:
            fills[k] = str(bar_ts)
    if not any(fills.get(k) for k in op["parts"]):
        if (bar_ts - sig_ts).total_seconds() > 3600:
            print("  no fill within 1h -- cancelling order")
            state["open"] = None
            save(state)
            return True
        return False
    op["filled"] = True

    for k in list(op["parts"]):
        if op["done"].get(k) or not fills.get(k):
            continue
        if str(bar_ts) == fills[k]:
            continue                      # never exit on the bar that filled you
        tgt = op["targets"][k]
        hit = exit_px = None
        if op["side"] == "long":
            if lo <= op["stop"]:                       # stop checked first
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
        f = op.get("forced", {}).get(k)
        eff = f if f is not None else STRATS[k]["risk"] * op["scale"].get(k, 1.0)
        comm = commission_in_R(state["accounts"][k], op["risk_pts"], eff)
        R_net = R - comm
        state["accounts"][k] *= (1 + R_net * eff)
        state["peak"][k] = max(state["peak"].get(k, START_EQUITY), state["accounts"][k])
        op["done"][k] = True
        if R < 0 and k == "r10_conservative":
            state["losses_today"] = state.get("losses_today", 0) + 1
        log_row({"time": str(bar_ts), "event": "EXIT", "variant": k,
                 "side": op["side"], "entry": round(op["entry"], 2),
                 "exit": round(exit_px, 2), "result": hit, "R": round(R_net, 3),
                 **{a: round(v, 2) for a, v in state["accounts"].items()}})
        print(f"  EXIT[{k}] {hit.upper()} {R:+.2f}R gross, {R_net:+.2f}R net  ({bar_ts:%H:%M})")
    if (bar_ts - sig_ts).total_seconds() > 3600:
        for k in op["parts"]:
            if not fills.get(k):
                op["done"][k] = True      # order expired unfilled: no trade
    if all(op["done"].get(k) for k in op["parts"]):
        state["open"] = None
        save(state)
        return True
    return False


def check(state, df):
    now = df.index[-1]
    op = state.get("open")
    blocked, reason = EC.blackout(now)

    if op:
        # Only looking at df.iloc[-1] used to SKIP bars (5-min polling against
        # a ~15-min-lagged feed). A skipped bar holding a stop-out silently
        # became a win when a later bar reached the target. Replay every bar.
        seen = op.get("last_bar")
        fresh = df[df.index > pd.Timestamp(seen)] if seen else df.iloc[-1:]
        for bar_ts, bar in fresh.iterrows():
            op["last_bar"] = str(bar_ts)
            if _process_bar(state, op, df, bar, bar_ts, blocked, reason):
                return
        save(state)
        return

    today = str(now.date())
    if state.get("day") != today:
        state["day"], state["losses_today"] = today, 0
    if state.get("losses_today", 0) >= MAX_LOSSES_PER_DAY:
        print(f"  [daily loss limit: {state['losses_today']} losers -- done today]")
        return

    if USE_SPEC:
        import spec_live
        sig = spec_live.find_signal_spec(df)
    elif USE_MTF:
        import mtf_live
        sig = mtf_live.find_signal_mtf(df)
    else:
        sig = find_signal(df)
    if not sig:
        return
    ts, side, entry, stop, risk = sig
    key = str(ts)
    if key in state["fired"] or (now - ts).total_seconds() > 3600:
        return

    parts = [k for k in STRATS if not (STRATS[k]["news"] and blocked)]
    scale = {k: risk_scale(state, k) for k in parts}
    parts = [k for k in parts if scale[k] > 0]
    if CONTRACT_MODE:
        keep, forced = [], {}
        for k in parts:
            n, frac, why = contracts_for(state["accounts"][k],
                                         risk, STRATS[k]["risk"] * scale[k])
            if n < 1:
                print(f"     {k}: SKIPPED -- {why}")
                continue
            keep.append(k)
            forced[k] = frac
        parts = keep
        if not parts:
            return
    if not parts:
        print("  [all accounts halted]")
        return

    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    depth, rejection = sweep_features(df, len(df) - 1, stop, side, atr)

    sgn = 1 if side == "long" else -1
    state["fired"] = (state["fired"] + [key])[-200:]
    state["open"] = {"ts": key, "side": side, "entry": float(entry),
                     "stop": float(stop),
                     "targets": {k: float(entry + sgn * STRATS[k]["tmult"] * risk)
                                 for k in parts},
                     "done": {k: False for k in parts}, "parts": parts,
                     "filled": False, "risk_pts": float(risk), "scale": scale,
                     "forced": (forced if CONTRACT_MODE else {}),
                     "last_bar": str(now)}
    log_row({"time": str(now), "event": "ENTRY", "side": side,
             "entry": round(entry, 2), "stop": round(stop, 2),
             "risk_pts": round(risk, 2), "depth_atr": depth,
             "rejection": rejection, "tod_min": now.hour * 60 + now.minute,
             **{f"risk$_{k}": round(state["accounts"][k] * STRATS[k]["risk"] * scale[k], 2)
                for k in parts},
             **{a: round(v, 2) for a, v in state["accounts"].items()}})
    skipped = [k for k in STRATS if k not in parts]
    live_at = (pd.Timestamp(key) + pd.Timedelta(minutes=FEED_LAG_MIN))
    print(f"  ORDER {side.upper()} limit @ {entry:.2f}  stop {stop:.2f}  ({risk:.1f} pts)"
          f"  [working from {live_at:%H:%M}, {FEED_LAG_MIN}m feed lag]")
    if skipped:
        print(f"     sitting out ({reason or 'derisk'}): {', '.join(skipped)}")


def status(state):
    a = state["accounts"]
    start = pd.Timestamp(state["started"])
    days = max((pd.Timestamp.now(tz=ET) - start).days, 1)
    print(f"\n  {'account':<22}{'balance':>12}{'P&L':>12}{'$/day':>10}")
    for k in acct_names():
        pl = a[k] - START_EQUITY
        print(f"  {k:<22}{a[k]:>12,.0f}{pl:>+12,.0f}{pl/days:>10,.0f}")
    print(f"  (day {days}; r05 vs r10 = exit rule | _NEWS vs r05_aggressive = news filter)")


def main(once=False, interval=300):
    while True:
        try:
            df = fetch()
            state = load()
            now_et = datetime.now(ET)
            print(f"[{now_et:%Y-%m-%d %H:%M ET}] {LABEL} bars through {df.index[-1]:%m-%d %H:%M}"
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
    ap.add_argument("--symbol", default=None, help="yahoo symbol, e.g. YM=F")
    ap.add_argument("--label", default=None)
    ap.add_argument("--equity", type=float, default=None)
    ap.add_argument("--state", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--slip", type=float, default=None, help="stop slippage in points")
    ap.add_argument("--contracts", action="store_true",
                    help="integer-contract sizing (real small-account constraint)")
    ap.add_argument("--pv", type=float, default=None, help="$ per point")
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--comm", type=float, default=None, help="round-trip commission $")
    ap.add_argument("--stopmode", default=None, choices=["atr", "sweep"])
    ap.add_argument("--mtf", action="store_true", help="multi-timeframe model")
    ap.add_argument("--spec", action="store_true", help="full transcript spec build")
    a = ap.parse_args()
    if a.symbol:  SYMBOL = a.symbol   # noqa: module-level rebind, top-level scope
    if a.label:   LABEL = a.label
    if a.equity:  START_EQUITY = a.equity
    if a.state:   STATE = a.state
    if a.log:     LOG = a.log
    if a.slip is not None: STOP_SLIP = a.slip
    if a.contracts: CONTRACT_MODE = True
    if a.pv:      POINT_VALUE = a.pv
    if a.margin:  MARGIN_PER_CONTRACT = a.margin
    if a.comm is not None: COMMISSION_RT = a.comm
    if a.stopmode: STOP_MODE = a.stopmode
    if a.mtf: USE_MTF = True
    if a.spec: USE_SPEC = True
    main(once=a.once, interval=a.interval)
