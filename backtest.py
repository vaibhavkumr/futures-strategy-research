"""Event-driven backtester for the TJR strategy.

For each signal we simulate a bracket order (entry limit, stop, target) and
walk forward bar-by-bar to see which side gets hit first. Reports the stats
that actually tell you whether this is worth real money: expectancy, win
rate, profit factor, and max drawdown.

NQ contract note: 1 point = $20 per contract (E-mini NQ). Change POINT_VALUE
for MNQ ($2) or ES ($50) / MES ($5).
"""
from __future__ import annotations
import argparse
import pandas as pd
import data as datamod
from strategy import generate_signals, Signal

POINT_VALUE = 20.0    # $ per point per contract (NQ). MNQ=2, ES=50, MES=5
CONTRACTS = 1
COMMISSION = 2.50     # per side, per contract (round-trip = 2x)
# Realism knobs. Stops are MARKET orders on fast reversals -> they slip against
# you. Entry is a resting LIMIT at the FVG -> fills at price or not at all.
STOP_SLIP = 1.5       # points of adverse slippage on a stop-out (NQ, conservative)
ENTRY_SLIP = 0.25     # points of adverse slippage on entry fill


def simulate_trade(df: pd.DataFrame, sig: Signal, max_bars: int = 60):
    """Return (pnl_points, outcome, exit_ts) with realistic assumptions:
      - entry is a resting LIMIT that fills when price trades through it,
        with small adverse slippage (ENTRY_SLIP).
      - stop is a MARKET order -> adverse slippage (STOP_SLIP) added.
      - when a bar's range spans BOTH stop and target, we cannot know the
        intrabar path, so we assume the STOP filled first (worst case). This
        is the single most important honesty fix vs. the fantasy version.
    """
    start = df.index.get_loc(sig.ts)
    filled = False
    if sig.side == "long":
        entry = sig.entry + ENTRY_SLIP
        stop_fill = sig.stop - STOP_SLIP
    else:
        entry = sig.entry - ENTRY_SLIP
        stop_fill = sig.stop + STOP_SLIP

    # NOTE: fills start at start+1. The FVG is only known once the signal bar
    # CLOSES, so its limit order cannot be filled within that same bar --
    # doing so hands the backtest the bar's best price for free (lookahead).
    for j in range(start + 1, min(start + 1 + max_bars, len(df))):
        bar = df.iloc[j]
        if not filled:
            if bar["low"] <= sig.entry <= bar["high"]:
                filled = True
            else:
                continue
        hit_stop = bar["low"] <= sig.stop if sig.side == "long" else bar["high"] >= sig.stop
        hit_tgt = bar["high"] >= sig.target if sig.side == "long" else bar["low"] <= sig.target
        if hit_stop:  # worst-case: stop wins any ambiguous bar
            pnl = (stop_fill - entry) if sig.side == "long" else (entry - stop_fill)
            return pnl, "stop", df.index[j]
        if hit_tgt:
            pnl = (sig.target - entry) if sig.side == "long" else (entry - sig.target)
            return pnl, "target", df.index[j]
    if not filled:
        return None, "no_fill", None
    return None, "timeout", None


def run(df: pd.DataFrame, **kw):
    signals = generate_signals(df, **kw)
    trades = []
    for s in signals:
        pnl_pts, outcome, exit_ts = simulate_trade(df, s)
        if pnl_pts is None:
            continue
        gross = pnl_pts * POINT_VALUE * CONTRACTS
        net = gross - COMMISSION * 2 * CONTRACTS
        trades.append({"ts": s.ts, "side": s.side, "pts": pnl_pts,
                       "net": net, "outcome": outcome})
    return pd.DataFrame(trades)


def report(tr: pd.DataFrame):
    if tr.empty:
        print("No trades taken. Try more data or looser filters.")
        return
    wins = tr[tr.net > 0]
    losses = tr[tr.net <= 0]
    net = tr.net.sum()
    pf = wins.net.sum() / abs(losses.net.sum()) if len(losses) else float("inf")
    equity = tr.net.cumsum()
    dd = (equity.cummax() - equity).max()
    days = tr.ts.dt.normalize().nunique()

    print("=" * 46)
    print(f"  Trades:        {len(tr)}")
    print(f"  Win rate:      {len(wins)/len(tr)*100:5.1f}%")
    print(f"  Net P&L:       ${net:,.0f}")
    print(f"  Avg / trade:   ${tr.net.mean():,.0f}")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Max drawdown:  ${dd:,.0f}")
    print(f"  Trading days:  {days}")
    print(f"  Avg / day:     ${net/days:,.0f}   <- reality check vs your $200 goal")
    print("=" * 46)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="path to OHLCV csv; omit to use synthetic")
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--rr", type=float, default=2.0)
    args = ap.parse_args()

    df = datamod.load_csv(args.csv) if args.csv else datamod.synthetic(days=args.days)
    print(f"Loaded {len(df)} candles: {df.index[0]} -> {df.index[-1]}\n")
    tr = run(df, rr=args.rr)
    report(tr)
