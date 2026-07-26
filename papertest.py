"""Forward paper-test logger — accumulate REAL out-of-sample evidence, free.

Run this on a schedule (e.g. once a day). Each run:
  1. Pulls the latest real NQ 5-min data (free, rolling 60 days from Yahoo).
  2. Runs the strategy and finds every signal.
  3. Simulates each as a paper trade and records the outcome.
  4. Upserts into paper_log.csv keyed by signal time. Resolved outcomes are
     permanent (past bars don't change); newly-resolvable ones fill in.

Because it moves forward in wall-clock time, the trades it logs are ones the
backtest never got to peek at — the closest thing to live proof without money.
The log persists even as old bars roll out of the free window.
"""
from __future__ import annotations
import os
import pandas as pd
import data as datamod
from strategy import generate_signals
from backtest import simulate_trade, POINT_VALUE, CONTRACTS, COMMISSION

LOG = "paper_log.csv"
RR = 2.0


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in generate_signals(df, rr=RR):
        pnl_pts, outcome, exit_ts = simulate_trade(df, s)
        if outcome in ("no_fill", "timeout") or pnl_pts is None:
            status, net = "open", None   # not yet resolved
        else:
            status = "win" if pnl_pts > 0 else "loss"
            net = pnl_pts * POINT_VALUE * CONTRACTS - COMMISSION * 2 * CONTRACTS
        rows.append({
            "signal_ts": s.ts.isoformat(), "side": s.side,
            "entry": round(s.entry, 2), "stop": round(s.stop, 2),
            "target": round(s.target, 2), "status": status,
            "exit_ts": exit_ts.isoformat() if exit_ts is not None else "",
            "net": round(net, 2) if net is not None else "",
            "reason": s.reason})
    return pd.DataFrame(rows)


def merge_log(fresh: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(LOG):
        old = pd.read_csv(LOG)
        combined = pd.concat([old, fresh], ignore_index=True)
    else:
        combined = fresh.copy()
    # keep the most-resolved record per signal (win/loss beats open)
    rank = {"win": 2, "loss": 2, "open": 1}
    combined["_r"] = combined["status"].map(rank).fillna(0)
    combined = (combined.sort_values("_r")
                .drop_duplicates("signal_ts", keep="last")
                .drop(columns="_r")
                .sort_values("signal_ts"))
    return combined


def summarize(log: pd.DataFrame):
    done = log[log.status.isin(["win", "loss"])].copy()
    print("=" * 48)
    print(f"  Logged signals:  {len(log)}   (resolved: {len(done)}, "
          f"open: {(log.status=='open').sum()})")
    if len(done):
        done["net"] = pd.to_numeric(done["net"], errors="coerce")
        wins = (done.status == "win").sum()
        print(f"  Win rate:        {wins/len(done)*100:5.1f}%")
        print(f"  Net P&L:         ${done.net.sum():,.0f}")
        print(f"  Avg / trade:     ${done.net.mean():,.0f}")
        first = pd.to_datetime(done.signal_ts).min().date()
        last = pd.to_datetime(done.signal_ts).max().date()
        print(f"  Span:            {first} -> {last}")
        print("  (let this grow for weeks before trusting it)")
    else:
        print("  No resolved trades yet. Run again after more data arrives.")
    print("=" * 48)


if __name__ == "__main__":
    df = datamod.fetch_yahoo("NQ=F", "5m", "60d").between_time("09:30", "16:00")
    print(f"Pulled NQ 5m: {len(df)} bars up to {df.index[-1]}")
    log = merge_log(evaluate(df))
    log.to_csv(LOG, index=False)
    print(f"Wrote {LOG}\n")
    summarize(log)
