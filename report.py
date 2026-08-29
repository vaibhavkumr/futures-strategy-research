"""Show the archived run and the live run in the same format as the bot.

The 27 Jul run is kept for comparison. It is not a performance record -- it
is what the numbers look like when execution is modelled optimistically
(no feed latency, fills assumed on the bar after the signal). The gap
between it and the current run is the cost of being honest about execution.

    python report.py
"""
from __future__ import annotations
import numpy as np
import json
import os
import pandas as pd

import live_paper as L

# (state file, log file, title, starting equity)
RUNS = [
    ("live_paper_state.json", "live_paper_trades.csv",
     "TJR BOOK  --  sweep+FVG, MNQ, $10,000 (reset 30 Jul)", 10000.0),
    ("mtf_state.json", "mtf_trades.csv",
     "MTF BOOK  --  multi-timeframe model, $10,000 (reset 30 Jul)", 10000.0),
    ("spec_state.json", "spec_trades.csv",
     "SPEC BOOK  --  transcript spec build, $10,000 (reset 30 Jul)", 10000.0),
    ("live_paper_state_PRE_LATENCY_FIX.json",
     "live_paper_trades_PRE_LATENCY_FIX.csv",
     "ARCHIVED 27 Jul  --  optimistic execution, no feed lag (reference only)", 10000.0),
]


def block(state_file, log_file, title, start_eq=10000.0):
    if not os.path.exists(state_file):
        return
    st = json.load(open(state_file))
    acc = st["accounts"]
    start = pd.Timestamp(st.get("started", pd.Timestamp.now(tz=L.ET)))
    if start.tzinfo is None:
        start = start.tz_localize(L.ET)
    now = pd.Timestamp.now(tz=L.ET)
    # $/day needs a denominator of days the market was actually OPEN. The old
    # code used (now - start).days, which floors ELAPSED 24h blocks: a book
    # live from Mon 15:19 to Thu 10:53 reported "2" despite covering four
    # calendar days and three sessions. Count weekdays inclusive instead.
    days = max(int(np.busday_count(start.date(), now.date())) + 1, 1)

    last_bar = trades = ""
    if os.path.exists(log_file):
        try:
            t = pd.read_csv(log_file)
            if len(t):
                last_bar = pd.Timestamp(t["time"].iloc[-1]).strftime("%m-%d %H:%M")
                ex = t[t.event == "EXIT"]
                r = pd.to_numeric(ex["R"], errors="coerce").dropna()
                if len(r):
                    # All six strategies exit the SAME trade, so counting exit
                    # rows ("legs") inflates the apparent sample 6x. The real
                    # sample size is the number of DISTINCT trades.
                    ex2 = ex.dropna(subset=["entry"]).copy()
                    ex2["Rn"] = pd.to_numeric(ex2["R"], errors="coerce")
                    per_trade = ex2.groupby("entry")["Rn"].mean().dropna()
                    n_tr = len(per_trade)
                    won = (per_trade > 0).mean() * 100 if n_tr else 0
                    trades = (f"  {len(t[t.event=='ENTRY'])} signals -> "
                              f"{n_tr} TRADES ({len(r)} exit legs), "
                              f"{won:.0f}% of trades won, mean {r.mean():+.2f}R")
                    if n_tr and won == 100:
                        trades += (f"\n  NOTE: {n_tr} trades is a tiny sample. "
                                   f"P(all win by chance) ~ {0.5**n_tr*100:.0f}% "
                                   f"at the 1R rule.")
        except Exception:
            pass

    print(f"[{title}]"
          + (f" last activity {last_bar}" if last_bar else " no trades yet"))
    print()
    print(f"  {'account':<26}{'balance':>10}{'P&L':>12}{'%':>10}{'$/day':>10}")
    for k in acc:
        pl = acc[k] - start_eq
        print(f"  {k:<26}{acc[k]:>10,.0f}{pl:>+12,.0f}{pl/start_eq*100:>+9.1f}%{pl/days:>10,.0f}")
    if trades:
        print(trades)
    print()


if __name__ == "__main__":
    for sf, lf, title, eq in RUNS:
        block(sf, lf, title, eq)
    print("  r05_aggressive vs r05_aggr_NOLAG = cost of the 15-min data delay")
    print("  r05_aggressive vs r05_aggr_NEWS  = cost/benefit of news protection")
    print("  r05_* vs r10_*                   = which exit target wins")
    print("  *_conservative vs *_aggressive   = does 6.5% sizing survive")
