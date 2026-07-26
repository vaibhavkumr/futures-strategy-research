"""Walk-forward validation — the honesty test for a trading strategy.

Curve-fitting = tuning parameters until the backtest looks great on data you
already saw. It always "works" and always fails live. Walk-forward stops that:

  1. Split history into consecutive chunks.
  2. On each chunk (IN-SAMPLE) pick the best parameters.
  3. Trade the NEXT chunk (OUT-OF-SAMPLE) with those params, unseen.
  4. Stitch the out-of-sample results together. THAT equity curve is the only
     one that resembles live trading.

If the strategy only makes money in-sample and dies out-of-sample, it has no
real edge — better to learn that here for free.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import data as datamod
from backtest import run

RR_GRID = [1.5, 2.0, 2.5, 3.0]
SWING_GRID = [2, 3]


def score(tr: pd.DataFrame) -> float:
    """Rank parameter sets by net P&L, but penalize tiny samples."""
    if tr.empty or len(tr) < 5:
        return -1e9
    return tr.net.sum()


def best_params(df: pd.DataFrame):
    best, best_s = None, -1e18
    for rr in RR_GRID:
        for sw in SWING_GRID:
            s = score(run(df, rr=rr, swing_lb=sw))
            if s > best_s:
                best_s, best = s, (rr, sw)
    return best


def walk_forward(df: pd.DataFrame, n_folds: int = 5):
    bounds = np.linspace(0, len(df), n_folds + 1).astype(int)
    oos_all = []
    print(f"{'fold':>4} {'train':>16} {'test':>16} {'params':>12} {'oos_net':>10} {'trades':>7}")
    for k in range(n_folds - 1):
        tr_df = df.iloc[bounds[k]:bounds[k + 1]]
        te_df = df.iloc[bounds[k + 1]:bounds[k + 2]]
        if len(tr_df) < 50 or len(te_df) < 50:
            continue
        rr, sw = best_params(tr_df)
        oos = run(te_df, rr=rr, swing_lb=sw)
        oos_all.append(oos)
        net = oos.net.sum() if not oos.empty else 0.0
        print(f"{k:>4} {str(tr_df.index[0].date()):>16} {str(te_df.index[0].date()):>16}"
              f" {f'rr{rr},sw{sw}':>12} {net:>10,.0f} {len(oos):>7}")

    allt = pd.concat(oos_all) if oos_all else pd.DataFrame()
    print("\n" + "=" * 52)
    print("  OUT-OF-SAMPLE (stitched) — the only number that matters")
    print("=" * 52)
    if allt.empty:
        print("  No out-of-sample trades.")
        return
    wins = allt[allt.net > 0]
    losses = allt[allt.net <= 0]
    pf = wins.net.sum() / abs(losses.net.sum()) if len(losses) else float("inf")
    eq = allt.net.cumsum()
    dd = (eq.cummax() - eq).max()
    print(f"  OOS trades:    {len(allt)}")
    print(f"  OOS win rate:  {len(wins)/len(allt)*100:5.1f}%")
    print(f"  OOS net P&L:   ${allt.net.sum():,.0f}")
    print(f"  OOS avg/trade: ${allt.net.mean():,.0f}")
    print(f"  Profit factor: {pf:.2f}   (>1.3 = maybe real, <1.0 = no edge)")
    print(f"  Max drawdown:  ${dd:,.0f}")
    print("=" * 52)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h", help="5m (60d) or 1h (2yr)")
    ap.add_argument("--period", default="730d")
    ap.add_argument("--folds", type=int, default=6)
    args = ap.parse_args()
    df = datamod.fetch_yahoo("NQ=F", args.interval, args.period)
    df = df.between_time("09:30", "16:00")
    print(f"Real NQ {args.interval}: {len(df)} bars, "
          f"{df.index[0].date()} -> {df.index[-1].date()}\n")
    walk_forward(df, n_folds=args.folds)
