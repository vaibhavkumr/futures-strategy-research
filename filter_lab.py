"""RUN THE FILTER-STACKING METHOD, HONESTLY.

The proposal: look at the trades that lost, work out what they had in common,
add that as a filter, repeat. With enough filters, reach efficiency.

This automates exactly that loop -- and does it far more thoroughly than by
hand, testing every feature at every threshold at each step and greedily
keeping whatever helps most.

The one addition is that after each filter we ALSO measure a holdout period
the search never touched. Both numbers are printed side by side, so the
method gets to show what it actually produces rather than being argued about.

Then the same procedure is run on SHUFFLED outcomes, where the answer is
known in advance to be nothing. Whatever the method finds there is what it
manufactures from noise -- the baseline any real result must beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATS = ["risk_atr", "sweep_depth", "bars_since_sweep", "atr_pct",
         "ema_slope", "fill_delay", "hour", "trend_align"]
CATS = ["sess", "side"]


def expectancy(df):
    return df.R.mean() if len(df) else np.nan


def best_filter(dev, feats, cats, min_keep=0.55):
    """Find the single rule that most improves DEV expectancy while keeping
    at least `min_keep` of the trades (so it cannot just delete everything)."""
    base_n = len(dev)
    best = None
    for f in feats:
        v = dev[f].values
        for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            thr = np.quantile(v, q)
            for op in (">=", "<="):
                keep = (v >= thr) if op == ">=" else (v <= thr)
                if keep.sum() < min_keep * base_n:
                    continue
                e = dev.R.values[keep].mean()
                if best is None or e > best[0]:
                    best = (e, f, op, float(thr))
    for c in cats:
        for val in dev[c].unique():
            keep = (dev[c] != val).values
            if keep.sum() < min_keep * base_n:
                continue
            e = dev.R.values[keep].mean()
            if best is None or e > best[0]:
                best = (e, c, "!=", val)
    return best


def apply_rule(df, rule):
    _, f, op, thr = rule
    if op == ">=":
        return df[df[f] >= thr]
    if op == "<=":
        return df[df[f] <= thr]
    return df[df[f] != thr]


def stack(dev, hold, n_filters=10, label=""):
    rules = []
    d, h = dev.copy(), hold.copy()
    print(f"\n{'step':<5}{'filter':<34}{'DEV n':>7}{'DEV expR':>10}"
          f"{'HOLD n':>8}{'HOLD expR':>11}")
    print("-" * 76)
    print(f"{'0':<5}{'(no filters)':<34}{len(d):>7}{expectancy(d):>+10.3f}"
          f"{len(h):>8}{expectancy(h):>+11.3f}")
    for i in range(1, n_filters + 1):
        r = best_filter(d, FEATS, CATS)
        if r is None:
            break
        rules.append(r)
        d, h = apply_rule(d, r), apply_rule(h, r)
        if len(d) < 200 or len(h) < 50:
            break
        _, f, op, thr = r
        desc = f"{f} {op} {thr:.3f}" if not isinstance(thr, str) else f"{f} != {thr}"
        print(f"{i:<5}{desc:<34}{len(d):>7}{expectancy(d):>+10.3f}"
              f"{len(h):>8}{expectancy(h):>+11.3f}")
    return rules, d, h


if __name__ == "__main__":
    T = pd.read_pickle("live_autopsy.pkl")
    T["ts"] = pd.to_datetime(T["ts"])
    T = T.sort_values("ts")
    dev = T[T.ts < "2025-01-01"].copy()
    hold = T[T.ts >= "2025-01-01"].copy()

    print("=" * 76)
    print("THE METHOD, RUN PROPERLY")
    print("  DEV = 2022-2024 (filters are chosen here, seeing outcomes)")
    print("  HOLDOUT = 2025-2026 (never touched by the search)")
    print("=" * 76)
    print(f"start: DEV {len(dev):,} trades, HOLDOUT {len(hold):,} trades")
    rules, d, h = stack(dev, hold, n_filters=10)

    print("\n" + "=" * 76)
    print("RESULT")
    print("=" * 76)
    d0, h0 = expectancy(dev), expectancy(hold)
    print(f"  DEV     {d0:+.3f}  ->  {expectancy(d):+.3f}   "
          f"(improved {expectancy(d)-d0:+.3f})")
    print(f"  HOLDOUT {h0:+.3f}  ->  {expectancy(h):+.3f}   "
          f"(changed  {expectancy(h)-h0:+.3f})")

    # ---- the control: same procedure, outcomes shuffled -------------------
    print("\n" + "=" * 76)
    print("CONTROL -- identical procedure, but trade outcomes SHUFFLED.")
    print("There is provably NOTHING to find. Whatever it 'finds' here is")
    print("what the method manufactures out of noise.")
    print("=" * 76)
    rng = np.random.default_rng(0)
    devS, holdS = dev.copy(), hold.copy()
    devS["R"] = rng.permutation(devS.R.values)
    holdS["R"] = rng.permutation(holdS.R.values)
    _, ds, hs = stack(devS, holdS, n_filters=10)
    print("\n" + "=" * 76)
    print(f"  SHUFFLED DEV     {expectancy(devS):+.3f}  ->  {expectancy(ds):+.3f}"
          f"   (improved {expectancy(ds)-expectancy(devS):+.3f})")
    print(f"  SHUFFLED HOLDOUT {expectancy(holdS):+.3f}  ->  {expectancy(hs):+.3f}")
    print("=" * 76)
