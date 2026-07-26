"""THE ONE LOOK at the holdout set.

Config was chosen from DEV only, before this file was ever run:
    rr=1.0, swing_lb=2, fvg_window=12, zones=ny_am, no extra filters.
Chosen for robustness (centre of a broad dev plateau), not peak dev score.

Also runs two sanity checks a real edge must survive:
  - random-entry control: same trade times, coin-flip direction. If our
    signal scores no better than this, the "edge" is just market drift.
  - per-year stability.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import data as datamod
from strategy import add_indicators
import fastcore as fc

CSV = "download/usatechidxusd-m5-bid-2022-01-01-2026-07-24.csv"
BEST = dict(rr=1.0, swing_lb=2, fvg_window=12)
DEV_END = "2025-01-01"


def run(df, label, **p):
    P = fc.prep(df, p.get("swing_lb", 2))
    R = fc.simulate_fast(P, fc.signals_fast(P, **p))
    s = fc.stats(R)
    print(f"{label:22} n={s['n']:4}  win={s['win']:5.1f}%  expR={s['expR']:+.3f}  "
          f"PF={s['pf']:.2f}  ddR={s['ddR']:5.1f}  t={s['t']:.2f}  totR={s['totR']:+.1f}")
    return R


def control(df, seed=0, **p):
    """Same signals/timing, but the SIDE is randomized. Isolates whether the
    setup logic adds anything beyond being in the market at those moments."""
    P = fc.prep(df, p.get("swing_lb", 2))
    idx, side, entry, stop, tgt = fc.signals_fast(P, **p)
    rng = np.random.default_rng(seed)
    flip = rng.choice([1, -1], len(side))
    new_side = side * flip
    # rebuild stop/target on the flipped side, same risk distance
    risk = np.abs(entry - stop)
    new_stop = np.where(new_side == 1, entry - risk, entry + risk)
    new_tgt = np.where(new_side == 1, entry + p.get("rr", 1.0) * risk,
                       entry - p.get("rr", 1.0) * risk)
    return fc.simulate_fast(P, (idx, new_side, entry, new_stop, new_tgt))


if __name__ == "__main__":
    df = add_indicators(datamod.load_csv(CSV)).between_time("09:30", "16:00")
    dev = df[df.index < DEV_END]
    hold = df[df.index >= DEV_END]

    print("=" * 78)
    print("CONFIG:", BEST)
    print("=" * 78)
    run(dev, "DEV (searched)", **BEST)
    Rh = run(hold, "HOLDOUT (unseen)", **BEST)

    print("\n--- random-side control on holdout (10 seeds) ---")
    ctrl = [fc.stats(control(hold, seed=s, **BEST)) for s in range(10)]
    ce = np.array([c["expR"] for c in ctrl])
    print(f"control expR: mean {ce.mean():+.3f}  range [{ce.min():+.3f}, {ce.max():+.3f}]")
    print(f"our signal:   {fc.stats(Rh)['expR']:+.3f}   "
          f"-> beats {(fc.stats(Rh)['expR'] > ce).sum()}/10 random controls")

    print("\n--- per-year (full period) ---")
    for yr in range(2022, 2027):
        sl = df[df.index.year == yr]
        if len(sl) > 2000:
            run(sl, f"  {yr}", **BEST)
