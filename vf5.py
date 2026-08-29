"""WHAT FILL RATE DOES THIS NEED?

The strategy is positive at a 100% limit fill rate (+0.117R with stop slippage)
and negative when fills require a close beyond the level (-0.077R). Everything
turns on how often a resting limit at the FVG midpoint actually fills.

Real limit behaviour: if price trades THROUGH your price, you are almost
certainly filled. If price merely tags the level and reverses, you are at the
back of the queue and often are not. So the realistic model is penetration-
based: require price to trade N ticks beyond the limit before counting a fill.

  penetration 0 ticks  = filled on any touch          (optimistic, 100%)
  penetration 1 tick   = price must trade through you (realistic)
  penetration 2 ticks  = conservative

This finds the break-even penetration, which is the honest statement of how
good the execution has to be for the strategy to work at all.
"""
import numpy as np, pandas as pd
from video_model import load_m1
from video_fix import backtest, stat, CHOCH_Q
import video_fix as VF

d = load_m1("usa500idxusd")
print(f"S&P 500: {len(d):,} bars\n", flush=True)
print(f"  {'entry requirement':<32}{'n':>7}{'fill%':>7}{'win%':>7}{'expR':>9}{'t':>7}")
print("  "+"-"*70, flush=True)
base=None
for ticks,lab in ((0.0,"touch only (100% fill)"),
                  (0.25,"through by 1 tick"),
                  (0.50,"through by 2 ticks"),
                  (1.00,"through by 4 ticks")):
    # `through` uses `tick` as the penetration distance
    T = backtest(d, fill=("touch" if ticks==0 else "through"),
                 slip_ticks=1.0, tick=(0.25 if ticks==0 else ticks))
    if not len(T): continue
    thr = T.choch.quantile(CHOCH_Q)
    s = stat(T[T.choch>=thr])
    if base is None: base=s[0]
    if s:
        print(f"  {lab:<32}{s[0]:>7}{s[0]/base*100:>6.0f}%{s[1]:>7.1f}%"
              f"{s[2]:>+9.3f}{s[3]:>7.2f}", flush=True)
