"""FIXING THE FILL PROBLEM -- pay a worse price to stop missing the winners.

Diagnosis: the limit-at-midpoint approach lost 387 of 2,584 setups (15%) and
they were disproportionately winners -- the sharp rejections that touch the
level and rip away. Win rate fell 26.7% -> 21.9% and expectancy went negative.

So the fix is not a better limit. It is to stop using a limit:

  touch          limit at midpoint, filled if touched          (optimistic)
  next_open      MARKET order: touched -> fill at next bar open  (realistic)
  touch_close    MARKET order: fill at the touch bar's close     (realistic)
  close_beyond   wait for a close beyond, fill next open      (what just failed)

A market order gets EVERY setup, at a worse price. The question is whether
capturing the missing 15% outweighs paying up on all of them. Slippage of one
tick is charged on stops throughout.
"""
import numpy as np, pandas as pd
from video_model import load_m1
from video_fix import backtest, stat, CHOCH_Q

d = load_m1("usa500idxusd")
print(f"S&P 500: {len(d):,} bars\n", flush=True)
print(f"  {'fill model':<30}{'n':>7}{'win%':>7}{'expR':>9}{'t':>7}")
print("  "+"-"*62, flush=True)
for lab, kw in (
    ("limit @ mid (optimistic)",   dict(fill="touch",        slip_ticks=0.0)),
    ("limit @ mid + slip",         dict(fill="touch",        slip_ticks=1.0)),
    ("MARKET @ next open",         dict(fill="next_open",    slip_ticks=1.0)),
    ("wait for close beyond",      dict(fill="close_beyond", slip_ticks=1.0)),
):
    T = backtest(d, **kw)
    if not len(T):
        print(f"  {lab:<30} no trades", flush=True); continue
    thr = T.choch.quantile(CHOCH_Q)
    s = stat(T[T.choch>=thr])
    if s: print(f"  {lab:<30}{s[0]:>7}{s[1]:>7.1f}%{s[2]:>+9.3f}{s[3]:>7.2f}", flush=True)
