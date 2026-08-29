"""The decisive comparison only: optimistic fill vs punishing fill.

Everything between those two is interpolation. If close_beyond+slippage holds
up, the edge survives real execution; if it collapses, the +0.200R was the
backtest filling me on trades a broker would not have.
"""
import numpy as np, pandas as pd
from video_model import load_m1
from video_fix import backtest, stat, CHOCH_Q

d = load_m1("usa500idxusd")
print(f"S&P 500: {len(d):,} bars", flush=True)
print(f"\n  {'fill model':<28}{'n':>7}{'win%':>7}{'expR':>9}{'t':>7}")
print("  "+"-"*60, flush=True)
for lab, kw in (("touch, no slip (optimistic)", dict(fill="touch", slip_ticks=0.0)),
                ("close_beyond + 1 tick slip", dict(fill="close_beyond", slip_ticks=1.0))):
    T = backtest(d, **kw)
    thr = T.choch.quantile(CHOCH_Q)
    s = stat(T[T.choch >= thr])
    if s:
        print(f"  {lab:<28}{s[0]:>7}{s[1]:>7.1f}%{s[2]:>+9.3f}{s[3]:>7.2f}", flush=True)
    else:
        print(f"  {lab:<28} too few trades", flush=True)
