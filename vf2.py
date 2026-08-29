"""Fill realism on one market first, then extend if it survives."""
import sys, numpy as np, pandas as pd
from video_model import load_m1
from video_fix import backtest, stat, CHOCH_Q

mkt = sys.argv[1] if len(sys.argv)>1 else "usa500idxusd"
d = load_m1(mkt)
print(f"{mkt}: {len(d):,} bars\n", flush=True)
print(f"  {'fill model':<16}{'slip':>6}{'n':>7}{'win%':>7}{'expR':>9}{'t':>7}")
print("  "+"-"*54, flush=True)
res={}
for fillm in ("touch","through","next_open","close_beyond"):
    for slip in (0.0,1.0):
        T = backtest(d, fill=fillm, slip_ticks=slip)
        if not len(T): continue
        thr = T.choch.quantile(CHOCH_Q)
        s = stat(T[T.choch>=thr])
        if s:
            res[(fillm,slip)]=s
            print(f"  {fillm:<16}{slip:>6.0f}{s[0]:>7}{s[1]:>7.1f}%{s[2]:>+9.3f}{s[3]:>7.2f}", flush=True)
import json
json.dump({f"{k[0]}_{k[1]}":list(v) for k,v in res.items()}, open(f"fill_{mkt}.json","w"))
