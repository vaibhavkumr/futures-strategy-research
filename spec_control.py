"""RANDOM-ENTRY CONTROL.

The whole positive result now comes from trades closed by the TIME STOP,
not by TJR's stop/target rules. So: does a random entry, at the same times,
with the same stop distance and the same time exit, do just as well?

If yes, the entries contribute nothing and the "edge" is intraday drift.
"""
import numpy as np, pandas as pd, glob, os, tjr_spec as S

from duka import load

T = pd.read_pickle("spec_fixed.pkl"); T["ts"]=pd.to_datetime(T["ts"])
MK={"NASDAQ":"usatechidxusd","S&P 500":"usa500idxusd","DOW":"usa30idxusd","DAX":"deuidxeur"}

def sim(d1, ts, side, risk_pts, max_hold=240, session_end=960, slip=0.0):
    """Same exit machinery: stop at risk_pts, time/session close, no targets
    (targets are HTF levels; the control isolates entry quality)."""
    i = d1.index.searchsorted(ts)
    if i >= len(d1)-2: return None
    e = float(d1["close"].iloc[i]); sgn = 1 if side=="long" else -1
    stop = e - sgn*risk_pts
    dl = d1.index[i] + pd.Timedelta(minutes=max_hold)
    h,l,c = d1["high"].values, d1["low"].values, d1["close"].values
    last = i
    for k in range(i+1, len(d1)):
        tk = d1.index[k]
        if tk > dl or (tk.hour*60+tk.minute) >= session_end or tk.date()!=d1.index[i].date():
            break
        last = k
        if (l[k] <= stop if sgn>0 else h[k] >= stop):
            return -1.0
    return (c[last]-e)*sgn/risk_pts

rng = np.random.default_rng(4)
print(f"{'variant':<26}{'n':>6}{'win%':>8}{'expR':>9}{'t':>8}")
print("-"*57)

real = T.R.values
m,se = real.mean(), real.std(ddof=1)/np.sqrt(len(real))
print(f"{'TJR spec (actual)':<26}{len(real):>6}{(real>0).mean()*100:>8.1f}{m:>9.3f}{m/se:>8.2f}")

for label, mode in (("RANDOM side, same bars","side"),
                    ("RANDOM time in window","time")):
    pooled=[]
    for name,slug in MK.items():
        d1 = load(slug,"m1"); d5 = load(slug,"m5")
        atr = (d5["high"]-d5["low"]).rolling(14).mean()
        g = T[T.mkt==name]
        for _,r in g.iterrows():
            ts = r.ts
            if mode=="time":
                day = ts.normalize()
                cand = d1[(d1.index>=day+pd.Timedelta(minutes=590)) &
                          (d1.index<=day+pd.Timedelta(minutes=630))]
                if len(cand)==0: continue
                ts = cand.index[rng.integers(len(cand))]
            side = ("long","short")[rng.integers(2)] if mode=="side" else r.side
            j = atr.index.searchsorted(ts)
            if j>=len(atr) or not np.isfinite(atr.iloc[min(j,len(atr)-1)]): continue
            rp = float(r.risk_atr)*float(atr.iloc[min(j,len(atr)-1)])
            if rp<=0: continue
            v = sim(d1, ts, side, rp)
            if v is not None: pooled.append(v)
    p=np.array(pooled); m,se=p.mean(), p.std(ddof=1)/np.sqrt(len(p))
    print(f"{label:<26}{len(p):>6}{(p>0).mean()*100:>8.1f}{m:>9.3f}{m/se:>8.2f}")
