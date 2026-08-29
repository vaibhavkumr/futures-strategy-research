"""ADD AN INSANE AMOUNT OF FILTERS. Keep going until nothing loses.

The proposal, run to its limit: every time a bad trade shows up, add a rule
so it never happens again. Do not stop at 3 or 6 filters -- stack until the
backtest is clean.

The loop below does that with no mercy: at every step it tests every feature
at every percentile, keeps whatever most improves DEV, and repeats 40 times
with no floor on how many trades may be discarded.

DEV and HOLDOUT are printed side by side at every single step.
"""
import numpy as np, pandas as pd

FEATS = ["risk_atr","sweep_depth","bars_since_sweep","atr_pct","ema_slope",
         "fill_delay","hour","trend_align"]
CATS = ["sess","side","mkt"]

def best(dev, min_keep):
    n0=len(dev); b=None
    for f in FEATS:
        v=dev[f].values
        for q in np.arange(0.05,0.96,0.05):
            thr=np.quantile(v,q)
            for op in (">=","<="):
                k=(v>=thr) if op==">=" else (v<=thr)
                if k.sum()<max(min_keep*n0,20): continue
                e=dev.R.values[k].mean()
                if b is None or e>b[0]: b=(e,f,op,float(thr))
    for c in CATS:
        for val in dev[c].unique():
            k=(dev[c]!=val).values
            if k.sum()<max(min_keep*n0,20): continue
            e=dev.R.values[k].mean()
            if b is None or e>b[0]: b=(e,c,"!=",val)
    return b

def apply(df,r):
    _,f,op,thr=r
    if op==">=": return df[df[f]>=thr]
    if op=="<=": return df[df[f]<=thr]
    return df[df[f]!=thr]

if __name__=="__main__":
    T=pd.read_pickle("live_autopsy.pkl"); T["ts"]=pd.to_datetime(T["ts"])
    T=T.sort_values("ts")
    dev=T[T.ts<"2025-01-01"].copy(); hold=T[T.ts>="2025-01-01"].copy()
    d,h=dev,hold
    print(f"{'#':<4}{'DEV n':>7}{'DEV win%':>10}{'DEV expR':>10}"
          f"{'| HOLD n':>10}{'HOLD win%':>11}{'HOLD expR':>11}")
    print("-"*70)
    print(f"{0:<4}{len(d):>7}{(d.R>0).mean()*100:>10.1f}{d.R.mean():>+10.3f}"
          f"{len(h):>10}{(h.R>0).mean()*100:>11.1f}{h.R.mean():>+11.3f}")
    for i in range(1,41):
        r=best(d, 0.70 if i<6 else 0.85)
        if r is None: break
        d2=apply(d,r)
        if len(d2)<20: break
        h2=apply(h,r)
        d,h=d2,h2
        if len(h)<5:
            print(f"{i:<4}{len(d):>7}{(d.R>0).mean()*100:>10.1f}{d.R.mean():>+10.3f}"
                  f"{len(h):>10}   HOLDOUT EXHAUSTED"); break
        if i<=12 or i%4==0:
            print(f"{i:<4}{len(d):>7}{(d.R>0).mean()*100:>10.1f}{d.R.mean():>+10.3f}"
                  f"{len(h):>10}{(h.R>0).mean()*100:>11.1f}{h.R.mean():>+11.3f}")
    print("-"*70)
    print(f"FINAL  DEV: {len(d)} trades, {(d.R>0).mean()*100:.1f}% win, "
          f"expR {d.R.mean():+.3f}")
    print(f"       HOLDOUT: {len(h)} trades, {(h.R>0).mean()*100:.1f}% win, "
          f"expR {h.R.mean():+.3f}")
