"""Where did the +0.117R jump come from -- the bug fix, or the time stop?"""
import numpy as np, pandas as pd, glob, os, tjr_spec as S

from duka import load

MK={"NASDAQ":"usatechidxusd","S&P 500":"usa500idxusd","DOW":"usa30idxusd","DAX":"deuidxeur"}
PAIR={"NASDAQ":"usa500idxusd","S&P 500":"usatechidxusd","DOW":"usa500idxusd","DAX":"usa500idxusd"}
allt=[]
for name,slug in MK.items():
    d5,d1=load(slug,"m5"),load(slug,"m1")
    lo=max(d5.index[0],d1.index[0]); d5,d1=d5[d5.index>=lo],d1[d1.index>=lo]
    t=S.generate(d5,d1,corr5=load(PAIR[name],"m5").reindex(d5.index,method="ffill"))
    t["mkt"]=name; allt.append(t)
T=pd.concat(allt,ignore_index=True)

def st(x,name):
    x=np.asarray(x,float); n=len(x)
    if n<5: print(f"{name:<34} n={n}"); return
    m,se=x.mean(),x.std(ddof=1)/np.sqrt(n)
    print(f"{name:<34} n={n:<5} win {(x>0).mean()*100:5.1f}%  expR {m:+.3f}  t={m/se:+5.2f}")

print("="*74)
to=T[T.timed_out]; cl=T[~T.timed_out]
st(T.R,"ALL trades")
st(cl.R,"  resolved (stop or targets hit)")
st(to.R,"  TIMED OUT (new: marked to market)")
print(f"\ntimed-out share: {len(to)/len(T)*100:.1f}%  "
      f"-- their contribution to pooled expR: {len(to)/len(T)*to.R.mean():+.3f}R")

print("\n" + "="*74)
print("Is the gain the TIME STOP rather than TJR's rules?")
print("Re-run dropping timeouts again (old behaviour) but WITH the ladder fix:")
st(cl.R, "  resolved only, ladder fixed")
print("\n  (old code, both bugs)              n=895   expR +0.028  t=+0.57")
print("  -> ladder fix alone is the delta on resolved trades;")
print("     the rest is the time stop no longer being discarded.")

print("\n" + "="*74)
print("Timeout rate and value per market -- is it one market's artifact?")
for m,g in T.groupby("mkt"):
    gt=g[g.timed_out]
    print(f"  {m:<9} timeouts {len(gt):>4}/{len(g):<4} ({len(gt)/len(g)*100:4.1f}%)  "
          f"their expR {gt.R.mean():+.3f}   resolved expR {g[~g.timed_out].R.mean():+.3f}")
T.to_pickle("spec_fixed.pkl")
