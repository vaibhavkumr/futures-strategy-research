"""MATCHED CONTROL -- identical machinery, only the SIDE is randomised.

Same entry bar, same stop distance, same target ladder, same time stop and
session close. The ONLY thing changed is long vs short. Anything left is
directional edge; anything that survives a coin flip is not.
"""
import numpy as np, pandas as pd
from duka import load
import tjr_spec as S

MK={"NASDAQ":"usatechidxusd","S&P 500":"usa500idxusd","DOW":"usa30idxusd","DAX":"deuidxeur"}
PAIR={"NASDAQ":"usa500idxusd","S&P 500":"usatechidxusd","DOW":"usa500idxusd","DAX":"usa500idxusd"}

def run_exit(d, i, sgn, risk, tgts_R, partials=(0.34,0.33,0.33),
             max_hold=240, session_end=960, slip_frac=0.05, atr=1.0):
    h,l,c = d["high"].values, d["low"].values, d["close"].values
    e = float(c[i]); stop = e - sgn*risk
    tg = [e + sgn*r*risk for r in tgts_R]
    dl = d.index[i] + pd.Timedelta(minutes=max_hold)
    lad=list(partials); rem=1.0; R=0.0; last=i; slip=slip_frac*atr
    for k in range(i+1, len(d)):
        tk=d.index[k]
        if tk>dl or (tk.hour*60+tk.minute)>=session_end or tk.date()!=d.index[i].date(): break
        last=k
        if (l[k]<=stop if sgn>0 else h[k]>=stop):
            return R + rem*((stop - sgn*slip - e)*sgn)/risk
        for ti,t in enumerate(tg):
            if lad[ti]<=0: continue
            if (h[k]>=t if sgn>0 else l[k]<=t):
                R += lad[ti]*((t-e)*sgn)/risk; rem-=lad[ti]; lad[ti]=0.0
                break
        if rem<=1e-9: return R
    return R + rem*((c[last]-e)*sgn)/risk

rows=[]
for name,slug in MK.items():
    d5,d1 = load(slug,"m5"), load(slug,"m1")
    lo=max(d5.index[0],d1.index[0]); d5,d1=d5[d5.index>=lo],d1[d1.index>=lo]
    t=S.generate(d5,d1,corr5=load(PAIR[name],"m5").reindex(d5.index,method="ffill"))
    t["mkt"]=name; rows.append(t)
T=pd.concat(rows,ignore_index=True); T["ts"]=pd.to_datetime(T["ts"])
T.to_pickle("spec_fixed.pkl")

rng=np.random.default_rng(20)
print(f"{'variant':<28}{'n':>6}{'win%':>8}{'expR':>9}{'t':>8}")
print("-"*59)
r=T.R.values; m,se=r.mean(),r.std(ddof=1)/np.sqrt(len(r))
print(f"{'TJR spec (actual)':<28}{len(r):>6}{(r>0).mean()*100:>8.1f}{m:>9.3f}{m/se:>8.2f}")

means=[]
for trial in range(10):
    out=[]
    for name,slug in MK.items():
        d5,d1=load(slug,"m5"),load(slug,"m1")
        lo=max(d5.index[0],d1.index[0]); d5,d1=d5[d5.index>=lo],d1[d1.index>=lo]
        atr=(d5["high"]-d5["low"]).rolling(14).mean()
        g=T[T.mkt==name]
        for _,row in g.iterrows():
            tf=row.tf; d = d1 if tf=="5m" else d5
            i=d.index.searchsorted(row.ts)
            if i>=len(d)-2: continue
            j=min(atr.index.searchsorted(row.ts), len(atr)-1)
            a=float(atr.iloc[j])
            if not np.isfinite(a) or a<=0: continue
            risk=float(row.risk_atr)*a
            sgn = 1 if rng.integers(2) else -1          # <-- ONLY change
            out.append(run_exit(d,i,sgn,risk,list(row.tgt_R),atr=a))
    o=np.array(out); means.append(o.mean())
    if trial<3:
        mm,ss=o.mean(),o.std(ddof=1)/np.sqrt(len(o))
        print(f"{'  random side trial '+str(trial+1):<28}{len(o):>6}{(o>0).mean()*100:>8.1f}{mm:>9.3f}{mm/ss:>8.2f}")
means=np.array(means)
print(f"\n10 coin-flip trials: mean expR {means.mean():+.3f}  best {means.max():+.3f}  worst {means.min():+.3f}")
print(f"real strategy       : {r.mean():+.3f}")
print("\nVERDICT:", "beats every coin-flip trial" if r.mean()>means.max()
      else f"does NOT beat the coin flip ({(means>=r.mean()).sum()}/10 trials matched or beat it)")
