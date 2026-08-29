"""EXIT STUDY with pre-registered dev/holdout split.

Entries are FIXED (TJR's rules). Only the exit changes. Choose the winner on
DEV markets (Nasdaq + DAX), then take ONE look at HOLDOUT (S&P + Dow).
Picking the best of N on the same data is how you fool yourself; the holdout
is the only number that counts.
"""
import numpy as np, pandas as pd
from duka import load
import tjr_spec as S

MK={"NASDAQ":"usatechidxusd","S&P 500":"usa500idxusd","DOW":"usa30idxusd","DAX":"deuidxeur"}
DEV, HOLD = ("NASDAQ","DAX"), ("S&P 500","DOW")
PAIR={"NASDAQ":"usa500idxusd","S&P 500":"usatechidxusd","DOW":"usa500idxusd","DAX":"usa500idxusd"}
_r=[]
for _n,_s in MK.items():
    _d5,_d1=load(_s,"m5"),load(_s,"m1"); _lo=max(_d5.index[0],_d1.index[0])
    _d5,_d1=_d5[_d5.index>=_lo],_d1[_d1.index>=_lo]
    _t=S.generate(_d5,_d1,corr5=load(PAIR[_n],"m5").reindex(_d5.index,method="ffill"))
    _t["mkt"]=_n; _r.append(_t)
T=pd.concat(_r,ignore_index=True); T["ts"]=pd.to_datetime(T["ts"]); T.to_pickle("spec_fixed.pkl")
BARS={}
for name,slug in MK.items():
    d5,d1=load(slug,"m5"),load(slug,"m1")
    lo=max(d5.index[0],d1.index[0])
    BARS[name]=(d5[d5.index>=lo], d1[d1.index>=lo],
                (d5["high"]-d5["low"]).rolling(14).mean())

def exit_sim(d, i, sgn, risk, tgts_R, scheme, atr, max_hold=240, session_end=960):
    h,l,c=d["high"].values,d["low"].values,d["close"].values
    e=float(c[i]); stop=e-sgn*risk; slip=0.05*atr
    dl=d.index[i]+pd.Timedelta(minutes=max_hold)
    if scheme=="htf":       lad,tg=[0.34,0.33,0.33],[e+sgn*r*risk for r in tgts_R]
    elif scheme=="none":    lad,tg=[],[]
    elif scheme=="r1":      lad,tg=[1.0],[e+sgn*1.0*risk]
    elif scheme=="r2":      lad,tg=[1.0],[e+sgn*2.0*risk]
    elif scheme=="htf_first": lad,tg=[1.0],[e+sgn*tgts_R[0]*risk]
    elif scheme=="half_r1": lad,tg=[0.5],[e+sgn*1.0*risk]   # half off at 1R, rest on time
    rem=1.0;R=0.0;last=i
    for k in range(i+1,len(d)):
        tk=d.index[k]
        if tk>dl or (tk.hour*60+tk.minute)>=session_end or tk.date()!=d.index[i].date(): break
        last=k
        if (l[k]<=stop if sgn>0 else h[k]>=stop):
            return R+rem*((stop-sgn*slip-e)*sgn)/risk
        for ti,t in enumerate(tg):
            if lad[ti]<=0: continue
            if (h[k]>=t if sgn>0 else l[k]<=t):
                R+=lad[ti]*((t-e)*sgn)/risk; rem-=lad[ti]; lad[ti]=0.0; break
        if rem<=1e-9: return R
    return R+rem*((c[last]-e)*sgn)/risk

SCHEMES=["htf","none","r1","r2","htf_first","half_r1"]
def evaluate(markets, scheme):
    out=[]
    for name in markets:
        d5,d1,atr=BARS[name]
        for _,row in T[T.mkt==name].iterrows():
            d = d1 if row.tf=="5m" else d5
            i=d.index.searchsorted(row.ts)
            if i>=len(d)-2: continue
            a=float(row.atr); sgn=1 if row.side=="long" else -1
            out.append(exit_sim(d,i,sgn,float(row.risk_pts),list(row.tgt_R),scheme,a))
    return np.array(out)

# SANITY: the 'htf' scheme must reproduce generate()'s own R exactly.
_chk=evaluate(list(MK), "htf")
_orig=T.R.values
_ok = abs(_chk.mean()-_orig.mean()) < 0.005
print("replay check: htf expR %+.4f  vs generate() %+.4f  -> %s"
      % (_chk.mean(), _orig.mean(), "MATCH" if _ok else "MISMATCH -- replay is wrong"))
print()

print("DEV markets (Nasdaq + DAX) -- choose here")
print(f"{'scheme':<12}{'n':>6}{'win%':>8}{'expR':>9}{'t':>8}")
print("-"*43)
dev={}
for s in SCHEMES:
    x=evaluate(DEV,s); dev[s]=x
    m,se=x.mean(),x.std(ddof=1)/np.sqrt(len(x))
    print(f"{s:<12}{len(x):>6}{(x>0).mean()*100:>8.1f}{m:>9.3f}{m/se:>8.2f}")
best=max(dev,key=lambda s:dev[s].mean())
print(f"\n>>> pre-registered pick from DEV: '{best}'")

print("\nHOLDOUT markets (S&P + Dow) -- single look, all schemes shown for honesty")
print(f"{'scheme':<12}{'n':>6}{'win%':>8}{'expR':>9}{'t':>8}")
print("-"*43)
for s in SCHEMES:
    x=evaluate(HOLD,s)
    m,se=x.mean(),x.std(ddof=1)/np.sqrt(len(x))
    mark=" <== the pick" if s==best else ""
    print(f"{s:<12}{len(x):>6}{(x>0).mean()*100:>8.1f}{m:>9.3f}{m/se:>8.2f}{mark}")
