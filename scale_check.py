"""DOES THE ACCOUNT SIZE CHANGE THE ODDS? -- no, and here is the proof.

Returns are multiplicative. A book that doubles $10,000 doubles $200 on the
same path, because every position is a PERCENTAGE of equity. So the doubling
probability should be identical at every size, and the only things that can
break that are frictions which do NOT scale: whole-share minimums (solved by
fractional shares) and per-trade commissions (zero at every major broker now).

Run the same simulation at both sizes to confirm.
"""
import numpy as np, pandas as pd
import factor_lab as F

px=F.universe(); wk=px.resample("W-FRI").last()
raw=px.pct_change(252).reindex(wk.index,method="ffill")
rd=px.pct_change(); COST=10/1e4; K=0.5; PSTOP=0.06
wkeys=[t for t in wk.index[:-1] if raw.loc[t].dropna().shape[0]>=8]

def stream():
    out=[];prev={}
    for i,t in enumerate(wkeys[:-1]):
        a=raw.loc[t].dropna(); z=(a-a.mean())/a.std(ddof=1)
        w=np.exp(K*z); w=w/w.sum()
        seg=px.loc[(px.index>t)&(px.index<=wkeys[i+1])]
        if seg.empty: continue
        cols=[c for c in w.index if c in seg.columns]
        if not cols: continue
        w=w[cols]; entry=seg[cols].iloc[0]; live=pd.Series(True,index=cols)
        rets=[]
        for d in range(len(seg)):
            r=rd.loc[seg.index[d],cols].fillna(0)
            rets.append(float((w*live.astype(float)*r).sum()))
            hit=live&((seg[cols].iloc[d]/entry-1)<=-PSTOP)
            if hit.any(): rets[-1]-=float(w[hit].sum())*COST; live&=~hit
        s=pd.Series(rets,index=seg.index)
        turn=sum(abs(w.get(x,0)-prev.get(x,0)) for x in set(w.index)|set(prev))
        s.iloc[0]-=turn*COST
        prev=w.to_dict(); out.append(s)
    return pd.concat(out).sort_index().values

DR=stream(); rng=np.random.default_rng(7)

def sim(eq0,lev=20,days=42,n=60000,block=5):
    tgt,floor=eq0*2,eq0*0.2          # same RATIOS as $10k->$20k, floor $2k
    nb=max(1,(days+block-1)//block)
    idx=rng.integers(0,len(DR)-block,size=(n,nb))
    p=np.concatenate([DR[idx+j] for j in range(block)],axis=1)[:,:days]*lev
    eq=np.full(n,float(eq0)); hit=np.zeros(n,bool); st=np.zeros(n,bool)
    for d in range(days):
        liv=~hit&~st
        eq[liv]*=(1+np.clip(p[liv,d],-0.999,None))
        hit|=liv&(eq>=tgt); eq[hit&(eq>tgt)]=tgt
        ns=liv&~hit&(eq<=floor); st|=ns; eq[ns]=floor
    return hit.mean(),st.mean(),np.median(eq)

print("="*76)
print("SAME STRATEGY, DIFFERENT ACCOUNT SIZES  (20x, 2 months, double-or-floor)")
print("="*76)
print(f"  {'start':>10}{'target':>10}{'floor':>9}{'P(double)':>12}{'P(floor)':>11}{'median':>11}")
print("  "+"-"*64)
for eq0 in (200,500,1000,5000,10000,100000):
    h,s,m=sim(eq0)
    print(f"  ${eq0:>9,}{eq0*2:>10,}{int(eq0*0.2):>9,}{h*100:>11.1f}%"
          f"{s*100:>10.1f}%{m:>11,.0f}")
print("\n  identical to within simulation noise -- size is irrelevant to the ODDS.")

print("\n"+"="*76)
print("WHAT DOES CHANGE WITH SIZE")
print("="*76)
print(f"  {'start':>9}{'if it doubles':>16}{'profit':>12}{'if floored':>13}{'loss':>11}")
print("  "+"-"*62)
for eq0 in (200,1000,10000):
    print(f"  ${eq0:>8,}{eq0*2:>16,}{eq0:>+12,}{int(eq0*0.2):>13,}{-int(eq0*0.8):>11,}")
print("\n  Only the DOLLARS scale. The probabilities do not.")
