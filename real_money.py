"""$10,000 OF REAL MONEY IN THIS BOT -- what actually lands in the account.

At LEGAL leverage only: 1x cash, or 2x Reg T margin. Bootstrapped in 5-day
blocks on the real daily returns of the verified system (weekly conviction-
tilted momentum, 6% per-position stops, transaction costs), so the clustering
of bad weeks is preserved rather than averaged away.

Reported as a distribution because a single "expected return" hides the thing
that actually decides whether someone keeps running a strategy: the drawdown
they have to sit through, and how often the year simply loses money.
"""
import numpy as np, pandas as pd
import factor_lab as F

px=F.universe(); wk=px.resample("W-FRI").last()
raw=px.pct_change(252).reindex(wk.index,method="ffill")
rd=px.pct_change(); COST=10/1e4; K=0.5; PSTOP=0.06
FIN=0.055      # margin interest on the borrowed portion
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

DR=stream(); rng=np.random.default_rng(13)
print(f"verified system, unlevered: {(1+DR.mean())**252-1:.1%}/yr, "
      f"vol {DR.std(ddof=1)*np.sqrt(252):.1%}, "
      f"Sharpe {DR.mean()/DR.std(ddof=1)*np.sqrt(252):.2f}\n")

def year(lev,eq0=10000,days=252,n=60000,block=5):
    nb=max(1,(days+block-1)//block)
    idx=rng.integers(0,len(DR)-block,size=(n,nb))
    p=np.concatenate([DR[idx+j] for j in range(block)],axis=1)[:,:days]*lev
    p=p-(lev-1)*FIN/252                       # financing drag, daily
    eq=eq0*np.cumprod(1+np.clip(p,-0.99,None),axis=1)
    dd=(eq/np.maximum.accumulate(eq,axis=1)-1).min(axis=1)
    return eq[:,-1],dd

print("="*78)
print("$10,000 AFTER ONE YEAR")
print("="*78)
print(f"  {'':<14}{'p10':>9}{'p25':>9}{'MEDIAN':>10}{'p75':>9}{'p90':>9}"
      f"{'P(loss)':>9}{'typ maxDD':>11}")
print("  "+"-"*72)
for lev,lab in ((1.0,"1x cash"),(2.0,"2x margin")):
    e,dd=year(lev)
    q=np.percentile(e,[10,25,50,75,90])
    print(f"  {lab:<14}{q[0]:>9,.0f}{q[1]:>9,.0f}{q[2]:>10,.0f}{q[3]:>9,.0f}"
          f"{q[4]:>9,.0f}{(e<10000).mean()*100:>8.0f}%{np.median(dd)*100:>10.0f}%")

print("\n"+"="*78)
print("WHAT THAT IS PER MONTH  (median case)")
print("="*78)
for lev,lab in ((1.0,"1x cash"),(2.0,"2x margin")):
    e,dd=year(lev)
    med=np.median(e)-10000
    print(f"  {lab:<14} ${med:>+8,.0f}/yr  =  ${med/12:>+7,.0f}/month"
          f"    worst 10%: ${np.percentile(e,10)-10000:>+8,.0f}")

print("\n"+"="*78)
print("AFTER TAX  -- weekly rebalancing means SHORT-term gains, taxed as income")
print("="*78)
for rate,lab in ((0.22,"22% bracket"),(0.24,"24% bracket")):
    e,_=year(2.0)
    med=np.median(e)-10000
    print(f"  2x, {lab:<14} pre-tax ${med:>+7,.0f}   after tax ${med*(1-rate):>+7,.0f}"
          f"   = ${med*(1-rate)/12:>+6,.0f}/month")

print("\n"+"="*78)
print("HOW LONG TO $10,000/MONTH  (needs ~$1,000,000 at a 12% draw)")
print("="*78)
for lev,lab in ((1.0,"1x"),(2.0,"2x")):
    e,_=year(lev)
    g=np.median(e)/10000-1
    yrs=np.log(100)/np.log(1+g) if g>0 else np.inf
    print(f"  {lab}: median {g*100:.1f}%/yr  ->  $10k reaches $1M in {yrs:.0f} years "
          f"(no additions)")
