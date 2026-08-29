"""DOES A PER-POSITION STOP HELP OR HURT?

Adding the feature is not the same as it being an improvement. On a momentum
strategy stops frequently HURT: momentum winners pull back routinely, a stop
sells them into the pullback, and you are out when the trend resumes. That is
a well-documented failure mode and it has to be measured, not assumed.

Simulated properly on real daily data:
  - weekly rebalance into conviction-tilted momentum weights
  - each position's entry price recorded at the rebalance
  - if a holding falls `pstop` below entry, it is SOLD and stays in cash
    until the next rebalance (exactly what the bot now does)
  - transaction costs charged on entry, rebalance and stop-outs
  - then the 2-month $10k -> $20k question at 30x, with the account floor

Swept across stop widths so the honest answer is visible either way.
"""
import numpy as np, pandas as pd
import factor_lab as F

px=F.universe(); wk=px.resample("W-FRI").last()
raw=px.pct_change(252).reindex(wk.index,method="ffill")
rd=px.pct_change(); COST=10/1e4; K=0.5
wkeys=[t for t in wk.index[:-1] if raw.loc[t].dropna().shape[0]>=8]

def daily_stream(pstop):
    """Daily returns of the book WITH per-position stops applied."""
    out=[]; prev={}
    for i,t in enumerate(wkeys[:-1]):
        a=raw.loc[t].dropna()
        z=(a-a.mean())/a.std(ddof=1); w=np.exp(K*z); w=w/w.sum()
        seg=px.loc[(px.index>t)&(px.index<=wkeys[i+1])]
        if seg.empty: continue
        cols=[c for c in w.index if c in seg.columns]
        if not cols: continue
        w=w[cols]; entry=seg[cols].iloc[0]
        live=pd.Series(True,index=cols)
        rets=[]
        for d in range(len(seg)):
            r=rd.loc[seg.index[d],cols].fillna(0)
            rets.append(float((w*live.astype(float)*r).sum()))
            if pstop:      # check stops on the close, then exit
                draw=seg[cols].iloc[d]/entry-1
                hit=live&(draw<=-pstop)
                if hit.any():
                    rets[-1]-=float((w[hit]).sum())*COST   # exit cost
                    live&=~hit
        s=pd.Series(rets,index=seg.index)
        turn=sum(abs(w.get(x,0)-prev.get(x,0)) for x in set(w.index)|set(prev))
        s.iloc[0]-=turn*COST
        prev=w.to_dict(); out.append(s)
    return pd.concat(out).sort_index().values

rng=np.random.default_rng(53)
def odds(DR,lev=30,floor=2000,days=42,n=40000,block=5):
    nb=max(1,(days+block-1)//block)
    idx=rng.integers(0,len(DR)-block,size=(n,nb))
    p=np.concatenate([DR[idx+j] for j in range(block)],axis=1)[:,:days]*lev
    eq=np.full(n,10000.0); hit=np.zeros(n,bool); st=np.zeros(n,bool)
    for d in range(days):
        liv=~hit&~st
        eq[liv]*=(1+np.clip(p[liv,d],-0.999,None))
        hit|=liv&(eq>=20000); eq[hit&(eq>20000)]=20000
        ns=liv&~hit&(eq<=floor); st|=ns; eq[ns]=floor
    return hit.mean(),st.mean(),np.median(eq),eq.mean()

print("="*80)
print("PER-POSITION STOP SWEEP  (30x, account floor $2,000, 2 months)")
print("="*80)
print(f"  {'pstop':<10}{'ann ret':>10}{'ann vol':>10}{'Sharpe':>9}"
      f"{'P($20k)':>10}{'P(floor)':>10}{'median':>10}{'EV':>10}")
print("  "+"-"*72)
rows=[]
for ps in (None,0.03,0.04,0.06,0.08,0.10,0.15):
    DR=daily_stream(ps)
    mu=DR.mean()*252*100; sd=DR.std(ddof=1)*np.sqrt(252)*100
    sh=DR.mean()/DR.std(ddof=1)*np.sqrt(252)
    h,s,med,ev=odds(DR)
    lab="none" if ps is None else f"{ps*100:.0f}%"
    rows.append((h,lab,med,ev,sh))
    print(f"  {lab:<10}{mu:>9.2f}%{sd:>9.2f}%{sh:>9.2f}"
          f"{h*100:>9.1f}%{s*100:>9.1f}%{med:>10,.0f}{ev:>10,.0f}")

print("\n"+"="*80)
base=[r for r in rows if r[1]=="none"][0]
best=max(rows,key=lambda r:r[0])
print(f"  no stop:   P($20k) {base[0]*100:.1f}%  median ${base[2]:,.0f}  "
      f"Sharpe {base[4]:.2f}")
print(f"  best stop: {best[1]:<5} P($20k) {best[0]*100:.1f}%  median ${best[2]:,.0f}  "
      f"Sharpe {best[4]:.2f}")
if best[1]=="none":
    print("\n  VERDICT: the per-position stop HURTS. Momentum pullbacks trigger")
    print("  it and you are out when the trend resumes -- the documented failure.")
else:
    print(f"\n  VERDICT: a {best[1]} stop helps; it is the value to run.")
