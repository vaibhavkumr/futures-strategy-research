"""WHAT LEVERAGE, GIVEN A STOP LOSS?

A stop loss changes the optimisation. Without one, 74x maximised P($20k) --
but at 74x a 1.35% adverse move is the whole account, and daily 1.35% moves
are routine, so the stop can never actually fire in time. The stop only
protects you at leverage where the account can SURVIVE a normal bad day.

Simulated on DAILY resampled real returns (not weekly) because the stop has
to be enforced daily to mean anything, and a weekly-only check gaps straight
through it.

Two rules, both hard:
    TARGET  touch $20,000 -> flat, stop trading, goal met
    FLOOR   touch the stop level -> flat, stop trading, capital preserved
"""
import numpy as np, pandas as pd
import factor_lab as F

px=F.universe(); wk=px.resample("W-FRI").last()
fwd=wk.pct_change().shift(-1); raw=px.pct_change(252).reindex(wk.index,method="ffill")
COST=10/1e4; K=0.5

# daily return stream of the verified system (weights held, marked daily)
rd=px.pct_change()
weights={}
for t in wk.index[:-1]:
    a=raw.loc[t].dropna()
    if len(a)<8: continue
    z=(a-a.mean())/a.std(ddof=1); w=np.exp(K*z); weights[t]=w/w.sum()
wkeys=sorted(weights)
daily=[]
prev={}
for i,t in enumerate(wkeys[:-1]):
    seg=rd.loc[(rd.index>t)&(rd.index<=wkeys[i+1])]
    if seg.empty: continue
    w=weights[t]
    s=(seg[w.index.intersection(seg.columns)]*w).sum(axis=1).copy()
    # REBALANCE COST, charged on the first day of the holding week.
    # Omitting this was an error: at 30x leverage the cost is magnified 30x
    # too, so it is the difference between a live edge and a dead one.
    turn=sum(abs(w.get(x,0)-prev.get(x,0)) for x in set(w.index)|set(prev))
    s.iloc[0]-=turn*COST
    prev=w.to_dict()
    daily.append(s)
DR=pd.concat(daily).sort_index().values
print(f"daily stream: {len(DR)} days, daily vol {DR.std(ddof=1)*100:.2f}%, "
      f"mean {DR.mean()*1e4:+.1f}bp\n")

rng=np.random.default_rng(41)
def run(lev,floor,days=42,n=60000,block=5):
    nb=max(1,(days+block-1)//block)
    idx=rng.integers(0,len(DR)-block,size=(n,nb))
    p=np.concatenate([DR[idx+j] for j in range(block)],axis=1)[:,:days]*lev
    eq=np.full(n,10000.0); hit=np.zeros(n,bool); stopped=np.zeros(n,bool)
    for d in range(days):
        live=~hit&~stopped
        eq[live]*=(1+np.clip(p[live,d],-0.999,None))
        hit|=live&(eq>=20000)
        eq[hit&(eq>20000)]=20000
        newstop=live&~hit&(eq<=floor)
        stopped|=newstop; eq[newstop]=floor
    return hit,stopped,eq

print("="*80)
print("LEVERAGE x STOP LOSS  -- 2 months (42 trading days)")
print("="*80)
print(f"  {'lev':>5}{'stop at':>10}{'P($20k)':>10}{'P(stopped)':>12}"
      f"{'median':>10}{'mean':>10}{'P(keep>8k)':>12}")
print("  "+"-"*68)
best=None
for lev in (5,10,15,20,30,50):
    for floor in (2000,5000,7000):
        hit,st,eq=run(lev,floor)
        med=np.median(eq); mean=eq.mean()
        row=(hit.mean(),lev,floor,med,mean,st.mean(),(eq>8000).mean())
        if best is None or hit.mean()>best[0]: best=row
        print(f"  {lev:>4}x{floor:>10,}{hit.mean()*100:>9.1f}%{st.mean()*100:>11.1f}%"
              f"{med:>10,.0f}{mean:>10,.0f}{(eq>8000).mean()*100:>11.1f}%")
    print()

h,lev,floor,med,mean,stp,keep=best
print("="*80)
print(f"BEST P($20k) WITH A STOP: {lev}x, stop at ${floor:,}")
print("="*80)
print(f"  P(reach $20,000)     {h*100:>6.1f}%")
print(f"  P(stopped out)       {stp*100:>6.1f}%   -> you keep ${floor:,}")
print(f"  expected value       ${mean:>9,.0f}  (vs $10,000 risked)")
print(f"\n  Without any stop, 74x gave 57.3% but a median of $158.")
print(f"  The stop trades some upside for a floor you actually keep.")

print("\n"+"="*80)
print("COST OF THE STOP -- same leverage, with and without")
print("="*80)
for lev in (15,20,30):
    a,_,eqa=run(lev,1)          # effectively no stop
    b,stb,eqb=run(lev,5000)
    print(f"  {lev:>3}x   no stop: P={a.mean()*100:>5.1f}% median ${np.median(eqa):>7,.0f}"
          f"   |  stop $5k: P={b.mean()*100:>5.1f}% median ${np.median(eqb):>7,.0f}")
print("\n  The stop costs a little probability and removes the zero-dollar tail.")
