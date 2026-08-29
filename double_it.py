"""MAXIMISE P($10,000 -> $20,000 IN 1-2 MONTHS).

This is a threshold-reaching problem, not a growth problem, and the two have
DIFFERENT optimal answers. Kelly maximises long-run growth. When you have a
deadline and a target, the optimum shifts toward more variance -- you do not
care about the median, only about touching $20k before the deadline.

So: sweep leverage on the verified system and, for each, measure by bootstrap
on the REAL weekly returns (4-week blocks, preserving the clustering of bad
weeks):

    P(touch $20,000 within 4 / 8 weeks)
    P(fall below $5,000)  -- half the account gone
    P(fall below $2,000)  -- effectively over
    median and 10th percentile ending equity

Everything reported at both horizons. No configuration is recommended; the
numbers are what they are and the risk decision is the account owner's.
"""
import numpy as np, pandas as pd
import factor_lab as F

px=F.universe(); wk=px.resample("W-FRI").last()
fwd=wk.pct_change().shift(-1); raw=px.pct_change(252).reindex(wk.index,method="ffill")
COST=10/1e4; K=0.5

def weekly(lev):
    out=[];prev={}
    for t in wk.index[:-1]:
        a=raw.loc[t].dropna()
        if len(a)<8: continue
        z=(a-a.mean())/a.std(ddof=1)
        w=np.exp(K*z); w=w/w.sum()*lev
        f=fwd.loc[t].reindex(w.index).fillna(0)
        turn=sum(abs(w.get(x,0)-prev.get(x,0)) for x in set(w.index)|set(prev))
        out.append(float((w*f).sum())-turn*COST); prev=w.to_dict()
    return np.array(out)

r1=weekly(1.0)          # 1x return stream; leverage scales it
print(f"verified system: {len(r1)} weeks, weekly vol {r1.std(ddof=1)*100:.2f}%, "
      f"mean {r1.mean()*100:+.3f}%\n")

rng=np.random.default_rng(31)
def paths(lev,weeks,n=60000,block=4):
    nb=max(1,weeks//block)
    idx=rng.integers(0,len(r1)-block,size=(n,nb))
    p=np.concatenate([r1[idx+j] for j in range(block)],axis=1)[:,:weeks]*lev
    eq=10000*np.cumprod(1+np.clip(p,-0.99,None),axis=1)
    return eq

print("="*80)
print("P(REACHING $20,000) -- verified edge, leverage swept")
print("="*80)
for weeks,lab in ((4,"1 MONTH"),(8,"2 MONTHS")):
    print(f"\n  --- {lab} ({weeks} weeks) ---")
    print(f"  {'lev':>5}{'P($20k)':>10}{'P(<$5k)':>10}{'P(<$2k)':>10}"
          f"{'median':>10}{'p10':>9}{'p90':>10}")
    print("  "+"-"*62)
    for lev in (1,2,3,5,8,12,20,35,60):
        eq=paths(lev,weeks)
        hit=(eq>=20000).any(axis=1).mean()*100
        low=eq.min(axis=1)
        print(f"  {lev:>4}x{hit:>9.1f}%{(low<5000).mean()*100:>9.1f}%"
              f"{(low<2000).mean()*100:>9.1f}%{np.median(eq[:,-1]):>10,.0f}"
              f"{np.percentile(eq[:,-1],10):>9,.0f}{np.percentile(eq[:,-1],90):>10,.0f}")

print("\n"+"="*80)
print("THE OPTIMUM, AND WHAT IT COSTS")
print("="*80)
best=None
for lev in np.arange(1,80,1.0):
    eq=paths(lev,8,n=25000)
    hit=(eq>=20000).any(axis=1).mean()
    if best is None or hit>best[1]: best=(lev,hit)
lev,hit=best
eq=paths(lev,8)
low=eq.min(axis=1)
print(f"  leverage maximising P($20k in 8 weeks): {lev:.0f}x")
print(f"    P(reach $20,000)      {hit*100:>6.1f}%")
print(f"    P(below $5,000)       {(low<5000).mean()*100:>6.1f}%")
print(f"    P(below $2,000)       {(low<2000).mean()*100:>6.1f}%")
print(f"    median ending equity  ${np.median(eq[:,-1]):>9,.0f}")
print(f"    10th percentile       ${np.percentile(eq[:,-1],10):>9,.0f}")
print(f"\n  So the best achievable is roughly {hit*100:.0f}% -- and the same")
print(f"  variance that creates that chance produces the downside beside it.")

print("\n"+"="*80)
print("IF IT WORKS, CAN IT REPEAT?  (the compounding question)")
print("="*80)
p=hit
print(f"  P(double once)        {p*100:>6.1f}%")
for k in (2,3,6):
    print(f"  P(double {k}x in a row) {p**k*100:>6.2f}%   "
          f"-> $10k becomes ${10000*2**k:,}")
print(f"\n  Six consecutive doublings is the year that reaches $640,000.")
print(f"  At {p*100:.0f}% per attempt that is {p**6*100:.4f}% -- about 1 in "
      f"{1/p**6:,.0f}.")
