"""ONE YEAR FROM $10,000 -- the actual distribution, not a point estimate.

'26.1%/yr' is a growth RATE, not a promise. One year is a short sample at 43%
volatility, so the honest answer is a distribution. This resamples the REAL
weekly returns of the k=0.5 / 3x system in 4-week blocks (block bootstrap
preserves the autocorrelation and the clustering of bad weeks, which an
iid draw would wash out), 50,000 times.

Reported at every leverage so the tradeoff is visible rather than argued.
"""
import numpy as np, pandas as pd
import factor_lab as F

px=F.universe(); wk=px.resample("W-FRI").last()
fwd=wk.pct_change().shift(-1); raw=px.pct_change(252).reindex(wk.index,method="ffill")
COST=10/1e4

def series(k,lev):
    out=[];prev={}
    for t in wk.index[:-1]:
        a=raw.loc[t].dropna()
        if len(a)<8: continue
        z=(a-a.mean())/a.std(ddof=1)
        w=np.exp(k*z); w=w/w.sum()*lev
        f=fwd.loc[t].reindex(w.index).fillna(0)
        turn=sum(abs(w.get(x,0)-prev.get(x,0)) for x in set(w.index)|set(prev))
        out.append(float((w*f).sum())-turn*COST); prev=w.to_dict()
    return np.array(out)

rng=np.random.default_rng(19)
def boot(r,weeks=52,n=50000,block=4):
    nb=weeks//block
    idx=rng.integers(0,len(r)-block,size=(n,nb))
    paths=np.concatenate([r[idx+j] for j in range(block)],axis=1)
    eq=np.cumprod(1+np.clip(paths,-0.99,None),axis=1)
    return eq[:,-1]*10000, eq.min(axis=1)*10000

print("="*78)
print("$10,000 AFTER ONE YEAR -- 50,000 bootstrapped paths")
print("="*78)
print(f"  {'config':<18}{'p10':>9}{'p25':>9}{'MEDIAN':>10}{'p75':>9}{'p90':>9}"
      f"{'P(loss)':>9}{'P(2x)':>7}")
print("  "+"-"*70)
cfgs=[("equal wt, 1x",0,1.0),("tilt 0.5, 1x",0.5,1.0),("tilt 0.5, 2x",0.5,2.0),
      ("tilt 0.5, 3x",0.5,3.0),("tilt 0.5, 5x",0.5,5.0)]
store={}
for lab,k,lev in cfgs:
    r=series(k,lev); fin,low=boot(r); store[lab]=(fin,low)
    p=np.percentile(fin,[10,25,50,75,90])
    print(f"  {lab:<18}{p[0]:>9,.0f}{p[1]:>9,.0f}{p[2]:>10,.0f}{p[3]:>9,.0f}"
          f"{p[4]:>9,.0f}{(fin<10000).mean()*100:>8.0f}%{(fin>=20000).mean()*100:>6.0f}%")

print("\n"+"="*78)
print("WHAT THAT IS PER MONTH")
print("="*78)
print(f"  {'config':<18}{'median profit':>15}{'= per month':>14}{'worst 10%':>12}")
print("  "+"-"*60)
for lab,k,lev in cfgs:
    fin,low=store[lab]
    med=np.median(fin)-10000
    p10=np.percentile(fin,10)-10000
    print(f"  {lab:<18}{med:>+15,.0f}{med/12:>+14,.0f}{p10:>+12,.0f}")

print("\n"+"="*78)
print("ODDS OF HITTING THE TARGET IN ONE YEAR")
print("="*78)
print("  ($10k/month sustainably needs roughly $1,000,000)\n")
for lab,k,lev in cfgs:
    fin,low=store[lab]
    print(f"  {lab:<18} P($1M in 1yr) = {(fin>=1_000_000).mean()*100:.4f}%"
          f"   P(down 50%+ at some point) = {(low<5000).mean()*100:>4.0f}%")
print("\n  For scale: to reach $1,000,000 in one year from $10,000 you need")
print("  +9,900%. The best single year in this system's 23-year history was")
for lab,k,lev in [("tilt 0.5, 3x",0.5,3.0)]:
    r=series(k,lev)
    yr=pd.Series(r).rolling(52).apply(lambda x:np.prod(1+x)-1).max()*100
    print(f"  {yr:+.0f}%.")
