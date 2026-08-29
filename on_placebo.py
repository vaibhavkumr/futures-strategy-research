"""PLACEBO for the overnight filters.

'down + VIX elevated' shows net +4.66bp/night, DEV +3.53, HOLDOUT +6.25.
That is the shape that has fooled me repeatedly today, so before it counts:

  1. RANDOM NIGHTS placebo. Pick the same NUMBER of nights at random, with
     the same block structure, and see how often that clears the same bar.
     This is the control that revealed 58% of my 'surviving' index signals
     were luck.
  2. The base overnight premium is +4.41bp gross at t=3.97, so ANY subset of
     nights inherits a positive mean. The filter must beat a random subset,
     not beat zero.
  3. Bonferroni: 10 filters tested, so the honest bar is |t| > 3.
"""
import numpy as np, pandas as pd, yfinance as yf
COST=2.0; DEV="2018-01-01"

d=yf.download(["SPY","QQQ","IWM","XLK","XLF","EEM"],start="2004-01-01",
              progress=False,auto_adjust=True)
C,O=d["Close"].ffill(),d["Open"].ffill()
V=yf.download("^VIX",start="2004-01-01",progress=False,auto_adjust=True)["Close"].ffill()
if isinstance(V,pd.DataFrame): V=V.iloc[:,0]
F=pd.DataFrame(index=C.index)
F["on"]=((O/C.shift(1)-1).mean(axis=1))*1e4
F["intra"]=((C/O-1).mean(axis=1))*1e4
F["vix"]=V.reindex(C.index).ffill()
F["vixz"]=(F.vix-F.vix.rolling(252).mean())/F.vix.rolling(252).std()
F=F.dropna(subset=["on","vixz"])

real=(F.intra.shift(1)<0)&(F.vixz.shift(1)>0.5)
real=real.reindex(F.index).fillna(False)
n_real=int(real.sum())
x=F.on[real]
t_real=x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
net_real=x.mean()-2*COST
print(f"REAL filter: {n_real} nights ({n_real/len(F)*100:.0f}%)  "
      f"gross {x.mean():.2f}bp  t={t_real:.2f}  net {net_real:+.2f}bp\n")

print("PLACEBO 1 -- random nights, same count, N=500")
rng=np.random.default_rng(11)
idx=np.arange(len(F)); on=F.on.values
dev_m=F.index<DEV
beat=0; better_net=0; nets=[]
for k in range(500):
    pick=rng.choice(idx,size=n_real,replace=False)
    m=np.zeros(len(F),bool); m[pick]=True
    a,b=on[m&dev_m],on[m&~dev_m]
    if len(a)<50 or len(b)<50: continue
    nt=on[m].mean()-2*COST
    nets.append(nt)
    if nt>0 and a.mean()-2*COST>0 and b.mean()-2*COST>0: beat+=1
    if nt>=net_real: better_net+=1
nets=np.array(nets)
print(f"  random subsets clearing the SAME bar (net>0 in both periods): "
      f"{beat}/{len(nets)} = {beat/len(nets)*100:.0f}%")
print(f"  random subsets matching or beating {net_real:+.2f}bp net: "
      f"{better_net}/{len(nets)} = {better_net/len(nets)*100:.1f}%")
print(f"  random-subset net: mean {nets.mean():+.2f}bp, sd {nets.std():.2f}, "
      f"95th pct {np.percentile(nets,95):+.2f}bp")

print("\nPLACEBO 2 -- shuffle the FILTER, keep its block structure")
vals=real.values.copy(); beat2=0
for k in range(500):
    sh=np.roll(vals,rng.integers(50,len(vals)-50))
    a,b=on[sh&dev_m],on[sh&~dev_m]
    if len(a)<50 or len(b)<50: continue
    if on[sh].mean()-2*COST>0 and a.mean()-2*COST>0 and b.mean()-2*COST>0: beat2+=1
print(f"  circular-shifted filter clearing the bar: {beat2}/500 = {beat2/5:.0f}%")

print(f"\nVERDICT")
print(f"  real t = {t_real:.2f}   Bonferroni bar for 10 filters = 3.00   "
      f"{'PASS' if abs(t_real)>3 else 'FAIL'}")
print(f"  beats random subsets at p = {better_net/len(nets):.3f}")
