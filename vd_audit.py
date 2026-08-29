"""AUDIT THE choch_str FILTER.

DEV +0.207R (t=7.95), HOLDOUT +0.184R (t=4.96), 4/4 markets positive,
cross-market correlation only 0.103. That is stronger than anything else
intraday in this project, so it gets the treatment that killed the pivot
result and six of my own bugs.

  1. ROBUSTNESS to the minimum-stop cutoff. If the edge only exists at
     MIN_RISK=0.15 it is a cost artifact, not a signal.
  2. ROBUSTNESS to the threshold. A real effect is monotone in filter
     strength, not a spike at one quantile.
  3. PLACEBO. A random filter keeping the same fraction of trades. This is
     what revealed 58% of my "surviving" index signals were luck.
  4. SUBPERIOD. Does it hold year by year, or is it one good stretch?
"""
import pandas as pd, numpy as np
T=pd.read_pickle('video_feat.pkl'); T['ts']=pd.to_datetime(T.ts)
DEV="2025-10-01"

def stat(t):
    if len(t)<60: return None
    R=t.R.values; m,se=R.mean(),R.std(ddof=1)/np.sqrt(len(R))
    return len(R),(R>2).mean()*100,m,m/se

print("="*74); print("1. ROBUSTNESS TO THE MINIMUM-STOP CUTOFF"); print("="*74)
print(f"  {'min stop (ATR)':<18}{'n':>7}{'win%':>7}{'expR':>9}{'t':>7}")
print("  "+"-"*48)
for mr in (0.05,0.10,0.15,0.25,0.40):
    C=T[T.risk_atr>=mr]
    thr=C[C.ts<DEV].choch_str.quantile(0.70)
    s=stat(C[C.choch_str>=thr])
    if s: print(f"  {mr:<18.2f}{s[0]:>7}{s[1]:>7.1f}%{s[2]:>+9.3f}{s[3]:>7.2f}")

C=T[T.risk_atr>=0.15].copy()
dev,hold=C[C.ts<DEV],C[C.ts>=DEV]
print("\n"+"="*74); print("2. MONOTONE IN FILTER STRENGTH?"); print("="*74)
print(f"  {'choch_str quantile':<22}{'n':>7}{'win%':>7}{'expR':>9}{'t':>7}")
print("  "+"-"*52)
for q in (0.0,0.3,0.5,0.6,0.7,0.8,0.9):
    thr=dev.choch_str.quantile(q)
    s=stat(C[C.choch_str>=thr])
    if s: print(f"  >= p{int(q*100):<19}{s[0]:>7}{s[1]:>7.1f}%{s[2]:>+9.3f}{s[3]:>7.2f}")

print("\n"+"="*74); print("3. PLACEBO -- random filter, same trade count"); print("="*74)
thr=dev.choch_str.quantile(0.70)
real=stat(C[C.choch_str>=thr]); frac=(C.choch_str>=thr).mean()
rng=np.random.default_rng(5); beat=0; means=[]
for i in range(400):
    m=rng.random(len(C))<frac
    s=stat(C[m])
    if s: means.append(s[2]); beat+= s[2]>=real[2]
means=np.array(means)
print(f"  real filter      expR {real[2]:+.3f}  t {real[3]:.2f}")
print(f"  random filters   mean {means.mean():+.3f}  sd {means.std():.3f}  "
      f"95th {np.percentile(means,95):+.3f}")
print(f"  random beating real: {beat}/{len(means)} = {beat/len(means)*100:.1f}%")

print("\n"+"="*74); print("4. YEAR BY YEAR"); print("="*74)
F=C[C.choch_str>=thr]
print(f"  {'period':<14}{'n':>7}{'win%':>7}{'expR':>9}{'t':>7}")
print("  "+"-"*44)
for per,g in F.groupby(F.ts.dt.to_period('Y')):
    s=stat(g)
    if s: print(f"  {str(per):<14}{s[0]:>7}{s[1]:>7.1f}%{s[2]:>+9.3f}{s[3]:>7.2f}")
print("\n  by half-year:")
for per,g in F.groupby(F.ts.dt.to_period('Q')):
    s=stat(g)
    if s: print(f"  {str(per):<14}{s[0]:>7}{s[1]:>7.1f}%{s[2]:>+9.3f}{s[3]:>7.2f}")
