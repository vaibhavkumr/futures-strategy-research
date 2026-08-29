"""Re-analysis with degenerate stops excluded.

94 trades had risk_atr ~ 0, where the cost-in-R term (0.5bp * price/risk)
divides by nearly zero and produces R values to -10,733. Nobody trades a
zero-width stop; these are artifacts of the FVG-producing candle being tiny.
Requiring a minimum stop of 0.15 ATR is what any real trader would do and is
the same "give it a little bit of space" the video describes.
"""
import pandas as pd, numpy as np
T=pd.read_pickle('video_feat.pkl'); T['ts']=pd.to_datetime(T.ts)
MIN_RISK=0.15
C=T[T.risk_atr>=MIN_RISK].copy()
print(f"{len(T):,} trades -> {len(C):,} after requiring stop >= {MIN_RISK} ATR")
print(f"R range now: {C.R.min():.2f} to {C.R.max():.2f}   std {C.R.std():.2f}\n")
DEV="2025-10-01"
dev,hold=C[C.ts<DEV],C[C.ts>=DEV]

def show(lab,t,ind=""):
    if len(t)<60: print(f"  {ind}{lab:<30} n={len(t)} too few"); return None
    R=t.R.values; m,se=R.mean(),R.std(ddof=1)/np.sqrt(len(R))
    print(f"  {ind}{lab:<30}{len(R):>7}{(R>2).mean()*100:>7.1f}%{m:>+9.3f}{m/se:>7.2f}")
    return dict(mean=m,t=m/se)

print("="*72); print("BASELINE (clean)"); print("="*72)
print(f"  {'':<30}{'n':>7}{'win%':>7}{'expR':>9}{'t':>7}"); print("  "+"-"*60)
show("dev",dev); show("holdout",hold)
print("\n  per-market, dev:")
for mk in C.mkt.unique(): show(mk,dev[dev.mkt==mk],"  ")

print("\n"+"="*72); print("THE choch_str FILTER (decisive break of structure)"); print("="*72)
thr=dev.choch_str.quantile(0.70)
print(f"  threshold: choch_str >= {thr:.3f} ATR (dev p70)\n")
print(f"  {'':<30}{'n':>7}{'win%':>7}{'expR':>9}{'t':>7}"); print("  "+"-"*60)
show("DEV",dev[dev.choch_str>=thr]); show("HOLDOUT",hold[hold.choch_str>=thr])
print("\n  per-market (whole sample, filtered):")
F=C[C.choch_str>=thr]
for mk in C.mkt.unique(): show(mk,F[F.mkt==mk],"  ")
d=pd.DataFrame({mk:F[F.mkt==mk].groupby('ts').R.mean() for mk in C.mkt.unique()})
cor=d.corr().values[np.triu_indices(d.shape[1],1)]
print(f"\n  cross-market corr {np.abs(cor).mean():.3f} -> effective independent markets "
      f"{d.shape[1]/(1+(d.shape[1]-1)*np.abs(cor).mean()):.2f} of {d.shape[1]}")
