"""DOES CONVICTION WORK ON AN EDGE THAT ACTUALLY EXISTS?

Testing confidence on a zero-edge strategy is rigged: you cannot rank the
magnitude of something whose mean is zero. Both previous tests did exactly
that. The fair test is whether conviction adds anything to mom_12m, the one
signal that has survived every control I have thrown at it.

If momentum SCORE predicts the SIZE of the subsequent return, and not merely
its sign, then sizing in proportion to score lifts Sharpe above 1.14 -- and
that is the only legitimate route toward the target that remains open.

  1. does score magnitude predict return magnitude, by quintile
  2. does score-weighted sizing beat equal weighting, dev and holdout
  3. what CV does the predictable component actually have
"""
import numpy as np, pandas as pd, yfinance as yf
import factor_lab as F

px=F.universe()
r=px.pct_change()
S=px.pct_change(252)              # the momentum score
DEV="2018-01-01"

# weekly forward returns paired with the score known beforehand
wk=px.resample("W-FRI").last()
fwd=wk.pct_change().shift(-1)     # next week's return
sc=S.reindex(wk.index,method="ffill")
rows=[]
for t in wk.index:
    a,b=sc.loc[t].dropna(),fwd.loc[t].dropna()
    j=a.index.intersection(b.index)
    if len(j)<8: continue
    z=(a[j]-a[j].mean())/a[j].std()
    for k in j: rows.append(dict(ts=t,asset=k,score=a[k],z=z[k],fwd=b[k]))
D=pd.DataFrame(rows).dropna()
print(f"{len(D):,} asset-weeks  {D.ts.min():%Y-%m} -> {D.ts.max():%Y-%m}\n")

print("1. DOES SCORE MAGNITUDE PREDICT RETURN?  (quintiles of momentum z-score)\n")
D["q"]=pd.qcut(D.z,5,labels=False,duplicates="drop")
print(f"  {'quintile':<12}{'n':>8}{'mean z':>9}{'fwd ret bp':>12}{'t':>8}"
      f"{'HOLDOUT bp':>12}{'t':>8}")
print("  "+"-"*62)
for q in sorted(D.q.unique()):
    g=D[D.q==q]; hd=g[g.ts>=DEV].fwd*1e4
    tt=lambda x: x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
    print(f"  {'Q'+str(int(q)+1):<12}{len(g):>8}{g.z.mean():>9.2f}"
          f"{g.fwd.mean()*1e4:>12.1f}{tt(g.fwd*1e4):>8.2f}"
          f"{hd.mean():>12.1f}{tt(hd):>8.2f}")
hi,lo=D[D.q==D.q.max()],D[D.q==D.q.min()]
sp=(hi.fwd.mean()-lo.fwd.mean())*1e4
se=np.sqrt(hi.fwd.var(ddof=1)/len(hi)+lo.fwd.var(ddof=1)/len(lo))*1e4
print(f"\n  TOP minus BOTTOM = {sp:+.1f}bp/week   t = {sp/se:+.2f}")
hh,ll=hi[hi.ts>=DEV],lo[lo.ts>=DEV]
sp2=(hh.fwd.mean()-ll.fwd.mean())*1e4
se2=np.sqrt(hh.fwd.var(ddof=1)/len(hh)+ll.fwd.var(ddof=1)/len(ll))*1e4
print(f"  HOLDOUT          = {sp2:+.1f}bp/week   t = {sp2/se2:+.2f}")

print("\n2. SCORE-WEIGHTED SIZING vs EQUAL WEIGHT (top 6, weekly, 10bp cost)\n")
def run(weight_by_score):
    out=[];prev={}
    for t in wk.index[:-1]:
        a=sc.loc[t].dropna()
        if len(a)<6: continue
        pick=a.nlargest(6)
        w=(pick-pick.min()+1e-9) if weight_by_score else pd.Series(1.0,index=pick.index)
        w=w/w.sum()
        f=fwd.loc[t].reindex(w.index).fillna(0)
        turn=sum(abs(w.get(k,0)-prev.get(k,0)) for k in set(w.index)|set(prev))
        out.append(dict(ts=t,r=float((w*f).sum())-turn*10/1e4))
        prev=w.to_dict()
    return pd.DataFrame(out).set_index("ts").r
def m(x,ann=52):
    x=x.dropna(); eq=(1+x).cumprod(); yrs=len(x)/ann
    return (eq.iloc[-1]**(1/yrs)-1)*100, x.mean()/x.std(ddof=1)*np.sqrt(ann)
for lab,ws in (("equal weight",False),("score-weighted",True)):
    s=run(ws)
    c,sh=m(s); cd,shd=m(s[s.index<DEV]); ch,shh=m(s[s.index>=DEV])
    print(f"  {lab:<16} CAGR {c:>6.2f}%  Sharpe {sh:>5.2f}   |  "
          f"DEV shp {shd:>5.2f}   HOLDOUT shp {shh:>5.2f}")

print("\n3. CV OF THE PREDICTABLE EDGE\n")
pred=D.groupby("q").fwd.mean()
cv=pred.std()/abs(pred.mean()) if pred.mean() else np.nan
print(f"  across-quintile predictable edge: mean {pred.mean()*1e4:.1f}bp, "
      f"sd {pred.std()*1e4:.1f}bp  ->  CV = {cv:.2f}")
print(f"  Sharpe 1.14 * sqrt(1+CV^2) = {1.14*np.sqrt(1+cv**2):.2f}"
      f"   (need 4.08 for the target)")
