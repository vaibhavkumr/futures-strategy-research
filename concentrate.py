"""CONCENTRATION: putting a BIG SLICE of the 10k on the best trade.

The previous test spread weights proportionally over six positions. That is
not the proposal. The proposal is: when confidence is high, commit a large
fraction of capital to THAT trade.

This is a different question and it deserves its own test, because the two
effects pull against each other:

  GOOD: the top-conviction bucket really does earn more (Q5 25.4bp/wk vs
        Q1 16.3bp/wk, and it held on holdout). Concentrating moves capital
        into the higher-mean bucket.
  BAD:  concentration destroys diversification. Six positions average away
        idiosyncratic risk; one position keeps all of it. Variance rises far
        faster than the mean, and growth rate is mean MINUS variance/2.

Which wins is arithmetic, so measure it. Two axes, on real data:
  1. NUMBER of positions: 6 -> 1
  2. FRACTION of the 10k committed: 25% -> 200% (leverage)

Reported as GROWTH RATE, the thing that compounds -- not average return.
Average return rises with concentration even when growth falls, and that gap
is precisely how accounts die while the "average trade" looks fine.
"""
import numpy as np, pandas as pd
import factor_lab as F

px=F.universe()
wk=px.resample("W-FRI").last()
fwd=wk.pct_change().shift(-1)
sc=px.pct_change(252).reindex(wk.index,method="ffill")
DEV="2018-01-01"
COST=10/1e4

def sim(npos,frac):
    """Hold the top-`npos` by conviction, committing `frac` of capital."""
    rets=[];prev={}
    for t in wk.index[:-1]:
        a=sc.loc[t].dropna()
        if len(a)<npos: continue
        pick=a.nlargest(npos).index
        w=pd.Series(frac/npos,index=pick)
        f=fwd.loc[t].reindex(pick).fillna(0)
        turn=sum(abs(w.get(k,0)-prev.get(k,0)) for k in set(pick)|set(prev))
        rets.append(dict(ts=t,r=float((w*f).sum())-turn*COST))
        prev=w.to_dict()
    return pd.DataFrame(rets).set_index("ts").r

def stats(x,ann=52):
    x=x.dropna()
    if len(x)<50: return None
    x=np.clip(x,-0.99,None)
    eq=(1+x).cumprod()
    g=np.log1p(x).mean()*ann                    # LOG growth -- what compounds
    return dict(growth=(np.exp(g)-1)*100,
                avg=x.mean()*ann*100,           # naive average -- what looks good
                vol=x.std(ddof=1)*np.sqrt(ann)*100,
                dd=(eq/eq.cummax()-1).min()*100,
                sharpe=x.mean()/x.std(ddof=1)*np.sqrt(ann),
                final=10000*eq.iloc[-1])

print("="*80)
print("1. HOW MANY POSITIONS?  (100% of capital, no leverage)")
print("="*80)
print(f"  {'positions':<12}{'avg ret':>10}{'GROWTH':>10}{'vol':>8}{'maxDD':>9}"
      f"{'Sharpe':>8}{'$10k -> ':>14}")
print("  "+"-"*66)
for n in (6,4,3,2,1):
    s=stats(sim(n,1.0))
    print(f"  {n:<12}{s['avg']:>9.1f}%{s['growth']:>9.1f}%{s['vol']:>7.0f}%"
          f"{s['dd']:>8.0f}%{s['sharpe']:>8.2f}{s['final']:>14,.0f}")
print("\n  note avg ret vs GROWTH: concentration raises the average and LOWERS")
print("  what actually compounds. That gap is variance drag.")

print("\n"+"="*80)
print("2. HOW MUCH OF THE 10k ON THE SINGLE BEST TRADE?")
print("="*80)
print(f"  {'capital':<12}{'avg ret':>10}{'GROWTH':>10}{'vol':>8}{'maxDD':>9}"
      f"{'$10k -> ':>14}{'DEV g':>9}{'HOLD g':>9}")
print("  "+"-"*74)
for f in (0.25,0.5,1.0,1.5,2.0,3.0,5.0):
    s=sim(1,f); t=stats(s)
    d=stats(s[s.index<DEV]); h=stats(s[s.index>=DEV])
    print(f"  {f*100:>4.0f}%{'':<7}{t['avg']:>9.1f}%{t['growth']:>9.1f}%"
          f"{t['vol']:>7.0f}%{t['dd']:>8.0f}%{t['final']:>14,.0f}"
          f"{d['growth'] if d else np.nan:>8.1f}%{h['growth'] if h else np.nan:>8.1f}%")

print("\n"+"="*80)
print("3. THE SAME THING, RESTRICTED TO THE HIGHEST-CONVICTION WEEKS ONLY")
print("="*80)
print("  (only trade when the top score is in the strongest 20% historically)\n")
top1=sc.max(axis=1)
thr=top1.expanding(52).quantile(0.8)
hot=(top1>thr)
rets=[];prev={}
for t in wk.index[:-1]:
    a=sc.loc[t].dropna()
    if len(a)<1: continue
    if not bool(hot.get(t,False)):
        rets.append(dict(ts=t,r=0.0)); prev={}; continue
    pick=a.nlargest(1).index
    w=pd.Series(1.0,index=pick); f=fwd.loc[t].reindex(pick).fillna(0)
    turn=sum(abs(w.get(k,0)-prev.get(k,0)) for k in set(pick)|set(prev))
    rets.append(dict(ts=t,r=float((w*f).sum())-turn*COST)); prev=w.to_dict()
S=pd.DataFrame(rets).set_index("ts").r
for lab,x in (("conviction-gated, 1 pos",S),("always on, 1 pos",sim(1,1.0)),
              ("always on, 6 pos",sim(6,1.0))):
    s=stats(x)
    print(f"  {lab:<26} growth {s['growth']:>6.1f}%  vol {s['vol']:>4.0f}%  "
          f"maxDD {s['dd']:>5.0f}%  Sharpe {s['sharpe']:>5.2f}")
