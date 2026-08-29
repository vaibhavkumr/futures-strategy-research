"""GRADUATED CONVICTION TILT -- the actual proposal.

Not gating (skip the weak ones -- tested, destroys it: 0.6% growth).
Not pure concentration (only the single best -- tested, 15.5%).
This is: hold everything at a base size, and ADD size in proportion to
confidence.

Implemented as an exponential tilt on the conviction z-score:

    w_i  proportional to  exp(k * z_i)

    k = 0   every position identical                (equal weight)
    k small mild overweight to the confident ones
    k large capital collapses onto the top name     (concentration)

So a single knob spans the whole range, with the two things I already
measured sitting at its ends. Sweeping k finds the optimum rather than
assuming it is at one extreme.

Scored on GROWTH RATE, since that is what compounds -- and reported next to
average return, because concentration always flatters the average.
"""
import numpy as np, pandas as pd
import factor_lab as F

px=F.universe()
wk=px.resample("W-FRI").last()
fwd=wk.pct_change().shift(-1)
raw=px.pct_change(252).reindex(wk.index,method="ffill")
DEV="2018-01-01"; COST=10/1e4

def sim(k,lev=1.0,universe_n=26):
    rets=[];prev={}
    for t in wk.index[:-1]:
        a=raw.loc[t].dropna()
        if len(a)<8: continue
        z=(a-a.mean())/a.std(ddof=1)
        z=z.nlargest(min(universe_n,len(z)))
        w=np.exp(k*z); w=w/w.sum()*lev
        f=fwd.loc[t].reindex(w.index).fillna(0)
        turn=sum(abs(w.get(x,0)-prev.get(x,0)) for x in set(w.index)|set(prev))
        rets.append(dict(ts=t,r=float((w*f).sum())-turn*COST))
        prev=w.to_dict()
    return pd.DataFrame(rets).set_index("ts").r

def stats(x,ann=52):
    x=x.dropna()
    if len(x)<50: return None
    x=np.clip(x,-0.99,None); eq=(1+x).cumprod()
    return dict(growth=(np.exp(np.log1p(x).mean()*ann)-1)*100,
                avg=x.mean()*ann*100, vol=x.std(ddof=1)*np.sqrt(ann)*100,
                dd=(eq/eq.cummax()-1).min()*100,
                sharpe=x.mean()/x.std(ddof=1)*np.sqrt(ann),
                final=10000*eq.iloc[-1], eff=1/((x.std(ddof=1))**2))

print("="*82)
print("TILT SWEEP -- k=0 is equal weight, larger k = more on high conviction")
print("="*82)
print(f"  {'k':<6}{'top wt':>8}{'avg ret':>10}{'GROWTH':>9}{'vol':>7}{'maxDD':>8}"
      f"{'Sharpe':>8}{'$10k ->':>12}{'DEV g':>8}{'HOLD g':>8}")
print("  "+"-"*76)
best=None
for k in (0,0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0,8.0):
    s=sim(k); t=stats(s)
    if t is None: continue
    d=stats(s[s.index<DEV]); h=stats(s[s.index>=DEV])
    # what share the single top name gets, on average
    a=raw.loc[wk.index[300]].dropna(); z=(a-a.mean())/a.std(ddof=1)
    w=np.exp(k*z); tw=(w/w.sum()).max()*100
    star=""
    if best is None or t["growth"]>best[1]: best=(k,t["growth"]); star=""
    print(f"  {k:<6.2f}{tw:>7.0f}%{t['avg']:>9.1f}%{t['growth']:>8.1f}%"
          f"{t['vol']:>6.0f}%{t['dd']:>7.0f}%{t['sharpe']:>8.2f}"
          f"{t['final']:>12,.0f}{d['growth'] if d else np.nan:>7.1f}%"
          f"{h['growth'] if h else np.nan:>7.1f}%")
print(f"\n  best growth at k = {best[0]}  ({best[1]:.1f}%/yr)")

print("\n"+"="*82)
print("BEST TILT + LEVERAGE  (the two knobs together)")
print("="*82)
print(f"  {'k':<6}{'lev':>6}{'GROWTH':>9}{'vol':>7}{'maxDD':>8}{'Sharpe':>8}"
      f"{'$10k ->':>13}{'DEV g':>8}{'HOLD g':>8}")
print("  "+"-"*70)
rows=[]
for k in (0,0.5,1.0,2.0):
    for lev in (1.0,1.5,2.0,3.0):
        s=sim(k,lev); t=stats(s)
        if t is None: continue
        d=stats(s[s.index<DEV]); h=stats(s[s.index>=DEV])
        rows.append((t["growth"],k,lev,t,d,h))
        print(f"  {k:<6.2f}{lev:>5.1f}x{t['growth']:>8.1f}%{t['vol']:>6.0f}%"
              f"{t['dd']:>7.0f}%{t['sharpe']:>8.2f}{t['final']:>13,.0f}"
              f"{d['growth'] if d else np.nan:>7.1f}%{h['growth'] if h else np.nan:>7.1f}%")
rows.sort(reverse=True)
g,k,lev,t,d,h=rows[0]
print(f"\n  BEST: k={k}, leverage {lev}x  ->  growth {g:.1f}%/yr, "
      f"maxDD {t['dd']:.0f}%, Sharpe {t['sharpe']:.2f}")
print(f"        DEV {d['growth']:.1f}%   HOLDOUT {h['growth']:.1f}%   "
      f"(both positive: {'YES' if d['growth']>0 and h['growth']>0 else 'NO'})")
print(f"        $10,000 -> ${t['final']:,.0f} over {len(wk)/52:.0f} years")

print("\n"+"="*82)
print("SANITY: is the tilt doing anything, or is it just leverage?")
print("="*82)
b=stats(sim(0,1.0)); bl=stats(sim(0,lev))
print(f"  equal weight, 1x        growth {b['growth']:>6.1f}%  Sharpe {b['sharpe']:.2f}")
print(f"  equal weight, {lev}x      growth {bl['growth']:>6.1f}%  Sharpe {bl['sharpe']:.2f}")
print(f"  tilted k={k}, {lev}x       growth {g:>6.1f}%  Sharpe {t['sharpe']:.2f}")
print(f"\n  tilt contributes {g-bl['growth']:+.1f} pct-pts of growth beyond pure leverage")
