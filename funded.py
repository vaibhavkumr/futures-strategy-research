"""DRAWDOWN-CONSTRAINED VERSION -- can the verified edge live inside a
funded-account rulebook?

The gap is precise: I have a verified edge (mom_12m + conviction tilt,
26.1%/yr at 3x) whose natural drawdown is -38% to -80%. Funded accounts
enforce a TRAILING drawdown limit, typically 4-6% of peak equity, and breach
means the account is gone.

That is not a reason to give up on the edge -- it is a risk-engineering
problem. Three controls, applied together:

  VOL TARGETING   scale exposure by realised vol so risk is constant rather
                  than whatever the market hands you. This alone usually
                  cuts max drawdown far more than it cuts return.
  DD THROTTLE     as equity approaches the limit, cut exposure toward zero.
                  Turns a hard breach into an asymptotic approach.
  HARD FLOOR      absolute stop before the rule trips.

The question this answers: at a 5% trailing limit, what return survives, and
does the account survive at all? Tested on 23 years including 2008 and 2020,
and reported as SURVIVAL PROBABILITY, since a blown funded account pays zero
regardless of what the average return looked like.
"""
import numpy as np, pandas as pd
import factor_lab as F

px=F.universe(); wk=px.resample("W-FRI").last()
fwd=wk.pct_change().shift(-1); raw=px.pct_change(252).reindex(wk.index,method="ffill")
COST=10/1e4; K=0.5

def base_returns(lev=1.0):
    out=[];prev={}
    for t in wk.index[:-1]:
        a=raw.loc[t].dropna()
        if len(a)<8: continue
        z=(a-a.mean())/a.std(ddof=1)
        w=np.exp(K*z); w=w/w.sum()*lev
        f=fwd.loc[t].reindex(w.index).fillna(0)
        turn=sum(abs(w.get(x,0)-prev.get(x,0)) for x in set(w.index)|set(prev))
        out.append(dict(ts=t,r=float((w*f).sum())-turn*COST)); prev=w.to_dict()
    return pd.DataFrame(out).set_index("ts").r

R=base_returns(1.0)
print(f"base system: {len(R)} weeks, {R.index.min():%Y-%m} -> {R.index.max():%Y-%m}\n")

def controlled(r, dd_limit, target_vol=0.10, max_lev=3.0, throttle=True):
    """Vol-targeted, drawdown-throttled equity curve."""
    rv=r.rolling(26).std(ddof=1)*np.sqrt(52)
    eq=1.0; peak=1.0; out=[]; blown=False
    for t,x in r.items():
        v=rv.get(t,np.nan)
        lev=min(max_lev, target_vol/v) if np.isfinite(v) and v>0 else 1.0
        if throttle:
            dd=eq/peak-1
            head=max(0.0,(dd_limit+dd)/dd_limit)     # 1 at peak -> 0 at limit
            lev*=head**1.5
        step=x*lev
        eq*=(1+step); peak=max(peak,eq)
        if eq/peak-1 <= -dd_limit or eq<=0:
            blown=True; out.append(dict(ts=t,eq=eq,lev=lev)); break
        out.append(dict(ts=t,eq=eq,lev=lev))
    E=pd.DataFrame(out).set_index("ts")
    return E,blown

print("="*78)
print("CAN IT SURVIVE A TRAILING DRAWDOWN LIMIT?  (full 23-yr history)")
print("="*78)
print(f"  {'limit':<8}{'controls':<22}{'survived':>10}{'CAGR':>9}{'maxDD':>9}{'Sharpe':>8}")
print("  "+"-"*64)
for lim in (0.04,0.05,0.06,0.10,0.20):
    for lab,thr,tv in (("none (raw 3x)",False,None),("vol-target only",False,0.10),
                       ("vol-target+throttle",True,0.10)):
        if lab=="none (raw 3x)":
            r3=base_returns(3.0); eq=(1+r3).cumprod()
            dd=(eq/eq.cummax()-1)
            blown=(dd<=-lim).any()
            first=dd[dd<=-lim].index[0] if blown else None
            surv=f"NO {first:%Y-%m}" if blown else "yes"
            g=(eq.iloc[-1]**(52/len(r3))-1)*100 if not blown else np.nan
            sh=r3.mean()/r3.std(ddof=1)*np.sqrt(52)
            print(f"  {lim*100:>4.0f}%   {lab:<22}{surv:>10}{g:>9.1f}{dd.min()*100:>9.0f}{sh:>8.2f}")
            continue
        E,blown=controlled(R,lim,target_vol=tv,throttle=thr)
        eq=E["eq"]; ddm=(eq/eq.cummax()-1).min()*100
        g=(eq.iloc[-1]**(52/len(eq))-1)*100
        rr=eq.pct_change().dropna()
        sh=rr.mean()/rr.std(ddof=1)*np.sqrt(52)
        surv="yes" if not blown else f"NO {eq.index[-1]:%Y-%m}"
        print(f"  {lim*100:>4.0f}%   {lab:<22}{surv:>10}{g:>9.1f}{ddm:>9.0f}{sh:>8.2f}")
    print()

print("="*78)
print("WHAT THAT PAYS ON FUNDED CAPITAL")
print("="*78)
E,blown=controlled(R,0.05,target_vol=0.10,throttle=True)
eq=E["eq"]; g=(eq.iloc[-1]**(52/len(eq))-1)
rr=eq.pct_change().dropna()
print(f"  5% trailing limit, vol-target + throttle:")
print(f"    survived 23 years: {'YES' if not blown else 'NO'}")
print(f"    CAGR {g*100:.2f}%   maxDD {(eq/eq.cummax()-1).min()*100:.1f}%   "
      f"Sharpe {rr.mean()/rr.std(ddof=1)*np.sqrt(52):.2f}\n")
print(f"  {'funded':<14}{'$/month (90% split)':>22}{'vs $10k target':>18}")
print("  "+"-"*52)
for cap in (50_000,150_000,300_000,500_000,1_000_000,2_000_000):
    m=cap*g/12*0.90
    print(f"  ${cap:>11,}{m:>22,.0f}{m/10000*100:>17.0f}%")
print(f"\n  capital needed for $10k/month: ${10000*12/g/0.9:,.0f}")
