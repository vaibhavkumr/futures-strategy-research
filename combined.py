"""THE TWO VERIFIED EDGES, COMBINED.

  mom_12m   cross-sectional momentum, 26 ETFs, weekly rebalance.
            Beats equal-weight buy&hold in DEV (+1.27% CAGR) and HOLDOUT
            (+4.51%), with ~8 points less drawdown in both.
  calendar  turn-of-month + Monday on index futures, daily.
            Excess +10.15bp, t=2.07 event-level, 3 of 4 markets.

Different assets, different horizons, different mechanisms -- so the question
is whether their losses actually come at different times. If they do, the
combination should show materially lower drawdown than either alone at the
same return, which is the only free improvement available.

Both are measured NET of costs and against their own honest benchmarks.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import momentum_verify as MV
import signal_lab as L

def calendar_daily():
    D={nm:L.daily(sl) for nm,sl in L.MK.items()}
    R=pd.concat([D[nm].r.rename(nm) for nm in L.MK],axis=1).mean(axis=1)
    mask=L.s_calendar(D["S&P 500"]).astype(bool).reindex(R.index).fillna(False)
    # excess over the unconditional mean, so it is not just long exposure
    base=R.mean()
    r=pd.Series(np.where(mask,(R-base).values-L.COST,0.0),index=R.index)
    return (r/1e4).dropna()

def met(r,lab,ann=252):
    r=pd.Series(r).dropna()
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()*100
    yrs=len(r)/ann; cagr=(eq.iloc[-1]**(1/yrs)-1)*100
    sh=r.mean()/r.std(ddof=1)*np.sqrt(ann)
    cal=cagr/abs(dd) if dd else np.nan
    print(f"  {lab:<30} CAGR {cagr:>6.2f}%  maxDD {dd:>6.1f}%  "
          f"Sharpe {sh:>5.2f}  Calmar {cal:>5.2f}")
    return sh,cal

if __name__=="__main__":
    px=MV.load_universe()
    mom=MV.backtest(px.loc["2018-01-01":])       # holdout period only
    cal=calendar_daily()
    idx=mom.index.intersection(cal.index)
    mom=mom.reindex(idx).fillna(0); cal=cal.reindex(idx).fillna(0)
    print(f"overlapping period: {idx.min():%Y-%m-%d} to {idx.max():%Y-%m-%d} "
          f"({len(idx)} days)\n")
    print("STANDALONE")
    s1,c1=met(mom,"mom_12m")
    s2,c2=met(cal,"calendar (excess)")
    print(f"\n  correlation between them: {mom.corr(cal):+.3f}\n")
    print("COMBINED (equal risk)")
    z1=mom/mom.std(); z2=cal/cal[cal!=0].std()
    for w in (0.5,0.6,0.7,0.8):
        port=(w*z1+(1-w)*z2)*mom.std()
        met(port,f"{int(w*100)}% mom / {int((1-w)*100)}% cal")
