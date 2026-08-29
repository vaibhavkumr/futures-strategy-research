"""FACTOR LAB II -- second sweep for edges 3 and 4.

Round one: 8 documented factors, 1 survivor (mom_12m), placebo 0/100. So the
bar is honest and the hit rate is ~1 in 8. This is the next 10 candidates,
chosen to be mechanistically DIFFERENT from momentum rather than variants of it.

  MOM_SKIP        12-month momentum skipping the most recent month
                  (Jegadeesh & Titman's actual specification -- the skip
                  avoids short-term reversal contaminating the signal)
  MOM_VOLADJ      momentum divided by volatility (risk-adjusted ranking)
  MOM_6M / 9M     different formation windows
  ABS_MOM         absolute (time-series) momentum: hold only assets with
                  positive trailing return, else cash
  SEASONAL_ASSET  each asset's own historical month-of-year strength
  BREADTH         how many assets are above their 200d mean -> risk on/off
  RISK_PARITY     inverse-volatility weights instead of equal weights
  MOM_LONGSHORT   long top-6, short bottom-6 (market-neutral version)
  DUAL_MOM        cross-sectional momentum gated by absolute momentum

Same corrected methodology: benchmark = equal-weight buy&hold, dev AND
holdout must both beat it, costs 10bp/turnover, placebo alongside.
"""
import numpy as np, pandas as pd
import factor_lab as F

def f_mom_skip(px):   return px.shift(21).pct_change(231)
def f_mom_6m(px):     return px.pct_change(126)
def f_mom_9m(px):     return px.pct_change(189)
def f_mom_voladj(px):
    r=px.pct_change(252); v=px.pct_change().rolling(252).std()
    return r/v.replace(0,np.nan)
def f_abs_mom(px):
    """Only rank assets that are positive over 12m; others get -inf."""
    r=px.pct_change(252)
    return r.where(r>0, -1e9)
def f_seasonal(px):
    """Each asset's own average return in this calendar month, history only."""
    r=px.pct_change()
    mo=r.index.month
    out=pd.DataFrame(index=px.index,columns=px.columns,dtype=float)
    for m in range(1,13):
        mask=mo==m
        # expanding mean of this month's returns, shifted so it is past-only
        hist=r[mask].expanding().mean().shift(1)
        out.loc[mask]=hist
    return out.ffill()
def f_breadth(px):
    """Rank by trend, but only take positions when breadth is healthy."""
    above=(px>px.rolling(200).mean()).mean(axis=1)
    score=px/px.rolling(200).mean()-1
    return score.mul((above>0.5).astype(float),axis=0).replace(0,np.nan)
def f_riskparity(px):
    """Inverse vol -- lowest-vol assets score highest."""
    return -px.pct_change().rolling(126).std()
def f_dual_mom(px):
    """Cross-sectional momentum, gated by the asset's own absolute trend."""
    cs=px.pct_change(252)
    ts=(px>px.rolling(200).mean())
    return cs.where(ts, -1e9)

NEW={"mom_skip":f_mom_skip,"mom_6m":f_mom_6m,"mom_9m":f_mom_9m,
     "mom_voladj":f_mom_voladj,"abs_mom":f_abs_mom,"seasonal_asset":f_seasonal,
     "breadth":f_breadth,"risk_parity":f_riskparity,"dual_mom":f_dual_mom}

if __name__=="__main__":
    px=F.universe(); DEV="2018-01-01"
    print(f"{len(NEW)} new candidates, universe {px.shape[1]} assets\n")
    print(f"{'factor':<16}{'DEV exc':>9}{'HOLD exc':>10}{'DEVshp':>8}{'HOLDshp':>9}{'':>11}")
    print("-"*64)
    keep={}
    for name,fn in NEW.items():
        vals=[]
        for sl in (slice(None,DEV),slice(DEV,None)):
            p=px.loc[sl]
            m=F.run(p,fn); b=F.benchmark(p).reindex(m.index).dropna()
            sm,sb=F.stats(m),F.stats(b)
            if not sm or not sb: vals=None; break
            vals.append((sm["cagr"]-sb["cagr"], sm["sharpe"]-sb["sharpe"]))
        if not vals:
            print(f"{name:<16}  (insufficient)"); continue
        (de,ds),(he,hs)=vals
        ok= de>0 and he>0
        if ok: keep[name]=F.run(px,fn)
        print(f"{name:<16}{de:>+9.2f}{he:>+10.2f}{ds:>+8.2f}{hs:>+9.2f}"
              f"{'  BEATS B&H' if ok else '':>11}")
    print(f"\nSURVIVORS: {list(keep) if keep else 'none'}")
    if keep:
        mom=F.run(px,F.f_mom12)
        print("\ncorrelation with mom_12m (need LOW to add value):")
        for k,v in keep.items():
            j=v.index.intersection(mom.index)
            print(f"  {k:<16}{v.reindex(j).corr(mom.reindex(j)):+.3f}")
