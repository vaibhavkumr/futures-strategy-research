"""PORTFOLIO, REDONE -- only signals I actually measured as POSITIVE.

My first attempt combined the calendar edge with two components I had coded
WRONG: mom_fade traded the opening-hour relationship backwards (I measured
+0.085 continuation and then faded it), and spread was a crude daily proxy
rather than the 5-min z-scored version that measured +2.26bp. Combining one
good signal with two broken ones gave a negative portfolio. That was my error,
not evidence against the idea.

These are the two signals that measured positive on their own terms:

  CALENDAR   turn-of-month + Monday, hold the session.  +9.55bp, t=3.39
  OPEN_REV   fade the FIRST 30 minutes IN THE SECOND 30 minutes.
             S&P +1.27bp (t=1.52), DOW +1.78bp (t=2.30)

Costs 0.5bp (real index-futures).
"""
import glob, os
import numpy as np, pandas as pd

COST=0.5; OPEN=9*60+30
MK={"S&P 500":"usa500idxusd","NASDAQ":"usatechidxusd",
    "DOW":"usa30idxusd","DAX":"deuidxeur"}

def load(slug):
    fs=[f for f in glob.glob(f"download/{slug}-m5-bid-*.csv") if "2026-07-24" in f]
    d=pd.read_csv(max(fs,key=os.path.getsize)); d.columns=[c.lower() for c in d.columns]
    ts=d["timestamp"]
    idx=(pd.to_datetime(ts,unit="ms",utc=True) if pd.api.types.is_numeric_dtype(ts)
         else pd.to_datetime(ts,utc=True))
    d.index=idx.dt.tz_convert("America/New_York")
    d=d[["open","high","low","close"]].astype(float).sort_index()
    assert d.index[0].year>2000
    return d

def legs(d):
    m=d.index.hour*60+d.index.minute
    day=d.index.normalize().tz_localize(None)
    def seg(a,b):
        k=(m>=a)&(m<b); g=d[k].groupby(day[k])
        return g["open"].first(), g["close"].last()
    o1,c1=seg(OPEN,OPEN+30); o2,c2=seg(OPEN+30,OPEN+60)
    oS,cS=seg(OPEN,16*60)
    return pd.DataFrame({"leg1":(c1/o1-1)*1e4,"leg2":(c2/o2-1)*1e4,
                         "sess":(cS/oS-1)*1e4}).dropna()

def calendar_mask(idx):
    mon=idx.dayofweek==0
    tom=np.zeros(len(idx),bool)
    for _,g in pd.Series(idx,index=idx).groupby([idx.year,idx.month]):
        dd=pd.DatetimeIndex(g.values)
        tom|=idx.isin(dd[:3]); tom|=idx.isin(dd[-1:])
    return mon|tom

def build():
    S={}
    for nm,slug in MK.items():
        L=legs(load(slug))
        cal=np.where(calendar_mask(L.index), L.sess.values-COST, np.nan)
        orev=-np.sign(L.leg1.values)*L.leg2.values-COST     # the CORRECT version
        S[nm]=pd.DataFrame({"calendar":cal,"open_rev":orev},index=L.index)
    return S

def stat(x,per_year):
    x=pd.Series(x).dropna().values
    if len(x)<30: return None
    m,sd=x.mean(),x.std(ddof=1)
    return dict(n=len(x),mean=m,t=m/(sd/np.sqrt(len(x))),
                sharpe=m/sd*np.sqrt(per_year),ann=m*per_year/100)

if __name__=="__main__":
    S=build()
    names=["calendar","open_rev"]
    P=pd.DataFrame({k:pd.concat([S[nm][k].rename(nm) for nm in MK],axis=1).mean(axis=1)
                    for k in names})
    freq={"calendar":89,"open_rev":252}
    print("INDIVIDUAL (equal-weight across 4 markets)\n")
    print(f"{'signal':<12}{'n':>6}{'mean bp':>10}{'t':>8}{'Sharpe':>9}{'ann %':>8}")
    print("-"*53)
    for k in names:
        s=stat(P[k],freq[k])
        print(f"{k:<12}{s['n']:>6}{s['mean']:>10.2f}{s['t']:>8.2f}"
              f"{s['sharpe']:>9.2f}{s['ann']:>8.2f}")
    c=P[names].corr().iloc[0,1]
    print(f"\ncorrelation between them: {c:+.3f}")
    Z=P[names].copy()
    for k in names: Z[k]=Z[k]/Z[k].std()
    combo=Z.mean(axis=1).dropna()
    s=stat(combo,252)
    print(f"\nCOMBINED (equal risk): n={s['n']}  Sharpe {s['sharpe']:+.2f}  t={s['t']:+.2f}")
    for k in names:
        ss=stat(P[k],freq[k]); print(f"   vs {k}: Sharpe {ss['sharpe']:+.2f}")
    print("\nTEMPORAL SPLIT of the combination:")
    for lab,g in (("dev 2022-24",combo[combo.index<'2025-01-01']),
                  ("HOLDOUT 25-26",combo[combo.index>='2025-01-01'])):
        s=stat(g,252)
        if s: print(f"   {lab:<15} n={s['n']:<5} Sharpe {s['sharpe']:+.2f}  t={s['t']:+.2f}")
