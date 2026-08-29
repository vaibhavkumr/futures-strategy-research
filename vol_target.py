"""CAN WE GET MORE RETURN PER UNIT OF DRAWDOWN?

Leverage scales return and drawdown together -- that ratio is fixed by the
strategy. But the RATIO itself can be improved without finding a new edge:

  VOLATILITY TARGETING -- size inversely to recent realised vol. Drawdowns
  cluster in high-vol periods, so cutting size there removes the worst of
  them while keeping most of the return. This is standard at managed-futures
  funds and it is the one legitimate way to raise Calmar without new alpha.

  DRAWDOWN CONTROL -- cut size after losses, restore after recovery.

Measured on the calendar edge: return, max drawdown, and CALMAR (return per
unit of drawdown), which is the number that decides how hard you can push.
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

def sessions(slug):
    d=load(slug); m=d.index.hour*60+d.index.minute
    k=(m>=OPEN)&(m<16*60); d=d[k]
    day=d.index.normalize().tz_localize(None)
    g=d.groupby(day)
    return pd.DataFrame({"r":(g["close"].last()/g["open"].first()-1)*1e4})

def cal_mask(idx):
    mon=idx.dayofweek==0
    tom=np.zeros(len(idx),bool)
    for _,g in pd.Series(idx,index=idx).groupby([idx.year,idx.month]):
        dd=pd.DatetimeIndex(g.values)
        tom|=idx.isin(dd[:3]); tom|=idx.isin(dd[-1:])
    return mon|tom

def metrics(r, per_year=252):
    r=pd.Series(r).dropna()
    eq=(1+r/1e4).cumprod()
    dd=(eq/eq.cummax()-1).min()*100
    yrs=len(r)/per_year
    ann=(eq.iloc[-1]**(1/yrs)-1)*100
    sd=r.std(ddof=1)
    sh=r.mean()/sd*np.sqrt(per_year) if sd>0 else 0
    calmar=ann/abs(dd) if dd else np.nan
    return dict(ann=ann,dd=dd,sharpe=sh,calmar=calmar,n=len(r))

if __name__=="__main__":
    S={nm:sessions(sl) for nm,sl in MK.items()}
    R=pd.concat([S[nm].r.rename(nm) for nm in MK],axis=1).mean(axis=1).dropna()
    fire=cal_mask(R.index)
    base=pd.Series(np.where(fire,R.values-COST,0.0),index=R.index)

    # realised vol of the strategy's own market, known BEFORE the day
    vol=R.rolling(20).std().shift(1)
    med=vol.rolling(252).median().shift(1)

    print(f"{'variant':<34}{'ann %':>8}{'maxDD %':>9}{'Sharpe':>8}{'Calmar':>8}")
    print("-"*67)
    m=metrics(base); print(f"{'1x, no vol targeting':<34}{m['ann']:>8.2f}"
        f"{m['dd']:>9.1f}{m['sharpe']:>8.2f}{m['calmar']:>8.2f}")

    for tgt in (0.75,1.0,1.25):
        size=(med/vol).clip(0.25,3.0).fillna(1.0)*tgt
        vt=base*size
        m=metrics(vt)
        print(f"{'vol-targeted x'+str(tgt):<34}{m['ann']:>8.2f}"
              f"{m['dd']:>9.1f}{m['sharpe']:>8.2f}{m['calmar']:>8.2f}")

    print()
    # best vol-target, then scale up to a drawdown budget
    size=(med/vol).clip(0.25,3.0).fillna(1.0)
    vt=base*size
    print("SCALING THE VOL-TARGETED VERSION to a drawdown budget:")
    print(f"{'leverage':<12}{'ann %':>9}{'maxDD %':>10}{'on $5,000':>12}{'on $50k prop':>14}")
    print("-"*57)
    for L in (1,2,3,5,8):
        m=metrics(vt*L)
        print(f"{str(L)+'x':<12}{m['ann']:>9.1f}{m['dd']:>10.1f}"
              f"{5000*m['ann']/100:>11,.0f}{50000*m['ann']/100:>14,.0f}")
