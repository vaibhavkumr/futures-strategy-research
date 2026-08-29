"""His described method: 15m key zones + first-60-min reversal + trailing exit.

  1. Mark key reversal zones on the 15-MINUTE chart (prior swing highs/lows,
     prior day high/low, overnight high/low)
  2. Trade ONLY 09:30-10:30 ET
  3. Wait for price to reach a zone and show a reversal (close back away)
  4. Stop beyond the zone
  5. TRAIL the stop and scale out as it moves -- he stresses this

Costs: 0.5bp round trip (real index-futures cost, not the 2bp I wrongly
used earlier today).
"""
import numpy as np, pandas as pd, glob, os
OPEN, WIN_END = 9*60+30, 10*60+30
COST_BP = 0.5
MK={"S&P 500":"usa500idxusd","NASDAQ":"usatechidxusd","DOW":"usa30idxusd"}

def load(slug):
    fs=[f for f in glob.glob(f"download/{slug}-m5-bid-*.csv") if "2026-07-24" in f]
    d=pd.read_csv(max(fs,key=os.path.getsize)); d.columns=[c.lower() for c in d.columns]
    ts=d["timestamp"]
    idx=pd.to_datetime(ts,unit="ms",utc=True) if pd.api.types.is_numeric_dtype(ts) \
        else pd.to_datetime(ts,utc=True)
    d.index=idx.dt.tz_convert("America/New_York")
    return d[["open","high","low","close"]].astype(float).sort_index()

def zones(d5):
    """Key levels from the 15m chart, all known BEFORE the open."""
    d15=d5.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    w=5
    sh=(d15["high"].rolling(w,center=True).max()==d15["high"])
    sl=(d15["low"].rolling(w,center=True).min()==d15["low"])
    hi=pd.Series(np.where(sh,d15["high"],np.nan),index=d15.index).ffill().shift(3)
    lo=pd.Series(np.where(sl,d15["low"],np.nan),index=d15.index).ffill().shift(3)
    day=d5.index.normalize()
    pdh=d5.groupby(day)["high"].max().shift(1); pdl=d5.groupby(day)["low"].min().shift(1)
    m=d5.index.hour*60+d5.index.minute
    on=(m<OPEN)
    onh=d5[on].groupby(day[on])["high"].max(); onl=d5[on].groupby(day[on])["low"].min()
    Z=pd.DataFrame(index=d5.index)
    Z["h15"]=hi.reindex(d5.index,method="ffill"); Z["l15"]=lo.reindex(d5.index,method="ffill")
    ds=pd.Series(day,index=d5.index)
    Z["pdh"]=ds.map(pdh); Z["pdl"]=ds.map(pdl)
    Z["onh"]=ds.map(onh); Z["onl"]=ds.map(onl)
    return Z

def bt(d, trail=True, tmult=2.0):
    Z=zones(d)
    o,h,l,c=(d[x].values for x in ("open","high","low","close"))
    tr=pd.concat([d["high"]-d["low"],(d["high"]-d["close"].shift()).abs(),
                  (d["low"]-d["close"].shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean().bfill().values
    idx=d.index; n=len(d); m=idx.hour*60+idx.minute
    H=Z[["h15","pdh","onh"]].values; L=Z[["l15","pdl","onl"]].values
    rows=[]; last_day=None
    for i in range(30,n-2):
        if not (OPEN<=m[i]<WIN_END): continue
        day=idx[i].date()
        if day==last_day: continue                  # one trade per morning
        a=atr[i]
        if not np.isfinite(a) or a<=0: continue
        sgn=None
        # price reached a RESISTANCE zone and closed back below -> short
        for z in H[i]:
            if np.isfinite(z) and h[i]>=z and c[i]<z and (h[i]-z)<1.0*a:
                sgn=-1; lvl=z; break
        if sgn is None:
            for z in L[i]:
                if np.isfinite(z) and l[i]<=z and c[i]>z and (z-l[i])<1.0*a:
                    sgn=1; lvl=z; break
        if sgn is None: continue
        e=c[i]
        stop=(l[i]-0.1*a) if sgn>0 else (h[i]+0.1*a)
        risk=abs(e-stop)
        if risk<0.20*a or risk>2.5*a: continue
        tgt=e+sgn*tmult*risk; R=None; peak=e
        for k in range(i+1,min(i+80,n)):
            if idx[k].date()!=day: R=(c[k-1]-e)*sgn/risk; break
            if sgn>0: peak=max(peak,h[k])
            else:     peak=min(peak,l[k])
            if trail:  # trail to breakeven after 1R
                if (peak-e)*sgn >= risk: stop = max(stop,e) if sgn>0 else min(stop,e)
            if (l[k]<=stop) if sgn>0 else (h[k]>=stop):
                R=(stop-e)*sgn/risk - 0.05*a/risk; break
            if (h[k]>=tgt) if sgn>0 else (l[k]<=tgt): R=tmult; break
        if R is None: continue
        cost_R = (COST_BP/1e4)*e/risk
        rows.append(dict(R=R-cost_R, side="long" if sgn>0 else "short",
                         risk_atr=risk/a))
        last_day=day
    return pd.DataFrame(rows)

def st(x,lab):
    x=np.asarray(x,float)
    if len(x)<30: print(f"  {lab:<26} n={len(x)}"); return
    m,se=x.mean(),x.std(ddof=1)/np.sqrt(len(x))
    print(f"  {lab:<26} n={len(x):<5} win {(x>0).mean()*100:5.1f}%  "
          f"expR {m:+.3f}  t={m/se:+6.2f}")

if __name__=="__main__":
    for nm,slug in MK.items():
        d=load(slug)
        print(f"\n{'='*66}\n{nm}\n{'='*66}")
        for trail,tm,lab in ((True,2.0,"trail to BE, 2R target"),
                             (False,2.0,"fixed stop, 2R target"),
                             (True,1.0,"trail to BE, 1R target")):
            t=bt(d,trail=trail,tmult=tm)
            st(t.R,lab)
