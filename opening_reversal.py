"""His testable claim: the first 60 minutes pushes one way, then REVERSES.

Interesting because it contradicts Gao/Han/Li/Zhou (JFE 2018), who found the
first half-hour predicts CONTINUATION into the close. My replication of that
paper found the slope had flipped NEGATIVE on 2022-2026 -- which is what this
claim predicts. So this is worth testing properly rather than dismissing.

Test: split the opening hour. Does the first leg predict a REVERSAL in the
second leg? And does the opening hour predict a reversal over the rest of
the session?
"""
import numpy as np, pandas as pd, glob, os
MK={"S&P 500":"usa500idxusd","NASDAQ":"usatechidxusd",
    "DOW":"usa30idxusd","GOLD":"xauusd"}
OPEN=9*60+30

def load(slug):
    fs=[f for f in glob.glob(f"download/{slug}-m5-bid-*.csv") if "2026-07-24" in f]
    d=pd.read_csv(max(fs,key=os.path.getsize)); d.columns=[c.lower() for c in d.columns]
    ts=d["timestamp"]
    idx=pd.to_datetime(ts,unit="ms",utc=True) if pd.api.types.is_numeric_dtype(ts) \
        else pd.to_datetime(ts,utc=True)
    d.index=idx.dt.tz_convert("America/New_York")
    return d[["open","high","low","close"]].astype(float).sort_index()

def legs(d):
    m=d.index.hour*60+d.index.minute
    day=d.index.normalize()
    def seg(a,b):
        k=(m>=a)&(m<b); g=d[k].groupby(day[k])
        return g["open"].first(), g["close"].last()
    o1,c1=seg(OPEN,OPEN+30)        # 9:30-10:00  first leg
    o2,c2=seg(OPEN+30,OPEN+60)     # 10:00-10:30 second leg
    o3,c3=seg(OPEN,OPEN+60)        # full opening hour
    o4,c4=seg(OPEN+60,16*60)       # rest of session
    return pd.DataFrame({"leg1":(c1/o1-1)*1e4,"leg2":(c2/o2-1)*1e4,
                         "hour":(c3/o3-1)*1e4,"rest":(c4/o4-1)*1e4}).dropna()

def rep(x,lab,cost=0.0):
    x=np.asarray(x,float)-cost
    m,se=x.mean(),x.std(ddof=1)/np.sqrt(len(x))
    print(f"    {lab:<34} n={len(x):<5} {m:+7.2f}bp  t={m/se:+6.2f}  "
          f"win {(x>0).mean()*100:5.1f}%")

if __name__=="__main__":
    for nm,slug in MK.items():
        L=legs(load(slug))
        print(f"\n{'='*74}\n{nm}   ({len(L)} sessions)\n{'='*74}")
        r=np.corrcoef(L.leg1,L.leg2)[0,1]
        t=r*np.sqrt(len(L)-2)/np.sqrt(max(1-r**2,1e-12))
        print(f"  corr(first 30min, second 30min) = {r:+.4f}  t={t:+5.2f}"
              f"   -> {'REVERSAL' if r<0 else 'continuation'}")
        r2=np.corrcoef(L.hour,L.rest)[0,1]
        t2=r2*np.sqrt(len(L)-2)/np.sqrt(max(1-r2**2,1e-12))
        print(f"  corr(opening hour, rest of day) = {r2:+.4f}  t={t2:+5.2f}"
              f"   -> {'REVERSAL' if r2<0 else 'continuation'}")
        print("  trading it (fade the first leg in the second):")
        rep(-np.sign(L.leg1)*L.leg2, "gross")
        rep(-np.sign(L.leg1)*L.leg2, "after 2bp cost", cost=2.0)
        print("  trading it (fade the opening hour, rest of day):")
        rep(-np.sign(L.hour)*L.rest, "gross")
        rep(-np.sign(L.hour)*L.rest, "after 2bp cost", cost=2.0)
