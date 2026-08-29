"""THE THREE UNTESTED CLAIMS FROM THE CHART GUIDE.

Everything else in that article I have already measured: candlestick patterns
(directional accuracy 47.6-51.3%), multi-timeframe confluence (fair odds),
timeframe selection (1s to daily, all fair odds), position sizing (multiplies
edge, never creates it).

These three I never tested:

  1. VOLUME CONFIRMATION -- "Always confirm volume before trading a pattern."
     "Skipping volume confirmation [is a] major cause of failed breakouts."
     Directly testable: same patterns, split by whether volume was elevated.

  2. VOLUME PROFILE / POINT OF CONTROL -- support and resistance from where
     volume actually traded rather than where price turned. The claim is that
     high-volume nodes act as real levels because that is where institutions
     transacted.

  3. HEIKIN-ASHI -- smoothed candles said to "help find trends."

Tested on index CFDs (tick volume, 0.5bp costs) and BTC (real exchange
volume, but 20bp costs).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

MK = {"S&P 500": "usa500idxusd", "NASDAQ": "usatechidxusd",
      "DOW": "usa30idxusd", "DAX": "deuidxeur"}
COST_BP = 0.5


def load_cfd(slug):
    fs = [f for f in glob.glob(f"download/{slug}-m5-bid-*.csv") if "2026-07-24" in f]
    d = pd.read_csv(max(fs, key=os.path.getsize))
    d.columns = [c.lower() for c in d.columns]
    ts = d["timestamp"]
    idx = (pd.to_datetime(ts, unit="ms", utc=True)
           if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts, utc=True))
    d.index = idx.dt.tz_convert("America/New_York")
    d = d[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
    assert d.index[0].year > 2000
    return d


def heikin_ashi(d):
    ha = pd.DataFrame(index=d.index)
    ha["close"] = (d.open + d.high + d.low + d.close) / 4
    o = [d.open.iloc[0]]
    for i in range(1, len(d)):
        o.append((o[-1] + ha["close"].iloc[i-1]) / 2)
    ha["open"] = o
    ha["high"] = pd.concat([d.high, ha.open, ha.close], axis=1).max(axis=1)
    ha["low"] = pd.concat([d.low, ha.open, ha.close], axis=1).min(axis=1)
    return ha


def poc_levels(d, lookback=78, bins=40):
    """Point of Control: the price with the most volume over the lookback."""
    px = d.close.values
    vol = d.volume.values
    hi, lo = d.high.values, d.low.values
    n = len(d)
    poc = np.full(n, np.nan)
    for i in range(lookback, n):
        s = slice(i-lookback, i)
        top, bot = hi[s].max(), lo[s].min()
        if top <= bot:
            continue
        edges = np.linspace(bot, top, bins+1)
        who = np.clip(np.digitize(px[s], edges) - 1, 0, bins-1)
        agg = np.bincount(who, weights=vol[s], minlength=bins)
        poc[i] = (edges[agg.argmax()] + edges[agg.argmax()+1]) / 2
    return poc


def feats(d):
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    tr = pd.concat([d.high-d.low, (d.high-d.close.shift()).abs(),
                    (d.low-d.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().bfill().values
    v = d.volume.values
    vma = pd.Series(v).rolling(48).mean().bfill().values
    vrel = v / np.maximum(vma, 1e-9)
    return o, h, l, c, atr, vrel


def pat_engulf(o, h, l, c, a, i):
    if i < 2:
        return 0
    if c[i] > o[i] and c[i-1] < o[i-1] and c[i] > o[i-1] and (c[i]-o[i]) > 0.6*a[i]:
        return 1
    if c[i] < o[i] and c[i-1] > o[i-1] and c[i] < o[i-1] and (o[i]-c[i]) > 0.6*a[i]:
        return -1
    return 0


def pat_break(o, h, l, c, a, i, look=20):
    """Breakout -- the article's specific volume claim is about these."""
    if i < look+1:
        return 0
    hi, lo = h[i-look:i].max(), l[i-look:i].min()
    if c[i] > hi:
        return 1
    if c[i] < lo:
        return -1
    return 0


def sim(d, patfn, vmin=0.0, vmax=99.0, tmult=2.0, cost=COST_BP, sess=(570,960)):
    o,h,l,c,a,vrel = feats(d)
    idx=d.index; m=idx.hour*60+idx.minute; n=len(d)
    out=[]; busy=-1
    for i in range(60,n-2):
        if i<busy or not (sess[0]<=m[i]<sess[1]): continue
        if not (vmin <= vrel[i] < vmax): continue
        s=patfn(o,h,l,c,a,i)
        if s==0: continue
        e=c[i]; stop=(min(l[i-1],l[i])-0.1*a[i]) if s>0 else (max(h[i-1],h[i])+0.1*a[i])
        risk=abs(e-stop)
        if risk<0.25*a[i] or risk>3*a[i]: continue
        tgt=e+s*tmult*risk; R=None
        for k in range(i+1,min(i+80,n)):
            if (l[k]<=stop) if s>0 else (h[k]>=stop): R=-1.0-0.05*a[i]/risk; busy=k; break
            if (h[k]>=tgt) if s>0 else (l[k]<=tgt): R=tmult; busy=k; break
        if R is None: continue
        out.append(R-(cost/1e4)*e/risk)
    return np.array(out)


def st(x,lab):
    if len(x)<30: print(f"  {lab:<32} n={len(x)}"); return
    m,se=x.mean(),x.std(ddof=1)/np.sqrt(len(x))
    print(f"  {lab:<32} n={len(x):<5} win {(x>0).mean()*100:5.1f}%  "
          f"expR {m:+.3f}  t={m/se:+6.2f}")


if __name__=="__main__":
    D={nm:load_cfd(sl) for nm,sl in MK.items()}
    print("="*70)
    print("1. VOLUME CONFIRMATION -- does elevated volume improve patterns?")
    print("="*70)
    for pname,pf in (("BREAKOUT",pat_break),("ENGULFING",pat_engulf)):
        print(f"\n  --- {pname} ---")
        for lo,hi,lab in ((0,0.8,"LOW volume  (<0.8x avg)"),
                          (0.8,1.2,"normal      (0.8-1.2x)"),
                          (1.2,2.0,"elevated    (1.2-2x)"),
                          (2.0,99,"HIGH volume (>2x avg)")):
            pool=[sim(d,pf,lo,hi) for d in D.values()]
            st(np.concatenate([x for x in pool if len(x)]), lab)

    print("\n"+"="*70)
    print("2. POINT OF CONTROL -- do high-volume nodes act as real levels?")
    print("="*70)
    pool_at=[];pool_away=[]
    for nm,d in D.items():
        poc=poc_levels(d)
        o,h,l,c,a,vrel=feats(d)
        near=np.abs(c-poc)<0.3*a
        idx=d.index; m=idx.hour*60+idx.minute
        for i in range(80,len(d)-2):
            if not (570<=m[i]<960) or not np.isfinite(poc[i]): continue
            s=pat_break(o,h,l,c,a,i)
            if s==0: continue
            e=c[i]; stop=(min(l[i-1],l[i])-0.1*a[i]) if s>0 else (max(h[i-1],h[i])+0.1*a[i])
            risk=abs(e-stop)
            if risk<0.25*a[i] or risk>3*a[i]: continue
            tgt=e+s*2.0*risk; R=None
            for k in range(i+1,min(i+80,len(d))):
                if (l[k]<=stop) if s>0 else (h[k]>=stop): R=-1.0-0.05*a[i]/risk; break
                if (h[k]>=tgt) if s>0 else (l[k]<=tgt): R=2.0; break
            if R is None: continue
            (pool_at if near[i] else pool_away).append(R-(COST_BP/1e4)*e/risk)
    st(np.array(pool_at),"breakout NEAR the POC")
    st(np.array(pool_away),"breakout AWAY from POC")

    print("\n"+"="*70)
    print("3. HEIKIN-ASHI -- do smoothed candles help?")
    print("="*70)
    for nm,d in list(D.items())[:2]:
        ha=heikin_ashi(d); ha["volume"]=d.volume
        st(sim(d,pat_engulf),f"{nm} engulfing, normal candles")
        st(sim(ha,pat_engulf),f"{nm} engulfing, Heikin-Ashi")
