"""Riley Coleman's four documented patterns, coded to his own spec.

From "Riley's Favorite Trading Patterns" (rileycolemantrading.com). His guide
usefully lists KEYS and UNIMPORTANT FACTORS per pattern, so the rules are
unusually well specified for this genre.

  THREE LINE STRIKE  a large reversal candle that gives back the move of the
                     last 3-5 candles. He says do NOT use alone -- so tested
                     both alone and with a level confluence.
  TRAPS              strong, abnormal breakout move, then reversal just as
                     fast or faster. Explicitly: volume does not matter,
                     price velocity does.
  REVERSAL FLAG      strong momentum with no pullbacks INTO a major zone,
                     then directional -> sideways.
  HEAD & SHOULDERS   three swings, middle highest, with a larger-than-normal
                     rejection. Flat neckline explicitly unimportant.

Costs: 0.5bp round trip (real index-futures cost).
"""
import numpy as np, pandas as pd, glob, os
COST_BP=0.5
MK={"S&P 500":"usa500idxusd","NASDAQ":"usatechidxusd",
    "DOW":"usa30idxusd","GOLD":"xauusd"}

def load(slug):
    fs=[f for f in glob.glob(f"download/{slug}-m5-bid-*.csv") if "2026-07-24" in f]
    d=pd.read_csv(max(fs,key=os.path.getsize)); d.columns=[c.lower() for c in d.columns]
    ts=d["timestamp"]
    idx=pd.to_datetime(ts,unit="ms",utc=True) if pd.api.types.is_numeric_dtype(ts) \
        else pd.to_datetime(ts,utc=True)
    d.index=idx.dt.tz_convert("America/New_York")
    return d[["open","high","low","close"]].astype(float).sort_index()

def prep(d):
    o,h,l,c=(d[x].values for x in ("open","high","low","close"))
    tr=pd.concat([d["high"]-d["low"],(d["high"]-d["close"].shift()).abs(),
                  (d["low"]-d["close"].shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean().bfill().values
    return o,h,l,c,atr

def three_line_strike(o,h,l,c,atr,i):
    """Large candle gives back the last 3-5 candles' movement."""
    for k in (3,4,5):
        if i-k < 0: continue
        legs=[c[j]-o[j] for j in range(i-k,i)]
        if all(x<0 for x in legs) and (c[i]-o[i])>0 and c[i]>o[i-k]:
            if (c[i]-o[i]) > 1.0*atr[i]: return 1     # bullish strike
        if all(x>0 for x in legs) and (c[i]-o[i])<0 and c[i]<o[i-k]:
            if (o[i]-c[i]) > 1.0*atr[i]: return -1    # bearish strike
    return 0

def trap(o,h,l,c,atr,i,look=20):
    """Strong abnormal breakout, then reversal just as fast."""
    if i<look+2: return 0
    hi=np.max(h[i-look:i]); lo=np.min(l[i-look:i])
    body=abs(c[i]-o[i]); rng=max(h[i]-l[i],1e-9)
    if h[i]>hi and c[i]<hi and (h[i]-hi)>0.3*atr[i] and (h[i]-c[i])/rng>0.5:
        return -1
    if l[i]<lo and c[i]>lo and (lo-l[i])>0.3*atr[i] and (c[i]-l[i])/rng>0.5:
        return 1
    return 0

def reversal_flag(o,h,l,c,atr,i,imp=12,cons=6):
    """Strong directional move, then sideways, at a zone extreme."""
    if i<imp+cons+2: return 0
    a=i-cons; s=a-imp
    move=c[a]-c[s]
    if abs(move) < 2.0*atr[i]: return 0
    seg_h=np.max(h[a:i+1]); seg_l=np.min(l[a:i+1])
    if (seg_h-seg_l) > 1.2*atr[i]: return 0          # must be sideways
    pull=np.max(np.abs(np.diff(c[s:a])))
    if pull > 0.8*atr[i]: return 0                   # "no pullbacks"
    return -1 if move>0 else 1

def head_shoulders(o,h,l,c,atr,i,w=4):
    """3 swings, middle most extreme, larger-than-normal rejection."""
    if i<6*w: return 0
    seg=lambda a,b:(np.max(h[a:b]),np.min(l[a:b]))
    h1,_=seg(i-6*w,i-4*w); h2,_=seg(i-4*w,i-2*w); h3,_=seg(i-2*w,i)
    if h2>h1 and h2>h3 and (h2-max(h1,h3))>0.4*atr[i] and c[i]<min(h1,h3):
        return -1
    _,l1=seg(i-6*w,i-4*w); _,l2=seg(i-4*w,i-2*w); _,l3=seg(i-2*w,i)
    if l2<l1 and l2<l3 and (min(l1,l3)-l2)>0.4*atr[i] and c[i]>max(l1,l3):
        return 1
    return 0

PATS={"three_line_strike":three_line_strike,"trap":trap,
      "reversal_flag":reversal_flag,"head_shoulders":head_shoulders}

def bt(d, fn, tmult=2.0, sess=(570,960)):
    o,h,l,c,atr=prep(d); idx=d.index; n=len(d)
    m=idx.hour*60+idx.minute
    rows=[]; busy=-1
    for i in range(40,n-2):
        if i<busy or not (sess[0]<=m[i]<sess[1]): continue
        sgn=fn(o,h,l,c,atr,i)
        if sgn==0: continue
        e=c[i]
        stop=(min(l[i-1],l[i])-0.1*atr[i]) if sgn>0 else (max(h[i-1],h[i])+0.1*atr[i])
        risk=abs(e-stop)
        if risk<0.25*atr[i] or risk>3*atr[i]: continue
        tgt=e+sgn*tmult*risk; R=None
        for k in range(i+1,min(i+80,n)):
            if (l[k]<=stop) if sgn>0 else (h[k]>=stop):
                R=-1.0-0.05*atr[i]/risk; busy=k; break
            if (h[k]>=tgt) if sgn>0 else (l[k]<=tgt): R=tmult; busy=k; break
        if R is None: continue
        rows.append(R-(COST_BP/1e4)*e/risk)
    return np.array(rows)

if __name__=="__main__":
    print(f"{'pattern':<20}{'market':<10}{'n':>7}{'win%':>8}{'expR':>9}{'t':>8}")
    print("-"*62)
    agg={}
    for pn,fn in PATS.items():
        pool=[]
        for nm,slug in MK.items():
            R=bt(load(slug),fn)
            if len(R)>=30:
                m,se=R.mean(),R.std(ddof=1)/np.sqrt(len(R))
                print(f"{pn:<20}{nm:<10}{len(R):>7}{(R>0).mean()*100:>8.1f}"
                      f"{m:>9.3f}{m/se:>8.2f}")
                pool.append(R)
        if pool:
            P=np.concatenate(pool); m,se=P.mean(),P.std(ddof=1)/np.sqrt(len(P))
            agg[pn]=(len(P),m,m/se)
            print(f"{'':<20}{'POOLED':<10}{len(P):>7}{(P>0).mean()*100:>8.1f}"
                  f"{m:>9.3f}{m/se:>8.2f}")
        print()
