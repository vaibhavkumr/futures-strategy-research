"""Test the video's ONE filter: sweep vs run, defined by the CLOSE.

His rule, precisely:
  SWEEP (trade it) -- price breaks the level but the candle CLOSES BACK
                      INSIDE, or leaves a large wick. Buyers/sellers stepped
                      in immediately.
  RUN   (avoid)    -- a big candle CLOSES THROUGH the level aggressively.
                      Price is gone; do not trade.

I previously tested sweep DEPTH (how far past the level). That is a different
thing. This tests the close-based rule as stated.
"""
import numpy as np, pandas as pd
from duka import load

SWING_LB, FVG_WINDOW = 2, 12
STOP_SLIP = 1.5
MK={"NASDAQ":"usatechidxusd","S&P 500":"usa500idxusd",
    "DOW":"usa30idxusd","DAX":"deuidxeur"}

def swings(df,lb):
    w=2*lb+1
    hi=(df["high"].rolling(w,center=True).max()==df["high"]).to_numpy()
    lo=(df["low"].rolling(w,center=True).min()==df["low"]).to_numpy()
    n=len(df); hi[:lb]=hi[n-lb:]=False; lo[:lb]=lo[n-lb:]=False
    return hi,lo

def run(df, tmult=1.0):
    o,h,l,c=(df[x].values for x in ("open","high","low","close"))
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift()).abs(),
                  (df["low"]-df["close"].shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean().bfill().values
    is_h,is_l=swings(df,SWING_LB)
    idx=df.index; n=len(df)
    lsh=lsl=np.nan; rows=[]; busy=-1
    for i in range(SWING_LB,n-1):
        conf=i-SWING_LB
        if is_h[conf]: lsh=h[conf]
        if is_l[conf]: lsl=l[conf]
        if i<busy: continue
        m=idx[i].hour*60+idx[i].minute
        if not (570<=m<960): continue
        sgn=lvl=None
        if not np.isnan(lsl) and l[i]<lsl: sgn,lvl=1,lsl
        elif not np.isnan(lsh) and h[i]>lsh: sgn,lvl=-1,lsh
        if sgn is None: continue
        # --- THE VIDEO'S CLASSIFICATION, on the sweeping candle ---
        closed_back_in = (c[i] > lvl) if sgn>0 else (c[i] < lvl)
        pierce = (lvl-l[i]) if sgn>0 else (h[i]-lvl)
        body   = abs(c[i]-o[i])
        rng    = max(h[i]-l[i],1e-9)
        wick_frac = pierce/rng                    # big wick = sweep
        close_beyond = abs(c[i]-lvl)/atr[i]       # how far it closed through
        kind = "SWEEP" if closed_back_in else ("RUN" if close_beyond>0.30 else "WEAK_RUN")
        # entry next bar at close, stop beyond the sweep extreme
        e=c[i+1] if i+1<n else c[i]
        stop=(min(l[i],l[i+1]) if sgn>0 else max(h[i],h[i+1]))
        risk=abs(e-stop)
        # BUG: STOP_SLIP is in POINTS. On a sweep-extreme stop the risk can be
        # 1-2 points, so fixed slippage produced -2R artifacts. Require a stop
        # of real size and scale slippage to ATR.
        if risk < 0.30*atr[i] or risk > 3*atr[i]: continue
        slip = 0.05*atr[i]
        tgt=e+sgn*tmult*risk
        R=None
        for k in range(i+2,min(i+2+80,n)):
            hs=(l[k]<=stop) if sgn>0 else (h[k]>=stop)
            ht=(h[k]>=tgt) if sgn>0 else (l[k]<=tgt)
            if hs: R=-1.0-slip/risk; busy=k; break
            if ht: R=tmult; busy=k; break
        if R is None: continue
        rows.append(dict(kind=kind,R=R,wick_frac=wick_frac,
                         body_atr=body/atr[i],side="long" if sgn>0 else "short"))
    return pd.DataFrame(rows)

if __name__=="__main__":
    allt=[]
    for nm,sl in MK.items():
        t=run(load(sl,"m5")); t["mkt"]=nm; allt.append(t)
        print(f"  {nm:<9} {len(t):>6} setups")
    T=pd.concat(allt,ignore_index=True); T.to_pickle("sweep_run.pkl")
    def st(x,lab):
        x=np.asarray(x,float)
        if len(x)<30: print(f"  {lab:<28} n={len(x)}"); return
        m,se=x.mean(),x.std(ddof=1)/np.sqrt(len(x))
        print(f"  {lab:<28} n={len(x):<6} win {(x>0).mean()*100:5.1f}%  "
              f"expR {m:+.3f}  t={m/se:+6.2f}")
    print(f"\n{'='*72}\nTHE VIDEO'S FILTER: does avoiding RUNS help?\n{'='*72}")
    st(T.R,"ALL setups (no filter)")
    for k,g in T.groupby("kind"): st(g.R,f"  {k}")
    print(f"\n{'='*72}\nHis 'big wick' refinement (sweeps only, by wick size)\n{'='*72}")
    s=T[T.kind=="SWEEP"].copy()
    s["wb"]=pd.cut(s.wick_frac,[0,.25,.5,.75,1.0])
    for b,g in s.groupby("wb",observed=True): st(g.R,f"  wick {b}")
