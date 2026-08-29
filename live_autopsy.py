"""Why does the LIVE bot take losing trades?

Replays live_paper.find_signal's exact logic over 4 markets x 4.5 years,
tags each trade with features known AT ENTRY, then asks which features
separate winners from losers -- chosen on DEV, confirmed on HOLDOUT.

11 live trades cannot answer this. ~10k backtest trades can.
"""
import numpy as np, pandas as pd
from duka import load

SWING_LB, FVG_WINDOW = 2, 12
ATR_STOP_MULT, STOP_SLIP = 1.0, 1.5
MK = {"NASDAQ":"usatechidxusd","S&P 500":"usa500idxusd",
      "DOW":"usa30idxusd","DAX":"deuidxeur"}

def swings(df, lb):
    w = 2*lb+1
    hi = (df["high"].rolling(w,center=True).max()==df["high"]).to_numpy()
    lo = (df["low"].rolling(w,center=True).min()==df["low"]).to_numpy()
    n=len(df); hi[:lb]=hi[n-lb:]=False; lo[:lb]=lo[n-lb:]=False
    return hi,lo

def session(ts):
    m = ts.hour*60+ts.minute
    if 120 <= m < 300:  return "LONDON"
    if 510 <= m < 660:  return "NY_AM"
    if 810 <= m < 960:  return "NY_PM"
    return None

def backtest(df, tmult=1.0):
    h,l,c,o = (df[x].values for x in ("high","low","close","open"))
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-df["close"].shift()).abs(),
                    (df["low"]-df["close"].shift()).abs()],axis=1).max(axis=1)
    atr = tr.rolling(14).mean().bfill().values
    ema50 = df["close"].ewm(span=50).mean().values
    ema200 = df["close"].ewm(span=200).mean().values
    is_h,is_l = swings(df,SWING_LB)
    idx = df.index; n=len(df)
    lsh=lsl=np.nan; armed=0; sweep=np.nan; expiry=-1; sweep_i=-1
    rows=[]; busy=-1
    for i in range(SWING_LB,n-1):
        conf=i-SWING_LB
        if is_h[conf]: lsh=h[conf]
        if is_l[conf]: lsl=l[conf]
        if not np.isnan(lsl) and l[i]<lsl:
            armed,sweep,expiry,sweep_i,swept=1,l[i],i+FVG_WINDOW,i,lsl
        elif not np.isnan(lsh) and h[i]>lsh:
            armed,sweep,expiry,sweep_i,swept=-1,h[i],i+FVG_WINDOW,i,lsh
        if armed!=0 and i>expiry: armed=0
        if armed==0 or i<busy: continue
        s = session(idx[i])
        if s is None: continue
        sig=None
        if armed==1 and h[i-2]<l[i]:
            entry=l[i]; stop=entry-ATR_STOP_MULT*atr[i]; sig=(1,entry,stop)
        elif armed==-1 and l[i-2]>h[i]:
            entry=h[i]; stop=entry+ATR_STOP_MULT*atr[i]; sig=(-1,entry,stop)
        if sig is None: continue
        sgn,entry,stop = sig
        risk=abs(entry-stop)
        if risk <= max(1e-6, 0.0005*entry): continue
        armed=0
        tgt = entry + sgn*tmult*risk
        # BUG (same class as #2): entry is the FVG EDGE -- a resting limit.
        # Assuming it fills for free at the signal bar's extreme hands the
        # backtest the best price in the gap and produced 74.6% wins at 1:1.
        # The order must actually be TOUCHED by a later bar, or it expires.
        kf=None
        for k in range(i+1, min(i+1+FVG_WINDOW, n)):
            if (l[k] <= entry) if sgn>0 else (h[k] >= entry):
                kf=k; break
        if kf is None: continue          # never filled -- not a trade
        R=None
        for k in range(kf+1, min(kf+80,n)):
            hit_stop = (l[k]<=stop) if sgn>0 else (h[k]>=stop)
            hit_tgt  = (h[k]>=tgt) if sgn>0 else (l[k]<=tgt)
            if hit_stop and hit_tgt: R=-1.0-STOP_SLIP/risk; busy=k; break
            if hit_stop: R=-1.0-STOP_SLIP/risk; busy=k; break
            if hit_tgt:  R=tmult; busy=k; break
        if R is None: continue
        rows.append(dict(ts=idx[kf], side="long" if sgn>0 else "short", R=R,
            sess=s, risk_atr=risk/atr[i],
            sweep_depth=abs(sweep-swept)/atr[i],
            bars_since_sweep=i-sweep_i, fill_delay=kf-i,
            atr_pct=atr[i]/c[i]*1e4,
            trend_align=int(np.sign(c[i]-ema200[i])==sgn),
            ema_slope=(ema50[i]-ema50[i-12])/atr[i],
            hour=idx[i].hour))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    allt=[]
    for name,slug in MK.items():
        d=load(slug,"m5")
        t=backtest(d); t["mkt"]=name; allt.append(t)
        print(f"{name:<9} {len(t):>6} trades  expR {t.R.mean():+.3f}")
    T=pd.concat(allt,ignore_index=True)
    T.to_pickle("live_autopsy.pkl")
    m,se=T.R.mean(),T.R.std(ddof=1)/np.sqrt(len(T))
    print(f"\nPOOLED {len(T):,} trades  win {(T.R>0).mean()*100:.1f}%  "
          f"expR {m:+.3f}  t={m/se:+.2f}")
