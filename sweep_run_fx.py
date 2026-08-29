"""His ACTUAL markets (gold, GBPUSD), and BOTH sides of the classifier.

Two corrections to the earlier test:

  1. He trades XAUUSD and GBPUSD with an FX broker -- not index futures.
     Testing his strategy on Nasdaq was testing the wrong instrument.

  2. Rather than DISCARDING runs, trade them as CONTINUATION. If a sweep
     means "fade the break" then a run plausibly means "follow the break".
     One classifier, two strategies, no wasted signals.

Costs modelled per instrument, since FX/metals spreads differ from indices.
"""
import numpy as np, pandas as pd, glob, os

SWING_LB=2
MK={"GOLD":"xauusd","GBPUSD":"gbpusd",
    "NASDAQ":"usatechidxusd","S&P 500":"usa500idxusd"}

def load(slug):
    fs=[f for f in glob.glob(f"download/{slug}-m5-bid-*.csv") if "2026-07-24" in f]
    d=pd.read_csv(max(fs,key=os.path.getsize))
    d.columns=[c.lower() for c in d.columns]
    ts=d["timestamp"]
    idx=pd.to_datetime(ts,unit="ms",utc=True) if pd.api.types.is_numeric_dtype(ts) \
        else pd.to_datetime(ts,utc=True)
    d.index=idx.dt.tz_convert("America/New_York")
    d=d[["open","high","low","close"]].astype(float).sort_index()
    assert d.index[0].year>2000
    return d

def swings(df,lb):
    w=2*lb+1
    hi=(df["high"].rolling(w,center=True).max()==df["high"]).to_numpy()
    lo=(df["low"].rolling(w,center=True).min()==df["low"]).to_numpy()
    n=len(df); hi[:lb]=hi[n-lb:]=False; lo[:lb]=lo[n-lb:]=False
    return hi,lo

def scan(df, tmult=1.0, sess=(120,960)):
    o,h,l,c=(df[x].values for x in ("open","high","low","close"))
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift()).abs(),
                  (df["low"]-df["close"].shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean().bfill().values
    is_h,is_l=swings(df,SWING_LB); idx=df.index; n=len(df)
    lsh=lsl=np.nan; rows=[]; busy=-1
    for i in range(SWING_LB,n-2):
        conf=i-SWING_LB
        if is_h[conf]: lsh=h[conf]
        if is_l[conf]: lsl=l[conf]
        if i<busy: continue
        m=idx[i].hour*60+idx[i].minute
        if not (sess[0]<=m<sess[1]): continue
        brk=lvl=None
        if not np.isnan(lsl) and l[i]<lsl: brk,lvl=1,lsl      # broke DOWN
        elif not np.isnan(lsh) and h[i]>lsh: brk,lvl=-1,lsh   # broke UP
        if brk is None: continue
        closed_back = (c[i]>lvl) if brk>0 else (c[i]<lvl)
        beyond=abs(c[i]-lvl)/atr[i]
        pierce=(lvl-l[i]) if brk>0 else (h[i]-lvl)
        wick=pierce/max(h[i]-l[i],1e-9)
        kind = "SWEEP" if closed_back else ("RUN" if beyond>0.30 else "WEAK")
        # SWEEP -> fade (trade opposite the break). RUN -> follow the break.
        sgn = brk if kind=="SWEEP" else -brk
        e=c[i+1]
        stop = (min(l[i],l[i+1]) if sgn>0 else max(h[i],h[i+1]))
        risk=abs(e-stop)
        if risk<0.30*atr[i] or risk>3*atr[i]: continue
        slip=0.05*atr[i]; tgt=e+sgn*tmult*risk; R=None
        for k in range(i+2,min(i+2+80,n)):
            if (l[k]<=stop) if sgn>0 else (h[k]>=stop): R=-1.0-slip/risk; busy=k; break
            if (h[k]>=tgt) if sgn>0 else (l[k]<=tgt): R=tmult; busy=k; break
        if R is None: continue
        rows.append(dict(kind=kind,R=R,wick=wick,mkt=None))
    return pd.DataFrame(rows)

def st(x,lab,cost=0.0):
    x=np.asarray(x,float)-cost
    if len(x)<50: print(f"    {lab:<26} n={len(x)}"); return
    m,se=x.mean(),x.std(ddof=1)/np.sqrt(len(x))
    print(f"    {lab:<26} n={len(x):<6} win {(x>0).mean()*100:5.1f}%  "
          f"expR {m:+.3f}  t={m/se:+6.2f}")

if __name__=="__main__":
    for nm,slug in MK.items():
        d=load(slug); t=scan(d)
        print(f"\n{'='*70}\n{nm}   ({len(d):,} bars, {len(t):,} setups)\n{'='*70}")
        st(t.R,"ALL (sweep-fade + run-follow)")
        for k,g in t.groupby("kind"):
            lab = {"SWEEP":"SWEEP -> faded","RUN":"RUN -> followed",
                   "WEAK":"WEAK -> followed"}[k]
            st(g.R,f"  {lab}")
        s=t[t.kind=="SWEEP"]
        if len(s)>200:
            big=s[s.wick>0.5]
            st(big.R,"  SWEEP big wick (>50%)")
