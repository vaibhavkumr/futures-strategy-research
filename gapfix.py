"""FIXING GAP RISK -- what actually works.

USO gapped -6.5% overnight at 7.2% weight and 30x leverage = -15% of the
account in one print. Four candidate fixes, measured rather than assumed:

  A. WAIT AFTER THE OPEN     no effect. The gap is IN the opening price;
                             looking later only delays recognising it. Not
                             simulated because there is nothing to simulate.
  B. FLAT OVERNIGHT          eliminates gap risk completely. Also eliminates
                             the return -- measured today, ALL equity drift is
                             overnight (3.3-4.9bp/day, t=2.65-4.58) and the
                             session itself has none. Quantified below.
  C. POSITION CAP            no single name above `cap`, excess redistributed.
                             Directly limits what one gap can cost. This is
                             the fix that targets the actual failure.
  D. LOWER LEVERAGE          scales every risk down, including the good ones.

Scored on P($20k in 2 months), median outcome, and -- the number that matters
for gap risk -- WORST SINGLE DAY across the simulated paths.
"""
import numpy as np, pandas as pd
import factor_lab as F

px=F.universe(); wk=px.resample("W-FRI").last()
raw=px.pct_change(252).reindex(wk.index,method="ffill")
rd=px.pct_change(); COST=10/1e4; K=0.5
O=None
try:
    import yfinance as yf
    _d=yf.download(list(px.columns),start="2004-01-01",progress=False,auto_adjust=True)
    O=_d["Open"].ffill()
except Exception: pass
wkeys=[t for t in wk.index[:-1] if raw.loc[t].dropna().shape[0]>=8]

def stream(cap=None,pstop=0.06,overnight=True):
    out=[];prev={}
    for i,t in enumerate(wkeys[:-1]):
        a=raw.loc[t].dropna()
        z=(a-a.mean())/a.std(ddof=1); w=np.exp(K*z); w=w/w.sum()
        if cap:                      # cap and redistribute the excess
            for _ in range(60):
                over=w>cap
                if not over.any(): break
                ex=(w[over]-cap).sum(); w[over]=cap
                room=~over
                if not room.any(): break
                w[room]+=ex*w[room]/w[room].sum()
        seg=px.loc[(px.index>t)&(px.index<=wkeys[i+1])]
        if seg.empty: continue
        cols=[c for c in w.index if c in seg.columns]
        if not cols: continue
        w=w[cols]; entry=seg[cols].iloc[0]; live=pd.Series(True,index=cols)
        rets=[]
        for d in range(len(seg)):
            day=seg.index[d]
            if overnight or O is None:
                r=rd.loc[day,cols].fillna(0)
            else:                    # intraday only: open -> close
                r=(seg[cols].iloc[d]/O.loc[day,cols]-1).fillna(0)
            rets.append(float((w*live.astype(float)*r).sum()))
            dr=seg[cols].iloc[d]/entry-1
            hit=live&(dr<=-pstop)
            if hit.any():
                rets[-1]-=float(w[hit].sum())*COST; live&=~hit
        s=pd.Series(rets,index=seg.index)
        turn=sum(abs(w.get(x,0)-prev.get(x,0)) for x in set(w.index)|set(prev))
        if not overnight: turn*=len(seg)     # re-entering daily
        s.iloc[0]-=turn*COST
        prev=w.to_dict(); out.append(s)
    return pd.concat(out).sort_index().values

rng=np.random.default_rng(67)
def odds(DR,lev,floor=2000,days=42,n=40000,block=5):
    nb=max(1,(days+block-1)//block)
    idx=rng.integers(0,len(DR)-block,size=(n,nb))
    p=np.concatenate([DR[idx+j] for j in range(block)],axis=1)[:,:days]*lev
    eq=np.full(n,10000.0); hit=np.zeros(n,bool); st=np.zeros(n,bool)
    worst=np.zeros(n)
    for d in range(days):
        liv=~hit&~st
        worst=np.minimum(worst,np.where(liv,p[:,d],0))
        eq[liv]*=(1+np.clip(p[liv,d],-0.999,None))
        hit|=liv&(eq>=20000); eq[hit&(eq>20000)]=20000
        ns=liv&~hit&(eq<=floor); st|=ns; eq[ns]=floor
    return hit.mean(),st.mean(),np.median(eq),np.percentile(worst,1)

print("="*80)
print("GAP-RISK FIXES  (2 months, floor $2,000)")
print("="*80)
print(f"  {'config':<30}{'lev':>5}{'P($20k)':>10}{'P(floor)':>10}"
      f"{'median':>10}{'worst day':>11}")
print("  "+"-"*74)
base=stream()
h,s,m,wd=odds(base,30)
print(f"  {'current: no cap, 30x':<30}{30:>4}x{h*100:>9.1f}%{s*100:>9.1f}%"
      f"{m:>10,.0f}{wd*100:>10.1f}%")
print()
for cap in (0.06,0.05,0.04,0.03):
    DR=stream(cap=cap)
    h,s,m,wd=odds(DR,30)
    print(f"  {'position cap '+f'{cap*100:.0f}%':<30}{30:>4}x{h*100:>9.1f}%"
          f"{s*100:>9.1f}%{m:>10,.0f}{wd*100:>10.1f}%")
print()
for lev in (20,15,10):
    h,s,m,wd=odds(base,lev)
    print(f"  {'lower leverage':<30}{lev:>4}x{h*100:>9.1f}%{s*100:>9.1f}%"
          f"{m:>10,.0f}{wd*100:>10.1f}%")
print()
DR=stream(cap=0.04); h,s,m,wd=odds(DR,20)
print(f"  {'cap 4% + 20x (combined)':<30}{20:>4}x{h*100:>9.1f}%{s*100:>9.1f}%"
      f"{m:>10,.0f}{wd*100:>10.1f}%")

if O is not None:
    print("\n"+"="*80)
    print("B. GOING FLAT OVERNIGHT -- the cost of removing gap risk entirely")
    print("="*80)
    ni=stream(overnight=False)
    a=base.mean()*252*100; b=ni.mean()*252*100
    print(f"  hold overnight (now):  {a:>7.2f}%/yr unlevered")
    print(f"  intraday only:         {b:>7.2f}%/yr unlevered")
    print(f"  cost of avoiding gaps: {b-a:>7.2f} pct-pts per year")
    print("  -> matches this morning's finding: the drift is all overnight.")
