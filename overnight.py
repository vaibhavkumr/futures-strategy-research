"""THE OVERNIGHT PREMIUM -- can it actually be harvested?

Measured, 6/6 assets, t = 2.65 to 4.58:

    asset   overnight bp/day      intraday bp/day
    SPY          3.34  t=3.58        1.41  t=1.16
    QQQ          4.72  t=4.58        1.67  t=1.14
    IWM          4.94  t=4.42       -0.39  t=-0.23
    XLF          4.61  t=3.19       -0.57  t=-0.30

The equity risk premium accrues almost entirely between the close and the
next open. The session itself has no measurable drift. This is Lou, Polk &
Skouras and it replicates cleanly on my data.

It also explains ~46 dead candidates: intraday strategies compete for a
component whose expected value is zero, which is exactly the "fair odds" wall
I keep hitting. Day trading is not merely hard, it is fishing in the half of
the day where the drift is not.

THE OBSTACLE: capturing it means buy-at-close / sell-at-open EVERY day.
252 round trips at ~4bp = ~10%/yr of cost against ~8-12%/yr of gross premium.
Net is roughly zero -- already measured at -0.57%.

THE QUESTION HERE: does the premium CONCENTRATE? If most of it lands on an
identifiable minority of nights, holding only those nights cuts turnover
enough to clear the cost. Every filter below uses only information available
at the close, before the position is taken.
"""
import numpy as np, pandas as pd, yfinance as yf
COST=2.0    # bp per side

def load():
    t=["SPY","QQQ","IWM","XLK","XLF","EEM"]
    d=yf.download(t,start="2004-01-01",progress=False,auto_adjust=True)
    C,O=d["Close"].ffill(),d["Open"].ffill()
    V=yf.download("^VIX",start="2004-01-01",progress=False,
                  auto_adjust=True)["Close"].ffill()
    if isinstance(V,pd.DataFrame): V=V.iloc[:,0]
    return C,O,V

def build(C,O,V):
    on=(O/C.shift(1)-1)          # the tradeable overnight return
    it=(C/O-1)                   # today's session, known at the close
    F=pd.DataFrame(index=C.index)
    F["on"]=on.mean(axis=1)*1e4
    F["intra"]=it.mean(axis=1)*1e4
    F["vix"]=V.reindex(C.index).ffill()
    F["vixz"]=(F.vix-F.vix.rolling(252).mean())/F.vix.rolling(252).std()
    return F.dropna(subset=["on"])

def filters(F):
    ix=F.index
    ser=pd.Series(ix,index=ix); tom=np.zeros(len(ix),bool)
    for _,g in ser.groupby([ix.year,ix.month]):
        dd=pd.DatetimeIndex(g.values)
        tom|=ix.isin(dd[:3]); tom|=ix.isin(dd[-2:])
    # every condition uses ONLY data known at the close we buy on
    return {
      "all nights":            pd.Series(True,index=ix),
      "after DOWN session":    F.intra.shift(1)<0,
      "after UP session":      F.intra.shift(1)>0,
      "after big DOWN (<-50bp)":F.intra.shift(1)<-50,
      "VIX elevated (z>0.5)":  F.vixz.shift(1)>0.5,
      "VIX calm (z<0)":        F.vixz.shift(1)<0,
      "turn of month":         pd.Series(tom,index=ix),
      "Mon night":             pd.Series(ix.dayofweek==0,index=ix),
      "Fri night (weekend)":   pd.Series(ix.dayofweek==4,index=ix),
      "down + VIX elevated":   (F.intra.shift(1)<0)&(F.vixz.shift(1)>0.5),
    }

if __name__=="__main__":
    C,O,V=load(); F=build(C,O,V)
    print(f"{len(F):,} sessions  {F.index.min():%Y-%m} -> {F.index.max():%Y-%m}")
    print(f"cost assumption: {COST:.0f}bp/side = {2*COST:.0f}bp per night held\n")
    DEV="2018-01-01"
    print(f"{'filter':<26}{'nights':>7}{'%':>5}{'gross':>8}{'t':>7}"
          f"{'NET':>8}{'net/yr':>8}{'DEV':>7}{'HOLD':>7}")
    print("-"*84)
    rows=[]
    for name,m in filters(F).items():
        m=m.reindex(F.index).fillna(False)
        x=F.on[m]
        if len(x)<150: continue
        g=x.mean(); net=g-2*COST
        t=g/(x.std(ddof=1)/np.sqrt(len(x)))
        per_yr=net*len(x)/(len(F)/252)/100     # % per year
        dv=F.on[m&(F.index<DEV)].mean()-2*COST
        hd=F.on[m&(F.index>=DEV)].mean()-2*COST
        rows.append((name,net,dv,hd,per_yr))
        print(f"{name:<26}{len(x):>7}{len(x)/len(F)*100:>5.0f}{g:>8.2f}{t:>7.2f}"
              f"{net:>8.2f}{per_yr:>7.1f}%{dv:>7.2f}{hd:>7.2f}")
    good=[r for r in rows if r[1]>0 and r[2]>0 and r[3]>0]
    print(f"\nprofitable NET of cost in both DEV and HOLDOUT: "
          f"{[r[0] for r in good] if good else 'NONE'}")
    if good:
        for nm,net,dv,hd,py in good:
            print(f"   {nm:<26} net {net:+.2f}bp/night  ->  {py:+.1f}%/yr")
    print("\nSANITY: what does the cost assumption have to be to work?")
    allon=F.on.mean()
    print(f"  all-nights gross = {allon:.2f}bp; breakeven round-trip cost = {allon:.2f}bp")
    print(f"  = {allon/2:.2f}bp per side. SPY spread is ~1bp, so ~2bp round trip.")
    print(f"  margin = {allon-2*1.0:+.2f}bp/night at a 1bp/side fill.")
