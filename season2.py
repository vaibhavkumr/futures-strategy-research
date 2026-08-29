"""SEASONALS DONE PROPERLY -- switch, don't sit in cash.

season_lab.py scored these against buy&hold while sitting in CASH half the
year. In a 10.76% CAGR bull market that is unwinnable by construction, and it
hid the actual effect: halloween earned 6.70% while invested only ~50% of the
time, which is 13.4% per unit of time exposed vs the benchmark's 10.76%.

The published implementation of Bouman & Jacobsen is "sell in May and buy
BONDS", not "sell in May and hold cash". So: rotate equities <-> TLT.

Also nails down the overnight/intraday split per-asset with t-stats, since
that was the single largest effect in the previous run.
"""
import numpy as np, pandas as pd, yfinance as yf
COST=2.0

def data():
    t=["SPY","QQQ","IWM","EFA","EEM","XLK","XLF","XLE","XLV","XLY","TLT"]
    d=yf.download(t,start="2004-01-01",progress=False,auto_adjust=True)
    return d["Close"].ffill(),d["Open"].ffill()

def perf(r,ann=252):
    r=pd.Series(r).dropna()
    if len(r)<200: return None
    eq=(1+r).cumprod(); yrs=len(r)/ann
    return dict(cagr=(eq.iloc[-1]**(1/yrs)-1)*100,dd=(eq/eq.cummax()-1).min()*100,
                sharpe=r.mean()/r.std(ddof=1)*np.sqrt(ann))

if __name__=="__main__":
    C,O=data()
    eq_cols=[c for c in C.columns if c!="TLT"]
    eq=C[eq_cols].pct_change().mean(axis=1).dropna()
    bond=C["TLT"].pct_change().reindex(eq.index).fillna(0)
    ix=eq.index; DEV="2018-01-01"

    print("A. IS THE EFFECT REAL? per-day return, in-season vs out\n")
    print(f"{'effect':<16}{'IN bp/d':>9}{'OUT bp/d':>10}{'diff':>8}{'t':>7}{'':>8}")
    print("-"*58)
    defs={"halloween":(ix.month>=11)|(ix.month<=4),
          "january":ix.month==1,
          "q4":ix.month.isin([10,11,12])}
    ser=pd.Series(ix,index=ix); tom=np.zeros(len(ix),bool)
    for _,g in ser.groupby([ix.year,ix.month]):
        dd=pd.DatetimeIndex(g.values)
        tom|=ix.isin(dd[:3]); tom|=ix.isin(dd[-2:])
    defs["turn_of_month"]=tom
    for name,mask in defs.items():
        a,b=eq[mask].values*1e4, eq[~mask].values*1e4
        d=a.mean()-b.mean()
        se=np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
        print(f"{name:<16}{a.mean():>9.2f}{b.mean():>10.2f}{d:>8.2f}{d/se:>7.2f}"
              f"{'  REAL' if abs(d/se)>2 else '':>8}")

    print("\nB. ROTATE EQUITIES <-> TLT instead of sitting in cash\n")
    print(f"{'strategy':<20}{'DEV exc':>9}{'HOLD exc':>10}{'CAGR':>8}"
          f"{'maxDD':>8}{'Shrp':>7}{'Calmar':>8}{'':>10}")
    print("-"*80)
    bh=perf(eq)
    print(f"{'buy & hold':<20}{'':>9}{'':>10}{bh['cagr']:>8.2f}{bh['dd']:>8.1f}"
          f"{bh['sharpe']:>7.2f}{bh['cagr']/abs(bh['dd']):>8.2f}")
    keep={}
    for name,mask in defs.items():
        pos=pd.Series(mask.astype(float),index=ix)
        r=eq*pos + bond*(1-pos)
        r=r - pos.diff().abs().fillna(0)*COST/1e4
        exc=[]
        for sl in (slice(None,DEV),slice(DEV,None)):
            x,y=perf(r.loc[sl]),perf(eq.loc[sl])
            exc.append(x["cagr"]-y["cagr"] if x and y else np.nan)
        f=perf(r); ok=exc[0]>0 and exc[1]>0
        if ok: keep[name]=r
        print(f"{name+' + bonds':<20}{exc[0]:>+9.2f}{exc[1]:>+10.2f}{f['cagr']:>8.2f}"
              f"{f['dd']:>8.1f}{f['sharpe']:>7.2f}{f['cagr']/abs(f['dd']):>8.2f}"
              f"{'  BEATS B&H' if ok else '':>10}")
    print(f"\nSURVIVORS: {list(keep) if keep else 'none'}")

    print("\nC. OVERNIGHT vs INTRADAY, per asset (gross, no cost)\n")
    print(f"{'asset':<8}{'ON bp/d':>9}{'t':>7}{'INTRA bp/d':>12}{'t':>7}"
          f"{'ON share':>10}")
    print("-"*54)
    for c in ["SPY","QQQ","IWM","XLK","XLF","EEM"]:
        if c not in C: continue
        on=(O[c]/C[c].shift(1)-1).dropna()*1e4
        it=(C[c]/O[c]-1).dropna()*1e4
        j=on.index.intersection(it.index); on,it=on[j],it[j]
        ton=on.mean()/(on.std(ddof=1)/np.sqrt(len(on)))
        tit=it.mean()/(it.std(ddof=1)/np.sqrt(len(it)))
        tot=on.mean()+it.mean()
        print(f"{c:<8}{on.mean():>9.2f}{ton:>7.2f}{it.mean():>12.2f}{tit:>7.2f}"
              f"{on.mean()/tot*100 if tot else np.nan:>9.0f}%")
