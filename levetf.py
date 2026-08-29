"""SCALING DOWN: run the strategy on LEVERAGED ETFs.

The blocker is not account size, it is that 20x on an ETF book does not exist
for retail. But leverage can be bought INSIDE the instrument: 3x sector ETFs
are ordinary shares, purchasable in any account, at any size, fractionally,
with no PDT rule and no portfolio-margin minimum.

    SMH -> SOXL (3x semis)     QQQ -> TQQQ (3x nasdaq)
    XBI -> LABU (3x biotech)   SPY -> UPRO (3x s&p)
    IWM -> TNA  (3x smallcap)  XLF -> FAS  (3x financials)
    XLE -> ERX  (2x energy)    GLD -> UGL  (2x gold)
    SLV -> AGQ  (2x silver)    EEM -> EDC  (3x emerging)

Held in a 2x Reg T account that is 6x effective -- legally, at $200.

THE CATCH, and the reason this must be measured rather than assumed: these
reset leverage DAILY. That is precisely the volatility drag I had to fix in
moonshot.py, except here it is baked into the instrument and cannot be
engineered away. A 3x ETF does NOT return 3x its index over time; in choppy
markets it returns considerably less, sometimes negative while the index rises.

So this uses the leveraged ETFs' REAL price history, which already contains
every bit of that decay. No modelling, just what they actually did.
"""
import numpy as np, pandas as pd, yfinance as yf

LEV_UNIV = ["SOXL","TQQQ","LABU","UPRO","TNA","FAS","ERX","UGL","AGQ","EDC",
            "TECL","CURE","DRN","YINN","NUGT"]
BASE_UNIV = ["SMH","QQQ","XBI","SPY","IWM","XLF","XLE","GLD","SLV","EEM",
             "XLK","XLV","VNQ","EFA","GDX"]

def load(tk):
    d = yf.download(tk, start="2010-01-01", progress=False, auto_adjust=True)["Close"]
    return d.dropna(axis=1, how="all").ffill()

def run(px, top=6, k=0.5, cost_bp=10, lev=1.0):
    """Weekly conviction-tilted momentum, same engine as the live bot."""
    r = px.pct_change()
    S = px.pct_change(252).shift(1)
    weeks = px.resample("W-FRI").last().index
    out=[]; prev={}
    for i in range(1,len(weeks)):
        t0,t1 = weeks[i-1], weeks[i]
        s = S.reindex([t0],method="ffill").iloc[0].dropna()
        if len(s)<top: continue
        pick = s.nlargest(top)
        z = (pick-pick.mean())/pick.std(ddof=1) if pick.std(ddof=1)>0 else pick*0
        w = np.exp(k*z); w = w/w.sum()*lev
        seg = r.loc[(r.index>t0)&(r.index<=t1), list(pick.index)]
        if seg.empty: continue
        pr = (seg*w).sum(axis=1).copy()
        turn = sum(abs(w.get(x,0)-prev.get(x,0)) for x in set(w.index)|set(prev))
        pr.iloc[0] -= turn*cost_bp/1e4
        prev = w.to_dict(); out.append(pr)
    return pd.concat(out).sort_index() if out else pd.Series(dtype=float)

def stats(x,ann=252):
    x=x.dropna()
    if len(x)<200: return None
    x=np.clip(x,-0.99,None); eq=(1+x).cumprod()
    return dict(cagr=(eq.iloc[-1]**(ann/len(x))-1)*100,
                dd=(eq/eq.cummax()-1).min()*100,
                sharpe=x.mean()/x.std(ddof=1)*np.sqrt(ann),
                vol=x.std(ddof=1)*np.sqrt(ann)*100)

L = load(LEV_UNIV); B = load(BASE_UNIV)
common = L.index.intersection(B.index)
L, B = L.loc[common], B.loc[common]
print(f"leveraged universe: {L.shape[1]} ETFs, {L.index.min():%Y-%m} -> {L.index.max():%Y-%m}")
print(f"base universe     : {B.shape[1]} ETFs, same window\n")

print("="*78)
print("STRATEGY ON EACH UNIVERSE  (identical engine, weekly, k=0.5, top 6)")
print("="*78)
print(f"  {'universe':<30}{'CAGR':>9}{'vol':>8}{'maxDD':>9}{'Sharpe':>8}")
print("  "+"-"*64)
for lab,px,lv in (("base ETFs, 1x",B,1.0),
                  ("base ETFs, 2x margin",B,2.0),
                  ("3x LEVERAGED ETFs, 1x",L,1.0),
                  ("3x LEVERAGED ETFs, 2x margin",L,2.0)):
    s = stats(run(px,lev=lv))
    if s: print(f"  {lab:<30}{s['cagr']:>8.1f}%{s['vol']:>7.0f}%{s['dd']:>8.0f}%{s['sharpe']:>8.2f}")

print("\n"+"="*78)
print("IS 3x ETF ACTUALLY 3x?  -- decay check, buy & hold")
print("="*78)
print(f"  {'pair':<18}{'base CAGR':>11}{'3x CAGR':>10}{'3x/base':>9}{'implied':>9}")
print("  "+"-"*58)
for a,b in (("SOXL","SMH"),("TQQQ","QQQ"),("UPRO","SPY"),("LABU","XBI"),
            ("TNA","IWM"),("FAS","XLF")):
    if a not in L or b not in B: continue
    ra=L[a].pct_change().dropna(); rb=B[b].pct_change().dropna()
    j=ra.index.intersection(rb.index)
    ca=((1+ra[j]).prod()**(252/len(j))-1)*100
    cb=((1+rb[j]).prod()**(252/len(j))-1)*100
    print(f"  {a:<6} vs {b:<8}{cb:>10.1f}%{ca:>9.1f}%{ca/cb if cb else np.nan:>9.2f}x"
          f"{'3.00x':>9}")
print("\n  ratio below 3.00 = decay is eating the leverage you paid for.")

print("\n"+"="*78)
print("WHAT $200 WOULD DO")
print("="*78)
for lab,px,lv in (("base ETFs 2x",B,2.0),("3x lev ETFs, cash",L,1.0),
                  ("3x lev ETFs + 2x margin",L,2.0)):
    s=stats(run(px,lev=lv))
    if not s: continue
    fin = (lv-1)*0.055
    net = s['cagr']/100 - fin
    print(f"  {lab:<26} net {net*100:>6.1f}%/yr   $200 -> ${200*(1+net):>6,.0f}"
          f"   maxDD {s['dd']:>5.0f}%")
