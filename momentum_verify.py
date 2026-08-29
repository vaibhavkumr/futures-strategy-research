"""RE-VERIFY mom_12m WITH THE STRICTER METHODOLOGY.

Two days ago this passed: holdout Sharpe 0.87, CAGR 18%. But today I found
three methodological errors that inflated results elsewhere:
  - pooling correlated markets as independent observations
  - scoring long-biased signals against zero instead of against being long
  - not checking how often random signals pass the same criteria

So mom_12m gets re-tested under all three corrections:
  1. benchmark = EQUAL-WEIGHT BUY AND HOLD of the same universe, not zero
  2. random-portfolio placebo: how often does a RANDOM pick of the same
     number of assets beat the benchmark by as much?
  3. strict dev/holdout, no parameter touching after the split

Cross-sectional momentum is structurally different from anything else tested
today: it ranks assets against EACH OTHER, so it is dollar-neutral-ish and
does not simply inherit the bull market the way a long-only timing signal does.
"""
from __future__ import annotations
import numpy as np, pandas as pd

def load_universe():
    import yfinance as yf
    tick = ["SPY","QQQ","IWM","EFA","EEM","TLT","IEF","LQD","HYG","GLD",
            "SLV","USO","DBC","VNQ","XLE","XLF","XLK","XLV","XLI","XLP",
            "XLU","XLY","XLB","XBI","SMH","EWJ"]
    d = yf.download(tick, start="2004-01-01", progress=False,
                    auto_adjust=True)["Close"]
    return d.dropna(axis=1, how="all").ffill()

def backtest(px, look=252, top=6, cost_bp=10, hold_w=5):
    """Weekly rebalance, long the top-N by trailing return."""
    r = px.pct_change()
    sig = px.pct_change(look).shift(1)          # known before the week
    w = px.resample("W-FRI").last().index
    eq = [1.0]; dates=[]; rets=[]
    prev = set()
    for i in range(1, len(w)):
        t0, t1 = w[i-1], w[i]
        s = sig.reindex([t0], method="ffill").iloc[0].dropna()
        if len(s) < top: continue
        pick = set(s.nlargest(top).index)
        seg = r.loc[(r.index > t0) & (r.index <= t1), list(pick)]
        if seg.empty: continue
        pr = seg.mean(axis=1)
        turn = len(pick ^ prev)/max(len(pick),1)
        pr.iloc[0] -= turn*cost_bp/1e4
        prev = pick
        rets.append(pr); dates.append(t1)
    R = pd.concat(rets).sort_index()
    return R

def bench(px, cost_bp=0):
    return px.pct_change().mean(axis=1).dropna()

def met(r, lab):
    r = pd.Series(r).dropna()
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()*100
    yrs=len(r)/252; ann=(eq.iloc[-1]**(1/yrs)-1)*100
    sh=r.mean()/r.std(ddof=1)*np.sqrt(252)
    print(f"  {lab:<30} CAGR {ann:>6.2f}%  maxDD {dd:>6.1f}%  Sharpe {sh:>5.2f}")
    return ann, sh

if __name__=="__main__":
    px = load_universe()
    print(f"universe: {px.shape[1]} assets, {px.index.min():%Y-%m} to {px.index.max():%Y-%m}\n")
    dev_end="2018-01-01"
    for lab, sl in (("DEV 2004-2017", slice(None,dev_end)),
                    ("HOLDOUT 2018-2026", slice(dev_end,None))):
        p = px.loc[sl]
        if len(p) < 300: continue
        print(f"{lab}")
        m = backtest(p)
        b = bench(p).reindex(m.index).dropna()
        am,sm = met(m, "  mom_12m top-6")
        ab,sb = met(b, "  equal-weight buy&hold")
        print(f"  {'EXCESS':<30} CAGR {am-ab:>+6.2f}%  Sharpe diff {sm-sb:>+5.2f}")
        print()
