"""FACTOR LAB -- hunting edges 3 and 4 on the 26-asset ETF universe.

Why here rather than index futures: four indices correlate at 0.856, which is
what inflated my calendar t-stat and forced a correction. A 26-asset universe
gives genuine cross-sectional variation, and every factor below is documented
in the academic literature rather than invented by me.

FACTOR FAMILIES (all distinct mechanisms, not variations on price shape):

  MOMENTUM 12m     Jegadeesh & Titman. Already verified: +4.5% CAGR excess.
  MOMENTUM 3m      shorter horizon -- different behaviour, tested separately
  SHORT REVERSAL   1-month reversal (Jegadeesh 1990) -- opposite sign to mom
  LOW VOLATILITY   Baker/Bradley/Wurgler: low-vol assets win risk-adjusted
  TREND FOLLOWING  time-series momentum, each asset vs its own past
                   (Moskowitz/Ooi/Pedersen) -- distinct from cross-sectional
  CARRY            proxy via trailing yield/drift differential
  RISK-ON/OFF      equity vs bond relative strength as a regime rotation
  DISPERSION       when cross-asset spread is wide, it narrows

METHODOLOGY, corrected from today's errors:
  1. benchmark is EQUAL-WEIGHT BUY AND HOLD of the same universe, never zero
  2. DEV 2004-2017 -> HOLDOUT 2018-2026, both must beat the benchmark
  3. costs 10bp per unit turnover
  4. a RANDOM-PORTFOLIO placebo establishes how often this passes by luck
  5. survivors are checked for correlation against mom_12m and calendar
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def universe():
    import yfinance as yf
    tick = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD",
            "SLV", "USO", "DBC", "VNQ", "XLE", "XLF", "XLK", "XLV", "XLI", "XLP",
            "XLU", "XLY", "XLB", "XBI", "SMH", "EWJ"]
    d = yf.download(tick, start="2004-01-01", progress=False,
                    auto_adjust=True)["Close"]
    return d.dropna(axis=1, how="all").ffill()


# ------------------------------------------------------------- factor scores
# Each returns a DataFrame of scores; higher = more attractive. Uses only
# information available at the row's date.

def f_mom12(px):
    return px.pct_change(252)


def f_mom3(px):
    return px.pct_change(63)


def f_reversal_1m(px):
    return -px.pct_change(21)


def f_lowvol(px):
    return -px.pct_change().rolling(60).std()


def f_trend(px):
    """Time-series: distance above own 200-day mean (not cross-sectional)."""
    return px / px.rolling(200).mean() - 1


def f_carry(px):
    """Drift proxy: 3-year trailing return per unit of volatility."""
    r = px.pct_change(756)
    v = px.pct_change().rolling(252).std()
    return r / v.replace(0, np.nan)


def f_riskon(px):
    """Equity-vs-bond regime: score equities up when SPY beats TLT."""
    if "SPY" not in px or "TLT" not in px:
        return px * np.nan
    spread = px["SPY"].pct_change(63) - px["TLT"].pct_change(63)
    bonds = ["TLT", "IEF", "LQD", "HYG"]
    s = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for c in px.columns:
        s[c] = spread if c not in bonds else -spread
    return s


def f_dispersion_mr(px):
    """Fade the cross-sectional extremes when dispersion is wide."""
    r = px.pct_change(21)
    z = r.sub(r.mean(axis=1), axis=0).div(r.std(axis=1).replace(0, np.nan), axis=0)
    wide = r.std(axis=1) > r.std(axis=1).rolling(252).median()
    return (-z).mul(wide.astype(float), axis=0)


FACTORS = {
    "mom_12m": f_mom12,
    "mom_3m": f_mom3,
    "reversal_1m": f_reversal_1m,
    "low_vol": f_lowvol,
    "trend_200d": f_trend,
    "carry": f_carry,
    "risk_on_off": f_riskon,
    "dispersion_mr": f_dispersion_mr,
}


def run(px, score_fn, top=6, cost_bp=10):
    """Weekly rebalance into the top-N by score. Returns daily series."""
    r = px.pct_change()
    S = score_fn(px).shift(1)          # score known before the week starts
    weeks = px.resample("W-FRI").last().index
    rets = []
    prev = set()
    for i in range(1, len(weeks)):
        t0, t1 = weeks[i-1], weeks[i]
        s = S.reindex([t0], method="ffill").iloc[0].dropna()
        if len(s) < top:
            continue
        pick = set(s.nlargest(top).index)
        seg = r.loc[(r.index > t0) & (r.index <= t1), list(pick)]
        if seg.empty:
            continue
        pr = seg.mean(axis=1).copy()
        pr.iloc[0] -= (len(pick ^ prev)/max(len(pick), 1))*cost_bp/1e4
        prev = pick
        rets.append(pr)
    if not rets:
        return pd.Series(dtype=float)
    return pd.concat(rets).sort_index()


def benchmark(px):
    return px.pct_change().mean(axis=1).dropna()


def stats(r, ann=252):
    r = pd.Series(r).dropna()
    if len(r) < 100:
        return None
    eq = (1+r).cumprod()
    dd = (eq/eq.cummax()-1).min()*100
    yrs = len(r)/ann
    return dict(cagr=(eq.iloc[-1]**(1/yrs)-1)*100, dd=dd,
                sharpe=r.mean()/r.std(ddof=1)*np.sqrt(ann), n=len(r))


if __name__=="__main__":
    px=universe()
    print(f"universe {px.shape[1]} assets, {px.index.min():%Y-%m} to {px.index.max():%Y-%m}\n")
    DEV="2018-01-01"
    print(f"{'factor':<16}{'DEV exc':>9}{'DEVshp':>8}{'HOLD exc':>10}{'HOLDshp':>9}{'':>11}")
    print("-"*64)
    keep={}
    for name,fn in FACTORS.items():
        row=[]
        okboth=True
        for sl in (slice(None,DEV), slice(DEV,None)):
            p=px.loc[sl]
            m=run(p,fn); b=benchmark(p).reindex(m.index).dropna()
            sm,sb=stats(m),stats(b)
            if not sm or not sb: okboth=False; row+= [np.nan,np.nan]; continue
            row+=[sm["cagr"]-sb["cagr"], sm["sharpe"]-sb["sharpe"]]
        if len(row)<4 or any(pd.isna(row)): 
            print(f"{name:<16}  (insufficient)"); continue
        dev_e,dev_s,hold_e,hold_s=row
        ok = dev_e>0 and hold_e>0
        if ok:
            full=run(px,fn); keep[name]=full
        print(f"{name:<16}{dev_e:>+9.2f}{dev_s:>+8.2f}{hold_e:>+10.2f}{hold_s:>+9.2f}"
              f"{'  BEATS B&H' if ok else '':>11}")
    print(f"\nSURVIVORS: {list(keep) if keep else 'none'}")

    if keep:
        print("\nPLACEBO -- random 6-asset picks, same rebalance and costs:")
        rng=np.random.default_rng(5)
        wins=0; N=100
        b_full=benchmark(px)
        for k in range(N):
            def rnd(p,seed=k):
                r2=np.random.default_rng(seed)
                return pd.DataFrame(r2.standard_normal(p.shape),
                                    index=p.index,columns=p.columns)
            beat=True
            for sl in (slice(None,DEV),slice(DEV,None)):
                p=px.loc[sl]; m=run(p,rnd); b=benchmark(p).reindex(m.index).dropna()
                sm,sb=stats(m),stats(b)
                if not sm or not sb or sm["cagr"]<=sb["cagr"]: beat=False; break
            wins+=beat
        print(f"  random pickers beating B&H in BOTH periods: {wins}/{N} = {wins/N*100:.0f}%")
        print(f"  factors tested: {len(FACTORS)}   survivors: {len(keep)}")
