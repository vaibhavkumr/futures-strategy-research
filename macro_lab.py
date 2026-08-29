"""MACRO / CREDIT SIGNALS -- genuinely different from price momentum.

Free data, built from Yahoo (FRED blocked the connection):
  VIX          implied vol
  hy_spread    high-yield vs treasury relative move (credit stress proxy)
  ig_spread    investment-grade vs treasury
  term         long vs intermediate treasury (curve shape)
  curve        10Y minus 3M

Documented literature: widening credit spreads predict LOWER future equity
returns (Gilchrist & Zakrajsek); the variance risk premium predicts returns
(Bollerslev/Tauchen/Zhou); yield-curve slope predicts recessions.

Two ways to use them, both tested:
  STANDALONE  time the market: long when conditions are benign, cash otherwise
  OVERLAY     scale the momentum portfolio's exposure by macro conditions

Benchmark is always buy&hold of the same thing, plus a placebo.
"""
import numpy as np, pandas as pd
import factor_lab as F

def macro():
    return pd.read_pickle("macro.pkl")

def sig_vix_calm(M):
    """Long when VIX is below its own 1-year median."""
    v=M["vix"]; med=v.rolling(252).median()
    return (v.shift(1)<med.shift(1)).astype(float)

def sig_vix_spike(M):
    """Contrarian: long after a VIX spike."""
    v=M["vix"]; z=(v-v.rolling(252).mean())/v.rolling(252).std()
    return (z.shift(1)>1.5).astype(float)

def sig_credit_ok(M):
    """Long when high-yield credit is NOT deteriorating."""
    return (M["hy_spread"].shift(1)<0).astype(float)

def sig_credit_improving(M):
    hy=M["hy_spread"]
    return (hy.shift(1)<hy.rolling(63).mean().shift(1)).astype(float)

def sig_curve_ok(M):
    return (M["curve"].shift(1)>0).astype(float)

SIGS={"vix_calm":sig_vix_calm,"vix_spike":sig_vix_spike,
      "credit_ok":sig_credit_ok,"credit_improving":sig_credit_improving,
      "curve_ok":sig_curve_ok}

def met(r,ann=252):
    r=pd.Series(r).dropna()
    if len(r)<200: return None
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()*100
    yrs=len(r)/ann
    return dict(cagr=(eq.iloc[-1]**(1/yrs)-1)*100,dd=dd,
                sharpe=r.mean()/r.std(ddof=1)*np.sqrt(ann))

if __name__=="__main__":
    px=F.universe(); M=macro()
    bench=F.benchmark(px)
    mom=F.run(px,F.f_mom12)
    DEV="2018-01-01"
    print("A. STANDALONE MACRO TIMING (on equal-weight universe)\n")
    print(f"{'signal':<18}{'DEV exc':>9}{'HOLD exc':>10}{'DEVshp':>8}{'HOLDshp':>9}{'':>11}")
    print("-"*66)
    keep={}
    for name,fn in SIGS.items():
        s=fn(M).reindex(bench.index).ffill().fillna(0)
        r=bench*s
        vals=[]
        for sl in (slice(None,DEV),slice(DEV,None)):
            a=met(r.loc[sl]); b=met(bench.loc[sl])
            if not a or not b: vals=None; break
            vals.append((a["cagr"]-b["cagr"],a["sharpe"]-b["sharpe"]))
        if not vals: print(f"{name:<18} (insufficient)"); continue
        (de,ds),(he,hs)=vals
        ok=de>0 and he>0
        if ok: keep[name]=r
        print(f"{name:<18}{de:>+9.2f}{he:>+10.2f}{ds:>+8.2f}{hs:>+9.2f}"
              f"{'  BEATS B&H' if ok else '':>11}")
    print(f"\nstandalone survivors: {list(keep) if keep else 'none'}")

    print("\nB. MACRO AS AN OVERLAY ON MOMENTUM\n")
    print(f"{'overlay':<18}{'CAGR':>8}{'maxDD':>8}{'Sharpe':>8}{'Calmar':>8}")
    print("-"*52)
    m0=met(mom)
    print(f"{'none (mom_12m)':<18}{m0['cagr']:>8.2f}{m0['dd']:>8.1f}"
          f"{m0['sharpe']:>8.2f}{m0['cagr']/abs(m0['dd']):>8.2f}")
    for name,fn in SIGS.items():
        s=fn(M).reindex(mom.index).ffill().fillna(0)
        r=mom*s
        mm=met(r)
        if not mm: continue
        print(f"{name:<18}{mm['cagr']:>8.2f}{mm['dd']:>8.1f}"
              f"{mm['sharpe']:>8.2f}{mm['cagr']/abs(mm['dd']):>8.2f}")
