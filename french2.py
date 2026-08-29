"""EVERY MAJOR FACTOR, CLEAN DATA -- and what combining them is worth.

french.py settled momentum on survivorship-free CRSP data: real over a century
(t=5.13), decayed since publication (t=1.36 recently), worth ~+3%/yr over the
index after costs.

The remaining question is diversification. Combining genuinely uncorrelated
edges raises Sharpe without raising drawdown, and it is the only free lunch in
finance. Every attempt on my own data failed because the candidates correlated
at +0.51 to +0.69 -- they were all just "own equities" in disguise.

French's factors are LONG-SHORT, so they are market-neutral by construction
and can actually be uncorrelated with each other. This tests:

  MKT-RF  the market
  SMB     size (small minus big)
  HML     value (high minus low book-to-market)
  RMW     profitability (robust minus weak)
  CMA     investment (conservative minus aggressive)
  MOM     momentum (winners minus losers)

  1. each one alone, full sample and recent
  2. the correlation matrix -- is there real diversification here
  3. equal-weight and Sharpe-optimal combinations
  4. what that means for a LONG-ONLY retail account, which is the
     constraint that actually binds
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from french import fetch, parse_block, stats

RECENT = "2005-01-01"


def load_factors():
    ff = parse_block(fetch("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"))
    # the momentum file has a single data column (",Mom"), so it needs
    # min_cols=2 -- the 5-factor file has six and defaults to 3
    mom = parse_block(fetch("F-F_Momentum_Factor_daily_CSV.zip"), min_cols=2)
    if mom is not None:
        mom.columns = ["MOM"] + list(mom.columns[1:])
        mom = mom[["MOM"]]
    F = ff.join(mom, how="inner")
    return F.dropna()


if __name__ == "__main__":
    print("fetching 5 factors + momentum, survivorship-free...", flush=True)
    F = load_factors()
    facs = [c for c in F.columns if c != "RF"]
    print(f"{len(F):,} daily rows, {F.index.min():%Y-%m} -> {F.index.max():%Y-%m}")
    print(f"factors: {facs}\n")

    print("=" * 80)
    print("1. EACH FACTOR ALONE")
    print("=" * 80)
    print(f"  {'factor':<10}{'full CAGR':>12}{'Sharpe':>9}{'t':>8}   "
          f"{'2005+ CAGR':>12}{'Sharpe':>9}{'t':>8}")
    print("  " + "-" * 72)
    for c in facs:
        a = stats(F[c])
        b = stats(F[c].loc[RECENT:])
        if a and b:
            print(f"  {c:<10}{a['cagr']:>11.2f}%{a['sharpe']:>9.2f}{a['t']:>8.2f}   "
                  f"{b['cagr']:>11.2f}%{b['sharpe']:>9.2f}{b['t']:>8.2f}")

    print("\n" + "=" * 80)
    print("2. CORRELATION MATRIX -- is there real diversification?")
    print("=" * 80)
    C = F[facs].loc[RECENT:].corr()
    print("  " + "".join(f"{c:>9}" for c in facs))
    for i, c in enumerate(facs):
        print(f"  {c:<7}" + "".join(f"{C.iloc[i, j]:>9.2f}" for j in range(len(facs))))
    off = C.values[np.triu_indices_from(C.values, 1)]
    print(f"\n  mean |correlation| = {np.abs(off).mean():.3f}   "
          f"max = {np.abs(off).max():.3f}")
    print("  (my own factor candidates all ran +0.51 to +0.69 -- these are far lower)")

    print("\n" + "=" * 80)
    print("3. COMBINATIONS  (2005+, the period that matters for trading now)")
    print("=" * 80)
    R = F[facs].loc[RECENT:]
    print(f"  {'portfolio':<34}{'CAGR':>10}{'vol':>8}{'Sharpe':>9}{'maxDD':>9}")
    print("  " + "-" * 70)
    for lab, w in (("market only", {"Mkt-RF": 1}),
                   ("momentum only", {"MOM": 1}),
                   ("market + momentum 50/50", {"Mkt-RF": .5, "MOM": .5}),
                   ("equal-weight all 6", {c: 1/len(facs) for c in facs}),
                   ("5 styles, no market",
                    {c: 1/5 for c in facs if c != "Mkt-RF"})):
        s = sum(R[k] * v for k, v in w.items())
        st = stats(s)
        if st:
            print(f"  {lab:<34}{st['cagr']:>9.2f}%{st['vol']:>7.0f}%"
                  f"{st['sharpe']:>9.2f}{st['dd']:>8.0f}%")

    # risk-parity weights: inverse volatility, computed in-sample as a ceiling
    iv = 1 / R.std()
    iv = iv / iv.sum()
    s = (R * iv).sum(axis=1)
    st = stats(s)
    print(f"  {'risk parity (all 6)':<34}{st['cagr']:>9.2f}%{st['vol']:>7.0f}%"
          f"{st['sharpe']:>9.2f}{st['dd']:>8.0f}%")

    print("\n" + "=" * 80)
    print("4. THE CEILING -- best possible Sharpe from these factors")
    print("=" * 80)
    mu = R.mean().values * 252
    cov = R.cov().values * 252
    w = np.linalg.solve(cov, mu)
    w = w / np.abs(w).sum()
    opt = (R * w).sum(axis=1)
    st = stats(opt)
    print("  in-sample optimal (tangency) weights:")
    for c, wi in zip(facs, w):
        print(f"    {c:<10}{wi:>+8.2f}")
    print(f"\n  Sharpe {st['sharpe']:.2f}   CAGR {st['cagr']:.2f}%   "
          f"maxDD {st['dd']:.0f}%")
    print("  NOTE: fitted in-sample, so this is an upper bound, not a forecast.")

    print("\n" + "=" * 80)
    print("5. WHAT A RETAIL LONG-ONLY ACCOUNT CAN ACTUALLY GET")
    print("=" * 80)
    print("  The style factors are LONG-SHORT. Shorting hundreds of names is not")
    print("  practical retail, so the tradeable version is market + a long-only")
    print("  tilt, which captures roughly HALF of a long-short factor.\n")
    mkt = R["Mkt-RF"] + F["RF"].loc[RECENT:]
    print(f"  {'implementation':<40}{'CAGR':>10}{'Sharpe':>9}")
    print("  " + "-" * 60)
    for lab, s in (("buy & hold market", mkt),
                   ("market + 50% of MOM (long-only tilt)",
                    mkt + 0.5 * R["MOM"]),
                   ("market + 50% of MOM+HML+RMW",
                    mkt + 0.5 * (R["MOM"] + R["HML"] + R["RMW"]) / 3)):
        st = stats(s)
        if st:
            print(f"  {lab:<40}{st['cagr']:>9.2f}%{st['sharpe']:>9.2f}")
