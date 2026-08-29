"""IS THE 30%/YR REAL, OR IS IT SURVIVORSHIP BIAS?

stocks.py found ~30%/yr with +12% excess in BOTH dev and holdout. Before that
counts, the obvious flaw has to be tested: the ticker list is TODAY'S large
caps. Every company that went bankrupt, got acquired at a loss, or fell out of
the index is missing. The tell is that equal-weight buy&hold of this universe
shows 19.24% CAGR over 2010-2026, when the actual market did roughly 14%. That
5-point gap is bias, sitting in plain sight.

The question is whether momentum's EXCESS survives it. Three tests:

  1. REAL BENCHMARK. Score against SPY, which has no survivorship bias,
     instead of against the biased universe.

  2. DROP THE WINNERS. Remove the names with the highest FULL-PERIOD return --
     exactly the ones survivorship bias smuggles in. If the edge is an artifact
     of holding eventual mega-winners, it collapses. If it persists on the
     remaining names, the signal is doing real work.

  3. SUB-PERIOD STABILITY. A bias-driven result concentrates in the years the
     survivors ran. A real effect shows up across most years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

import stocks as S


def spy_bench(index):
    d = yf.download("SPY", start="2009-01-01", progress=False,
                    auto_adjust=True)["Close"]
    if isinstance(d, pd.DataFrame):
        d = d.iloc[:, 0]
    return d.pct_change().reindex(index).fillna(0)


if __name__ == "__main__":
    px, vol = S.load(S.UNIV)
    print(f"universe: {px.shape[1]} stocks\n")
    bench = px.pct_change().mean(axis=1).dropna()
    r20 = S.run(px, top=20)
    spy = spy_bench(r20.index)

    print("="*80)
    print("1. AGAINST A BENCHMARK WITH NO SURVIVORSHIP BIAS")
    print("="*80)
    print(f"  {'series':<34}{'CAGR':>9}{'Sharpe':>9}{'maxDD':>8}")
    print("  "+"-"*60)
    for lab, s in (("SPY (unbiased)", spy),
                   ("equal-weight of MY universe", bench.reindex(r20.index).fillna(0)),
                   ("momentum top 20", r20)):
        st = S.stats(s)
        if st:
            print(f"  {lab:<34}{st['cagr']:>8.2f}%{st['sharpe']:>9.2f}{st['dd']:>7.0f}%")
    a, b = S.stats(r20), S.stats(spy)
    ub = S.stats(bench.reindex(r20.index).fillna(0))
    print(f"\n  universe bias  = {ub['cagr']-b['cagr']:+.2f}%/yr "
          f"(my universe vs SPY -- this is the contamination)")
    print(f"  momentum excess over SPY      = {a['cagr']-b['cagr']:+.2f}%/yr")
    print(f"  momentum excess over universe = {a['cagr']-ub['cagr']:+.2f}%/yr")
    print("  -> the second number is the honest one: it nets out the bias,")
    print("     because signal and benchmark share the same tainted universe.")

    print("\n"+"="*80)
    print("2. DROP THE BIGGEST FULL-PERIOD WINNERS")
    print("="*80)
    total = (px.iloc[-1]/px.iloc[0] - 1).sort_values(ascending=False)
    print(f"  {'universe':<34}{'CAGR':>9}{'B&H':>9}{'excess':>9}{'Sharpe':>8}")
    print("  "+"-"*70)
    for drop in (0, 10, 20, 40, 60):
        keep = list(total.index[drop:])
        sub = px[keep]
        rr = S.run(sub, top=min(20, max(5, len(keep)//10)))
        bb = sub.pct_change().mean(axis=1).reindex(rr.index).fillna(0)
        sa, sb = S.stats(rr), S.stats(bb)
        if not sa or not sb:
            continue
        lab = "full universe" if drop == 0 else f"drop top {drop} winners"
        print(f"  {lab:<34}{sa['cagr']:>8.2f}%{sb['cagr']:>8.2f}%"
              f"{sa['cagr']-sb['cagr']:>+8.2f}%{sa['sharpe']:>8.2f}")
    print("\n  a stable excess as winners are removed = the signal is real")

    print("\n"+"="*80)
    print("3. YEAR BY YEAR")
    print("="*80)
    yr = pd.DataFrame({"sys": r20, "bench": bench.reindex(r20.index).fillna(0),
                       "spy": spy})
    g = yr.groupby(yr.index.year).apply(lambda d: (1+d).prod()-1)*100
    print(f"  {'year':<8}{'system':>10}{'universe':>11}{'SPY':>9}{'vs univ':>10}")
    print("  "+"-"*48)
    wins = 0
    for y, row in g.iterrows():
        d = row["sys"]-row["bench"]
        wins += d > 0
        print(f"  {y:<8}{row['sys']:>9.1f}%{row['bench']:>10.1f}%"
              f"{row['spy']:>8.1f}%{d:>+10.1f}%")
    print(f"\n  beat the universe in {wins}/{len(g)} years")
