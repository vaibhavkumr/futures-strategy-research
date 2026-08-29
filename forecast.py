"""FULL STATISTICS ON THE TUNED SYSTEM.

Config, every setting measured rather than assumed:
    top 5 by 12-month momentum, biweekly rebalance, k=0.25 conviction tilt,
    +/-2.5% daily band, survivorship-controlled universe, 20bp costs,
    slippage set to the MEASURED overnight-gap value (0.013%)

Reported as distributions, because a single "expected return" hides the two
things that decide whether a system is livable: how often it loses, and how
deep the drawdowns get. Monthly figures are bootstrapped in 5-day blocks so
the clustering of bad stretches survives the resampling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import stocks as S
import stocks2 as S2
from conc2 import rebal_backtest
from conviction import growth

SLIP = 0.00013
BAND = 0.025


def build():
    px, _ = S.load(S.UNIV)
    total = (px.iloc[-1]/px.iloc[0]-1).sort_values(ascending=False)
    clean = px[list(total.index[30:])]
    r = rebal_backtest(clean, S2.sc_mom12(clean), top=5, freq="2W-FRI",
                       pstop=0.99, k=0.25)
    x = r.clip(upper=BAND)
    return x.where(x > -BAND, -(BAND + SLIP)), clean


if __name__ == "__main__":
    r, clean = build()
    g = growth(r)
    ann = (1 + r.mean())**252 - 1
    print("=" * 78)
    print("TUNED SYSTEM -- HEADLINE STATISTICS")
    print("=" * 78)
    print(f"  sample                {len(r):,} days  "
          f"({r.index.min():%Y-%m} to {r.index.max():%Y-%m})")
    print(f"  growth rate           {g['growth']:.1f}%/yr   (what compounds)")
    print(f"  volatility            {g['vol']:.1f}%/yr")
    print(f"  Sharpe                {g['sharpe']:.2f}")
    print(f"  max drawdown          {g['dd']:.1f}%")
    print(f"  daily win rate        {(r>0).mean()*100:.1f}%")
    print(f"  avg up day            {r[r>0].mean()*100:+.2f}%")
    print(f"  avg down day          {r[r<0].mean()*100:+.2f}%")
    print(f"  best / worst day      {r.max()*100:+.2f}% / {r.min()*100:+.2f}%")

    print("\n" + "=" * 78)
    print("MONTHLY DISTRIBUTION  (bootstrap, 5-day blocks, 60,000 paths)")
    print("=" * 78)
    rng = np.random.default_rng(11)
    v = r.values

    def boot(days, n=60000, block=5):
        nb = max(1, (days+block-1)//block)
        idx = rng.integers(0, len(v)-block, size=(n, nb))
        p = np.concatenate([v[idx+j] for j in range(block)], axis=1)[:, :days]
        return np.prod(1+np.clip(p, -0.99, None), axis=1) - 1

    m = boot(21)
    q = np.percentile(m, [5, 10, 25, 50, 75, 90, 95])
    print(f"  {'':<12}{'p5':>9}{'p10':>9}{'p25':>9}{'MEDIAN':>10}"
          f"{'p75':>9}{'p90':>9}{'p95':>9}")
    print("  " + "-" * 68)
    print(f"  {'return':<12}" + "".join(f"{x*100:>8.1f}%" for x in q[:3]) +
          f"{q[3]*100:>9.1f}%" + "".join(f"{x*100:>8.1f}%" for x in q[4:]))
    print(f"  {'on $10k':<12}" + "".join(f"{10000*x:>9,.0f}" for x in q))
    print(f"\n  P(losing month)  {(m<0).mean()*100:.1f}%")
    print(f"  P(month > +10%)  {(m>0.10).mean()*100:.1f}%")
    print(f"  P(month < -10%)  {(m<-0.10).mean()*100:.1f}%")
    print(f"  mean {m.mean()*100:+.2f}%   median {np.median(m)*100:+.2f}%")

    print("\n" + "=" * 78)
    print("DOLLARS PER MONTH AT EACH ACCOUNT SIZE  (median month)")
    print("=" * 78)
    med, p25, p75 = np.median(m), np.percentile(m, 25), np.percentile(m, 75)
    print(f"  {'capital':>12}{'p25':>12}{'MEDIAN':>12}{'p75':>12}{'per yr':>13}")
    print("  " + "-" * 62)
    for cap in (10000, 25000, 50000, 100000, 200000):
        print(f"  ${cap:>10,}{cap*p25:>12,.0f}{cap*med:>12,.0f}"
              f"{cap*p75:>12,.0f}{cap*(g['growth']/100):>13,.0f}")

    print("\n" + "=" * 78)
    print("YEAR AHEAD FROM $10,000")
    print("=" * 78)
    y = boot(252)
    q = np.percentile(y, [10, 25, 50, 75, 90])
    print(f"  {'p10':>12}{'p25':>12}{'MEDIAN':>12}{'p75':>12}{'p90':>12}")
    print("  " + "-" * 60)
    print("  " + "".join(f"{10000*(1+x):>12,.0f}" for x in q))
    print(f"\n  P(losing year) {(y<0).mean()*100:.1f}%   "
          f"P(doubling) {(y>1.0).mean()*100:.1f}%")

    print("\n" + "=" * 78)
    print("PATH TO $3,000/MONTH")
    print("=" * 78)
    gr = g['growth']/100
    need = 36000/gr
    print(f"  at {gr*100:.1f}%/yr, $3,000/mo needs ${need:,.0f}\n")
    print(f"  {'monthly addition':<22}{'years to target':>18}")
    print("  " + "-" * 42)
    for add in (0, 250, 500, 1000):
        bal, yrs = 10000.0, 0
        while bal < need and yrs < 50:
            bal = bal*(1+gr) + add*12
            yrs += 1
        print(f"  ${add:>6,}/month{'':<7}{yrs:>15} yrs")
    print(f"\n  growth of $10k with no additions:")
    for yy in (1, 3, 5, 7, 10):
        print(f"    year {yy:<3} ${10000*(1+gr)**yy:>12,.0f}")
