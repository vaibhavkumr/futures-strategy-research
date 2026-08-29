"""THE 2-3%/DAY BAND -- take profit AND cut loss at a fixed daily level.

The earlier test was unfair to the idea: it capped gains only. The actual
proposal is symmetric -- exit at +2.5% OR -2.5% -- and that is a different
animal, because it truncates BOTH tails.

The argument, stated properly: with a symmetric band you win W% of days at
+b and lose (1-W)% at -b, so expectancy per day is b*(2W-1). Every bit of it
turns on W, the win rate. At b = 2.5%:

    W = 50%  ->  0.00%/day  ->  fair odds, zero
    W = 55%  ->  0.25%/day  ->    88%/yr
    W = 60%  ->  0.50%/day  ->   226%/yr
    W = 65%  ->  0.75%/day  ->   577%/yr

So the framework is arithmetically sound. It reduces entirely to whether the
win rate can be pushed above 50%, which is the same wall ~80 candidates hit.

What this file tests, on the tuned momentum system that DOES have edge:
  1. its actual daily win rate
  2. what a symmetric band does to it, at several widths
  3. the required win rate for each target, and how far the system is from it
  4. SLIPPAGE. The loss side cannot be executed at exactly -2.5% -- gaps and
     slippage mean real exits are worse. This is what killed the theoretical
     +119% result, so it is modelled explicitly rather than waved away.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import stocks as S
import stocks2 as S2
from conc2 import rebal_backtest
from conviction import growth

TUNED = dict(top=3, freq="2W-FRI", pstop=0.99, k=0.25)


def band(r, b, slip=0.0):
    """Symmetric daily band. `slip` is extra loss beyond the limit, modelling
    gaps and execution: you exit at -(b + slip), not at -b."""
    x = r.copy()
    x = x.clip(upper=b)
    x = x.where(x > -b, -(b + slip))
    return x


if __name__ == "__main__":
    px, _ = S.load(S.UNIV)
    total = (px.iloc[-1]/px.iloc[0]-1).sort_values(ascending=False)
    clean = px[list(total.index[30:])]
    r = rebal_backtest(clean, S2.sc_mom12(clean), **TUNED)
    g0 = growth(r)
    print(f"tuned system: {g0['growth']:.1f}%/yr, Sharpe {g0['sharpe']:.2f}, "
          f"{len(r):,} days\n")

    print("=" * 78)
    print("1. THE SYSTEM'S ACTUAL DAILY WIN RATE")
    print("=" * 78)
    W = (r > 0).mean()
    print(f"  up days   {W*100:.1f}%      down days {(1-W)*100:.1f}%")
    print(f"  avg up    {r[r>0].mean()*100:+.2f}%    avg down  {r[r<0].mean()*100:+.2f}%")
    print(f"  ratio     {abs(r[r>0].mean()/r[r<0].mean()):.2f} : 1")
    print(f"\n  edge comes from the SIZE of up days, not their frequency:")
    print(f"    {W*100:.0f}% x {r[r>0].mean()*100:.2f}% + {(1-W)*100:.0f}% x "
          f"{r[r<0].mean()*100:.2f}% = {r.mean()*100:+.3f}%/day")

    print("\n" + "=" * 78)
    print("2. REQUIRED WIN RATE FOR A SYMMETRIC BAND")
    print("=" * 78)
    print(f"  {'target/day':<14}{'need W for':>14}{'need W for':>14}{'need W for':>14}")
    print(f"  {'':<14}{'break-even':>14}{'+50%/yr':>14}{'+200%/yr':>14}")
    print("  " + "-" * 56)
    for b in (0.01, 0.02, 0.025, 0.03):
        need = {}
        for tgt_yr in (0.0, 0.50, 2.00):
            d = (1+tgt_yr)**(1/252) - 1
            need[tgt_yr] = (d/b + 1)/2
        print(f"  {b*100:>4.1f}%{'':<9}{need[0.0]*100:>13.1f}%"
              f"{need[0.50]*100:>13.1f}%{need[2.00]*100:>13.1f}%")
    print(f"\n  the tuned system's band win rate is computed below")

    print("\n" + "=" * 78)
    print("3. SYMMETRIC BAND ON THE TUNED SYSTEM  (perfect execution)")
    print("=" * 78)
    print(f"  {'band':<12}{'fires up':>11}{'fires dn':>11}{'band W':>9}"
          f"{'GROWTH':>10}{'Sharpe':>9}")
    print("  " + "-" * 62)
    for b in (0.01, 0.02, 0.025, 0.03, 0.05):
        x = band(r, b)
        w = (x > 0).mean()
        g = growth(x)
        if g:
            print(f"  {b*100:>4.1f}%{'':<7}{(r>b).mean()*100:>10.1f}%"
                  f"{(r<-b).mean()*100:>10.1f}%{w*100:>8.1f}%"
                  f"{g['growth']:>9.1f}%{g['sharpe']:>9.2f}")

    print("\n" + "=" * 78)
    print("4. WITH SLIPPAGE -- the loss side cannot execute at the limit")
    print("=" * 78)
    print("  USO gapped -6.5% overnight; no stop fires inside a gap.")
    print("  So the real exit is worse than the limit by `slip`.\n")
    print(f"  {'slippage':<14}{'band 2.5%':>13}{'band 3%':>12}{'Sharpe 2.5%':>14}")
    print("  " + "-" * 54)
    for slip in (0.000, 0.005, 0.010, 0.020, 0.030):
        a = growth(band(r, 0.025, slip))
        c = growth(band(r, 0.03, slip))
        if a and c:
            print(f"  {slip*100:>4.1f}%{'':<9}{a['growth']:>12.1f}%"
                  f"{c['growth']:>11.1f}%{a['sharpe']:>13.2f}")
    print(f"\n  uncapped, for reference: {g0['growth']:.1f}%/yr  "
          f"Sharpe {g0['sharpe']:.2f}")

    print("\n" + "=" * 78)
    print("5. HOW MUCH WOULD THE WIN RATE HAVE TO IMPROVE?")
    print("=" * 78)
    x = band(r, 0.025)
    w_now = (x > 0).mean()
    print(f"  current band win rate at 2.5%: {w_now*100:.1f}%")
    for tgt, lab in ((0.50, "+50%/yr"), (2.00, "+200%/yr"), (62.0, "$250/day")):
        d = (1+tgt)**(1/252) - 1
        need = (d/0.025 + 1)/2
        print(f"  to reach {lab:<10} need W = {need*100:>5.1f}%   "
              f"(gap: {(need-w_now)*100:+.1f} points)")
    print("\n  every candidate tested in this project landed at 48-52% on")
    print("  trade-level win rate -- that is the wall this framework runs into.")
