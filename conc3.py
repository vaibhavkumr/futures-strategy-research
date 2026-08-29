"""TOP-3 CONCENTRATION -- tuning first, then the placebo.

Ordered so the informative diagnostics land before the slow control, since a
200-draw placebo on 3-name portfolios is ~200 full backtests and the earlier
run died before printing anything useful.

Claim under test: momentum top-3 gives 32.7%/yr growth at Sharpe 0.94,
survivorship-controlled, stable across dev and holdout.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import stocks as S
import stocks2 as S2
from conc2 import rebal_backtest
from conviction import growth


def main():
    px, vol = S.load(S.UNIV)
    total = (px.iloc[-1] / px.iloc[0] - 1).sort_values(ascending=False)
    clean = px[list(total.index[30:])]
    print(f"clean universe: {clean.shape[1]} stocks\n", flush=True)

    real = rebal_backtest(clean, S2.sc_mom12(clean), top=3)
    g_real = growth(real)
    print(f"BASELINE  momentum top 3: growth {g_real['growth']:.1f}%/yr  "
          f"Sharpe {g_real['sharpe']:.2f}  maxDD {g_real['dd']:.0f}%\n", flush=True)

    print("=" * 78)
    print("1. ROBUSTNESS TO THE SURVIVORSHIP CONTROL")
    print("=" * 78)
    print(f"  {'drop top N':<20}{'GROWTH':>10}{'Sharpe':>9}{'maxDD':>9}")
    print("  " + "-" * 48, flush=True)
    for drop in (0, 15, 30, 50, 75):
        sub = px[list(total.index[drop:])]
        g = growth(rebal_backtest(sub, S2.sc_mom12(sub), top=3))
        if g:
            print(f"  drop {drop:<15}{g['growth']:>9.1f}%{g['sharpe']:>9.2f}"
                  f"{g['dd']:>8.0f}%", flush=True)

    print("\n" + "=" * 78)
    print("2. CONVICTION WEIGHT k  (0 = equal weight across the 3)")
    print("=" * 78)
    print(f"  {'k':<10}{'GROWTH':>10}{'vol':>8}{'maxDD':>9}{'Sharpe':>9}")
    print("  " + "-" * 46, flush=True)
    for k in (0.0, 0.25, 0.5, 1.0, 2.0):
        g = growth(rebal_backtest(clean, S2.sc_mom12(clean), top=3, k=k))
        if g:
            print(f"  {k:<10.2f}{g['growth']:>9.1f}%{g['vol']:>7.0f}%"
                  f"{g['dd']:>8.0f}%{g['sharpe']:>9.2f}", flush=True)

    print("\n" + "=" * 78)
    print("3. REBALANCE FREQUENCY")
    print("=" * 78)
    print(f"  {'frequency':<16}{'GROWTH':>10}{'vol':>8}{'maxDD':>9}{'Sharpe':>9}")
    print("  " + "-" * 52, flush=True)
    for freq, lab in (("W-FRI", "weekly"), ("2W-FRI", "biweekly"),
                      ("ME", "monthly"), ("QE", "quarterly")):
        g = growth(rebal_backtest(clean, S2.sc_mom12(clean), top=3, freq=freq))
        if g:
            print(f"  {lab:<16}{g['growth']:>9.1f}%{g['vol']:>7.0f}%"
                  f"{g['dd']:>8.0f}%{g['sharpe']:>9.2f}", flush=True)

    print("\n" + "=" * 78)
    print("4. STOP WIDTH  (one name is a third of the book)")
    print("=" * 78)
    print(f"  {'stop':<16}{'GROWTH':>10}{'vol':>8}{'maxDD':>9}{'Sharpe':>9}")
    print("  " + "-" * 52, flush=True)
    for ps, lab in ((0.03, "3%"), (0.06, "6%"), (0.10, "10%"),
                    (0.15, "15%"), (0.99, "none")):
        g = growth(rebal_backtest(clean, S2.sc_mom12(clean), top=3, pstop=ps))
        if g:
            print(f"  {lab:<16}{g['growth']:>9.1f}%{g['vol']:>7.0f}%"
                  f"{g['dd']:>8.0f}%{g['sharpe']:>9.2f}", flush=True)

    print("\n" + "=" * 78)
    print("5. PLACEBO -- random 3 names, same engine")
    print("=" * 78, flush=True)
    rng = np.random.default_rng(17)
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    gs, shs = [], []
    for i in range(N):
        sc = pd.DataFrame(rng.standard_normal(clean.shape),
                          index=clean.index, columns=clean.columns)
        g = growth(rebal_backtest(clean, sc, top=3))
        if g:
            gs.append(g["growth"]); shs.append(g["sharpe"])
        if (i + 1) % 15 == 0:
            print(f"    ...{i+1}/{N} draws", flush=True)
    gs, shs = np.array(gs), np.array(shs)
    bg = (gs >= g_real["growth"]).sum()
    bs = (shs >= g_real["sharpe"]).sum()
    print(f"\n  real: growth {g_real['growth']:.1f}%  Sharpe {g_real['sharpe']:.2f}")
    print(f"  random triples beating real GROWTH: {bg}/{len(gs)} = "
          f"{bg/len(gs)*100:.1f}%")
    print(f"  random triples beating real SHARPE: {bs}/{len(shs)} = "
          f"{bs/len(shs)*100:.1f}%")
    print(f"  random growth: mean {gs.mean():.1f}%  sd {gs.std():.1f}  "
          f"95th pct {np.percentile(gs,95):.1f}%")
    print(f"\n  under 5% = the momentum signal, not luck of the draw")


if __name__ == "__main__":
    main()
