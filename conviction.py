"""TRADING WITH CONVICTION -- concentrate hard on the best names.

Fair criticism: every system built here holds 20+ positions and behaves like an
index fund with a tilt. Real traders who compound fast hold 3-5 positions and
size up on their best idea. And concentration DID help when tested on the ETF
book -- 1 position grew at 15.5%/yr vs 11.4% for six, because the top-conviction
pick genuinely has a higher mean and that outweighed the added variance.

That was never run on the stock universe, where the cross-section is far
bigger and the top decile far more selective. This tests it properly:

  1. TOP-N SWEEP from 1 to 30 names, measuring GROWTH RATE (what compounds)
     rather than average return (which always flatters concentration).
  2. SURVIVORSHIP CONTROL on every row, because concentration loads hardest
     on exactly the names survivorship bias smuggles in -- the mega-winners.
  3. DEV/HOLDOUT, since a concentrated result is far easier to overfit.
  4. RUIN RISK: with 3 names a single blowup is 33% of the book.

Also calibrates against what the best documented traders ACTUALLY earn, so
the result can be judged against reality rather than against a hope.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import stocks as S
import stocks2 as S2


def growth(x, ann=252):
    """Log growth -- what actually compounds, unlike the arithmetic mean."""
    x = pd.Series(x).dropna()
    if len(x) < 200:
        return None
    x = np.clip(x, -0.99, None)
    eq = (1 + x).cumprod()
    return dict(growth=(np.exp(np.log1p(x).mean() * ann) - 1) * 100,
                avg=x.mean() * ann * 100,
                vol=x.std(ddof=1) * np.sqrt(ann) * 100,
                dd=(eq / eq.cummax() - 1).min() * 100,
                sharpe=x.mean() / x.std(ddof=1) * np.sqrt(ann),
                final=eq.iloc[-1])


if __name__ == "__main__":
    px, vol = S.load(S.UNIV)
    print(f"universe: {px.shape[1]} stocks, {px.index.min():%Y-%m} -> "
          f"{px.index.max():%Y-%m}\n", flush=True)

    total = (px.iloc[-1] / px.iloc[0] - 1).sort_values(ascending=False)
    clean = px[list(total.index[30:])]        # survivorship-controlled

    print("=" * 84)
    print("1. CONCENTRATION SWEEP  (full universe -- contains survivorship bias)")
    print("=" * 84)
    print(f"  {'hold':<10}{'avg ret':>10}{'GROWTH':>10}{'vol':>8}{'maxDD':>8}"
          f"{'Sharpe':>9}{'$10k ->':>13}")
    print("  " + "-" * 68)
    for n in (1, 2, 3, 5, 10, 20, 30):
        r = S2.backtest(px, S2.sc_mom12(px), top=n)
        g = growth(r)
        if g:
            print(f"  top {n:<6}{g['avg']:>9.1f}%{g['growth']:>9.1f}%{g['vol']:>7.0f}%"
                  f"{g['dd']:>7.0f}%{g['sharpe']:>9.2f}{10000*g['final']:>13,.0f}")

    print("\n" + "=" * 84)
    print("2. SAME SWEEP, SURVIVORSHIP-CONTROLLED  (top 30 winners removed)")
    print("=" * 84)
    print(f"  {'hold':<10}{'avg ret':>10}{'GROWTH':>10}{'vol':>8}{'maxDD':>8}"
          f"{'Sharpe':>9}{'$10k ->':>13}")
    print("  " + "-" * 68)
    best = None
    for n in (1, 2, 3, 5, 10, 20, 30):
        r = S2.backtest(clean, S2.sc_mom12(clean), top=n)
        g = growth(r)
        if not g:
            continue
        if best is None or g["growth"] > best[0]:
            best = (g["growth"], n)
        print(f"  top {n:<6}{g['avg']:>9.1f}%{g['growth']:>9.1f}%{g['vol']:>7.0f}%"
              f"{g['dd']:>7.0f}%{g['sharpe']:>9.2f}{10000*g['final']:>13,.0f}")
    print(f"\n  best growth at top {best[1]} ({best[0]:.1f}%/yr)")

    print("\n" + "=" * 84)
    print("3. DEV / HOLDOUT -- does concentration survive out of sample?")
    print("=" * 84)
    cb = clean.pct_change().mean(axis=1).dropna()
    print(f"  {'hold':<10}{'DEV growth':>13}{'HOLDOUT growth':>17}{'stable?':>12}")
    print("  " + "-" * 54)
    for n in (1, 3, 5, 10, 20):
        r = S2.backtest(clean, S2.sc_mom12(clean), top=n)
        a = growth(r.loc[:S.DEV])
        b = growth(r.loc[S.DEV:])
        ba = growth(cb.loc[:S.DEV])
        bb = growth(cb.loc[S.DEV:])
        if not (a and b and ba and bb):
            continue
        ea, eb = a["growth"] - ba["growth"], b["growth"] - bb["growth"]
        print(f"  top {n:<6}{ea:>+12.1f}%{eb:>+16.1f}%"
              f"{'YES' if ea > 0 and eb > 0 else 'no':>12}")

    print("\n" + "=" * 84)
    print("4. WHAT THE BEST DOCUMENTED TRADERS ACTUALLY EARN")
    print("=" * 84)
    print("  Audited, long-horizon, net of fees:\n")
    for name, cagr, yrs in (("Renaissance Medallion (best ever recorded)", 39, 30),
                            ("Soros Quantum Fund", 30, 30),
                            ("Warren Buffett / Berkshire", 20, 60),
                            ("Peter Lynch, Magellan", 29, 13),
                            ("typical good hedge fund", 10, 20),
                            ("S&P 500 long run", 10, 100)):
        print(f"    {name:<44}{cagr:>4}%/yr over {yrs} yrs")
    print("\n  $10,000 -> $3,000/month needs 360%/yr, sustained.")
    print("  The best verified record in history is 39%.")
