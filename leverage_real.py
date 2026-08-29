"""GETTING REAL LEVERAGE ONTO THE VERIFIED EDGE.

I said 20x does not exist for retail. That was wrong in an important way:
it does not exist for an ETF/stock margin account, but it absolutely exists
in FUTURES -- a $10k account legally controls $200-400k of index notional,
no PDT rule, no $25k minimum. That is how a trader targets 2-3% of equity per
day off a 0.1% index move.

The catch has never been the leverage. It is that my verified edge is
CROSS-SECTIONAL stock momentum, needing ~172 names to pick 5 from, and futures
offer four correlated index contracts with no cross-section to exploit. The
~50 intraday index strategies tested here all landed at fair odds.

So the real question: can the stock edge be levered by something other than
Reg T margin? Three routes, all retail-accessible:

  1. DEEP ITM LEAPS. A 0.85-delta call a year out behaves like ~3-5x the
     stock with defined downside. Extrinsic value is small deep ITM, so the
     cost is mostly implied financing (~risk-free + 1-2%).
  2. 2x REG T MARGIN. Boring, legal, ~5.5% financing.
  3. PORTFOLIO MARGIN. 6x, but needs $100k, so irrelevant at $10k.

Modelled honestly: leverage multiplies the return AND the drawdown, financing
is charged, and RUIN is tracked -- because a -20.6% system at 4x is -82%, and
leverage that kills the account pays nothing regardless of expectancy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import stocks as S
import stocks2 as S2
from conc2 import rebal_backtest
from conviction import growth

BAND, SLIP = 0.025, 0.00013
RF = 0.045          # risk-free, the floor on any financing cost


def tuned():
    px, _ = S.load(S.UNIV)
    total = (px.iloc[-1] / px.iloc[0] - 1).sort_values(ascending=False)
    clean = px[list(total.index[30:])]
    r = rebal_backtest(clean, S2.sc_mom12(clean), top=5, freq="2W-FRI",
                       pstop=0.99, k=0.25)
    x = r.clip(upper=BAND)
    return x.where(x > -BAND, -(BAND + SLIP))


def lever(r, L, fin):
    """Levered daily returns net of financing on the borrowed portion."""
    return r * L - (L - 1) * fin / 252


if __name__ == "__main__":
    r = tuned()
    g0 = growth(r)
    print(f"tuned system, unlevered: {g0['growth']:.1f}%/yr  "
          f"Sharpe {g0['sharpe']:.2f}  maxDD {g0['dd']:.1f}%\n")

    print("=" * 80)
    print("1. LEVERAGE ROUTES AVAILABLE AT $10,000")
    print("=" * 80)
    routes = [
        (1.0, 0.0, "cash account", "yes"),
        (2.0, 0.055, "2x Reg T margin", "yes"),
        (3.0, RF + 0.015, "deep ITM LEAPS ~3x", "yes, options approval"),
        (4.0, RF + 0.020, "deeper LEAPS ~4x", "yes, more decay risk"),
        (6.0, 0.055, "portfolio margin 6x", "needs $100k"),
    ]
    print(f"  {'route':<24}{'lev':>5}{'fin':>7}{'GROWTH':>10}{'maxDD':>8}"
          f"{'Sharpe':>8}{'$/day on 10k':>14}")
    print("  " + "-" * 76)
    rows = []
    for L, fin, lab, legal in routes:
        x = lever(r, L, fin)
        g = growth(x)
        if not g:
            continue
        per_day = 10000 * (g["growth"] / 100) / 252
        rows.append((g["growth"], L, fin, lab, g, per_day))
        print(f"  {lab:<24}{L:>4.0f}x{fin*100:>6.1f}%{g['growth']:>9.1f}%"
              f"{g['dd']:>7.0f}%{g['sharpe']:>8.2f}{per_day:>13,.0f}")

    print("\n" + "=" * 80)
    print("2. RUIN CHECK -- leverage that kills the account pays nothing")
    print("=" * 80)
    rng = np.random.default_rng(23)
    v = r.values
    print(f"  {'lev':<8}{'P(down 50%)':>14}{'P(down 80%)':>14}"
          f"{'median 1yr':>14}{'p10':>12}")
    print("  " + "-" * 64)
    for L, fin, lab, _ in routes:
        n, days, block = 40000, 252, 5
        nb = (days + block - 1) // block
        idx = rng.integers(0, len(v) - block, size=(n, nb))
        p = np.concatenate([v[idx + j] for j in range(block)], axis=1)[:, :days]
        p = p * L - (L - 1) * fin / 252
        eq = 10000 * np.cumprod(1 + np.clip(p, -0.99, None), axis=1)
        low = eq.min(axis=1)
        print(f"  {L:>3.0f}x{'':<4}{(low < 5000).mean()*100:>13.1f}%"
              f"{(low < 2000).mean()*100:>13.1f}%"
              f"{np.median(eq[:, -1]):>14,.0f}{np.percentile(eq[:,-1],10):>12,.0f}")

    print("\n" + "=" * 80)
    print("3. WHAT EACH ROUTE PAYS PER DAY, AND THE CAPITAL FOR $200/DAY")
    print("=" * 80)
    print(f"  {'route':<24}{'$/day @10k':>13}{'$/day @25k':>13}"
          f"{'capital for $200/day':>22}")
    print("  " + "-" * 74)
    for gr, L, fin, lab, g, per_day in rows:
        need = 200 * 252 / (gr / 100)
        print(f"  {lab:<24}{per_day:>12,.0f}{per_day*2.5:>13,.0f}"
              f"{need:>21,.0f}")

    print("\n" + "=" * 80)
    print("4. THE FUTURES ROUTE -- why the leverage is real but the edge is not")
    print("=" * 80)
    print("""  Micro futures (MES/MNQ) genuinely give 20-40x on a $10k account,
  legally, today. That part of the picture I had wrong.

  What futures do NOT give is a cross-section. The verified edge picks 5
  names out of 172 by relative momentum; with four correlated index
  contracts there is nothing to rank. Every intraday index strategy tested
  in this project -- ~50 of them, including the full TJR methodology at
  48.4% win over 10,645 trades -- landed at or below fair odds.

  So the honest position: the leverage exists, and I have no edge that
  survives on the instrument that offers it.""")
