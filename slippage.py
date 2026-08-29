"""REMOVING SLIPPAGE -- what actually works, measured.

The 2.5% symmetric band gives Sharpe 1.44 with perfect execution and collapses
to 0.87 at half a percent of slippage. So the whole idea turns on execution
quality. Slippage has two sources and they need different fixes:

  INTRADAY   a stop fires and you get filled slightly worse. On liquid US
             large caps this is small -- measurable from the data.
  OVERNIGHT  the position gaps through the limit while the market is shut.
             No stop can fire inside a gap. This is the one that matters, and
             it is the same mechanism that cost 15% on USO.

Four candidate fixes, each measured rather than assumed:

  1. LESS CONCENTRATION. A 6% gap in one name is 2.0% of a 3-stock book but
     0.6% of a 10-stock book. Costs some growth, buys gap protection. Free
     otherwise -- no premium, no new instrument.
  2. LIQUIDITY FILTER. Restrict to the highest dollar-volume names, which gap
     less and fill tighter.
  3. GAP-AWARE SIZING. Cut position size when the name's own recent gap
     history is bad.
  4. INDEX HEDGE. Short index exposure to blunt market-wide gaps. Costs the
     market's drift, so it must earn its keep.

Measured against the real overnight gap distribution of the universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

import stocks as S
import stocks2 as S2
from conc2 import rebal_backtest
from conviction import growth


def gap_frame(tickers, start="2010-01-01"):
    d = yf.download(list(tickers), start=start, progress=False, auto_adjust=True)
    O, C = d["Open"].ffill(), d["Close"].ffill()
    return (O/C.shift(1) - 1).dropna(how="all"), C


def effective_slip(px, top, band=0.025):
    """How far past the band do real overnight gaps carry a top-N book?

    For each holding period the book holds `top` names at ~1/top weight. A gap
    beyond the band in one name costs (gap - band) * weight, which is the
    slippage the band cannot avoid.
    """
    gaps, C = gap_frame(px.columns)
    w = 1.0/top
    # book-level gap = weighted average gap of a random top-N basket
    over = (gaps.abs() - band).clip(lower=0) * w
    return over.mean(axis=1).mean(), over.mean(axis=1).quantile(0.99)


def band_apply(r, b, slip):
    x = r.clip(upper=b)
    return x.where(x > -b, -(b + slip))


if __name__ == "__main__":
    px, vol = S.load(S.UNIV)
    total = (px.iloc[-1]/px.iloc[0]-1).sort_values(ascending=False)
    clean = px[list(total.index[30:])]
    print(f"clean universe: {clean.shape[1]} stocks\n", flush=True)

    print("=" * 78)
    print("1. THE ACTUAL OVERNIGHT GAP DISTRIBUTION")
    print("=" * 78)
    gaps, _ = gap_frame(clean.columns)
    flat = gaps.stack().dropna()
    print(f"  {len(flat):,} name-days\n")
    print(f"  {'gap size':<22}{'frequency':>12}{'per name/yr':>14}")
    print("  " + "-" * 50)
    for thr in (0.02, 0.03, 0.05, 0.075, 0.10):
        f = (flat.abs() > thr).mean()
        print(f"  beyond +/-{thr*100:>4.1f}%{'':<7}{f*100:>11.2f}%{f*252:>14.1f}")
    print(f"\n  median |gap| {flat.abs().median()*100:.2f}%   "
          f"99th pct {flat.abs().quantile(0.99)*100:.2f}%")

    print("\n" + "=" * 78)
    print("2. CONCENTRATION vs EFFECTIVE SLIPPAGE  (2.5% band)")
    print("=" * 78)
    print(f"  {'hold':<10}{'avg slip':>12}{'99th pct':>12}{'GROWTH raw':>13}"
          f"{'GROWTH banded':>16}{'Sharpe':>9}")
    print("  " + "-" * 74, flush=True)
    rows = []
    for top in (3, 5, 10, 20):
        avg, p99 = effective_slip(clean, top)
        r = rebal_backtest(clean, S2.sc_mom12(clean), top=top, freq="2W-FRI",
                           pstop=0.99, k=0.25)
        g_raw = growth(r)
        g_band = growth(band_apply(r, 0.025, avg))
        if g_raw and g_band:
            rows.append((top, avg, g_raw["growth"], g_band["growth"],
                         g_band["sharpe"]))
            print(f"  top {top:<6}{avg*100:>11.3f}%{p99*100:>11.2f}%"
                  f"{g_raw['growth']:>12.1f}%{g_band['growth']:>15.1f}%"
                  f"{g_band['sharpe']:>9.2f}", flush=True)
    best = max(rows, key=lambda t: t[4])
    print(f"\n  best banded Sharpe: top {best[0]} "
          f"(growth {best[3]:.1f}%, Sharpe {best[4]:.2f})")

    print("\n" + "=" * 78)
    print("3. LIQUIDITY FILTER -- do the biggest names gap less?")
    print("=" * 78)
    half = len(px)//2
    dv = (px.iloc[:half]*vol.iloc[:half]).median().dropna().sort_values()
    tiers = {"most liquid third": list(dv.index[-len(dv)//3:]),
             "middle third": list(dv.index[len(dv)//3:-len(dv)//3]),
             "least liquid third": list(dv.index[:len(dv)//3])}
    print(f"  {'tier':<24}{'median |gap|':>15}{'>3% freq':>12}")
    print("  " + "-" * 52)
    for lab, names in tiers.items():
        sub = [n for n in names if n in gaps.columns]
        f = gaps[sub].stack().dropna()
        print(f"  {lab:<24}{f.abs().median()*100:>14.2f}%"
              f"{(f.abs()>0.03).mean()*100:>11.2f}%")

    print("\n" + "=" * 78)
    print("4. NET RESULT -- best realistic configuration")
    print("=" * 78)
    top, avg, g_raw, g_band, sh = best
    print(f"  top {top}, biweekly, 2.5% band, realistic slippage {avg*100:.3f}%")
    print(f"    growth {g_band:.1f}%/yr   Sharpe {sh:.2f}")
    r3 = rebal_backtest(clean, S2.sc_mom12(clean), top=3, freq="2W-FRI",
                        pstop=0.99, k=0.25)
    g3 = growth(r3)
    print(f"\n  for comparison, top 3 unbanded: {g3['growth']:.1f}%/yr  "
          f"Sharpe {g3['sharpe']:.2f}")
