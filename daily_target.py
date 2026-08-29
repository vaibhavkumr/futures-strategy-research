"""A DAILY PROFIT TARGET -- "make $250, then stop for the day."

Intuitively appealing: bank the win, avoid giving it back. But it does
something specific and asymmetric to the return distribution:

    it TRUNCATES the right tail   (up days stop at +$250)
    it leaves the left tail ALONE (down days run their full course)

Whether that helps depends entirely on where the strategy's return lives. If
returns are spread evenly across days, capping costs little. If they are
concentrated in a handful of explosive days -- which is the documented shape of
momentum -- then capping removes exactly the days that produce the compounding.

Three tests:
  1. CONCENTRATION. What share of total return comes from the best 1%, 5%,
     10% of days? This decides the answer before any simulation.
  2. THE CAP ITSELF. Apply daily profit caps from $100 to $1000 on a $10,000
     account and measure what survives.
  3. THE MIRROR. A daily LOSS limit truncates the left tail instead. If
     asymmetry is the mechanism, this should behave oppositely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import stocks as S
import stocks2 as S2
from conc2 import rebal_backtest
from conviction import growth

TUNED = dict(top=3, freq="2W-FRI", pstop=0.99, k=0.25)


def apply_cap(r, cap_pct=None, floor_pct=None):
    """Cap daily gains and/or floor daily losses, as fractions of equity."""
    x = r.copy()
    if cap_pct is not None:
        x = x.clip(upper=cap_pct)
    if floor_pct is not None:
        x = x.clip(lower=-floor_pct)
    return x


if __name__ == "__main__":
    px, _ = S.load(S.UNIV)
    total = (px.iloc[-1]/px.iloc[0]-1).sort_values(ascending=False)
    clean = px[list(total.index[30:])]
    r = rebal_backtest(clean, S2.sc_mom12(clean), **TUNED)
    g0 = growth(r)
    print(f"tuned system: {g0['growth']:.1f}%/yr growth, Sharpe {g0['sharpe']:.2f}, "
          f"{len(r):,} trading days\n")

    print("=" * 78)
    print("1. WHERE DOES THE RETURN COME FROM?")
    print("=" * 78)
    srt = r.sort_values(ascending=False)
    tot = np.log1p(np.clip(r, -0.99, None)).sum()
    print(f"  {'excluding':<28}{'total log return':>20}{'share lost':>14}")
    print("  " + "-" * 62)
    for n_pct, lab in ((0.01, "best 1% of days"), (0.05, "best 5% of days"),
                       (0.10, "best 10% of days")):
        n = int(len(r)*n_pct)
        kept = r.drop(srt.index[:n])
        t2 = np.log1p(np.clip(kept, -0.99, None)).sum()
        print(f"  {lab:<28}{t2:>20.2f}{(1-t2/tot)*100:>13.0f}%")
    print(f"  {'(full sample)':<28}{tot:>20.2f}")
    print(f"\n  best day  {r.max()*100:+.2f}%   worst day {r.min()*100:+.2f}%")
    print(f"  on $10,000 the best day was ${r.max()*10000:+,.0f}")

    print("\n" + "=" * 78)
    print("2. DAILY PROFIT TARGET on a $10,000 account")
    print("=" * 78)
    print("  'stop for the day once up $X' = cap the daily gain at X/10000\n")
    print(f"  {'daily target':<20}{'days it fires':>15}{'GROWTH':>10}"
          f"{'Sharpe':>9}{'vs uncapped':>13}")
    print("  " + "-" * 68)
    for dollars in (100, 200, 250, 300, 500, 1000):
        cap = dollars/10000
        g = growth(apply_cap(r, cap_pct=cap))
        fires = (r > cap).mean()*100
        if g:
            print(f"  ${dollars:<19,}{fires:>14.1f}%{g['growth']:>9.1f}%"
                  f"{g['sharpe']:>9.2f}{g['growth']-g0['growth']:>+12.1f}%")

    print("\n" + "=" * 78)
    print("3. THE MIRROR -- a daily LOSS limit instead")
    print("=" * 78)
    print(f"  {'daily loss limit':<20}{'days it fires':>15}{'GROWTH':>10}"
          f"{'Sharpe':>9}{'vs uncapped':>13}")
    print("  " + "-" * 68)
    for dollars in (100, 200, 250, 300, 500, 1000):
        fl = dollars/10000
        g = growth(apply_cap(r, floor_pct=fl))
        fires = (r < -fl).mean()*100
        if g:
            print(f"  ${dollars:<19,}{fires:>14.1f}%{g['growth']:>9.1f}%"
                  f"{g['sharpe']:>9.2f}{g['growth']-g0['growth']:>+12.1f}%")

    print("\n" + "=" * 78)
    print("4. BOTH -- cap gains AND limit losses at $250")
    print("=" * 78)
    g = growth(apply_cap(r, cap_pct=0.025, floor_pct=0.025))
    print(f"  symmetric +/-$250 band: growth {g['growth']:.1f}%/yr  "
          f"Sharpe {g['sharpe']:.2f}  vs uncapped {g['growth']-g0['growth']:+.1f}%")

    print("\n" + "=" * 78)
    print("5. COULD IT EVER PAY $250/DAY?")
    print("=" * 78)
    print(f"  $250/day x 252 days = $63,000/yr = 630% on $10,000")
    print(f"  tuned system earns {g0['growth']:.1f}%/yr")
    need = 63000/(g0['growth']/100)
    print(f"  -> $250 EVERY day needs ${need:,.0f} of capital")
    print(f"  on $10,000 the average day is ${r.mean()*10000:+.2f}, "
          f"median ${r.median()*10000:+.2f}")
    print(f"  days above +$250: {(r>0.025).mean()*100:.1f}%   "
          f"days below -$250: {(r<-0.025).mean()*100:.1f}%")
