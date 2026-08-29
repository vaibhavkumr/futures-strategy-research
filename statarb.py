"""STATISTICAL ARBITRAGE between the index futures.

Every previous build asked "which way is this market going?" -- a directional
forecast. Eight approaches all landed on fair odds, and a published JFE
intraday effect failed to replicate on the same data. That question looks
genuinely unanswerable from free OHLC.

This asks a structurally different one. NASDAQ, S&P and DOW are driven by
largely the same factors, so the SPREAD between them is anchored in a way the
level is not. When one runs ahead of the other, we are not predicting the
future -- we are betting a mechanical relationship reasserts itself.

That matters because the payoff does not need direction to be forecastable.
It needs the spread to be mean-reverting, which is a testable property of the
data rather than a claim about the future.

Every statistic is computed from PAST bars only (rolling windows are shifted),
and the leak audit that caught the opening-range bug applies here too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from duka import load

MK = {"NASDAQ": "usatechidxusd", "S&P 500": "usa500idxusd",
      "DOW": "usa30idxusd", "DAX": "deuidxeur"}

PAIRS = [("NASDAQ", "S&P 500"), ("NASDAQ", "DOW"), ("S&P 500", "DOW"),
         ("NASDAQ", "DAX"), ("S&P 500", "DAX"), ("DOW", "DAX")]

# two legs, so two round trips of commission. MNQ/MES ~ $1.28 each.
# expressed in basis points of notional, deliberately pessimistic.
COST_BP = 2.0


def align():
    px = {}
    for name, slug in MK.items():
        d = load(slug, "m5")
        mins = d.index.hour * 60 + d.index.minute
        px[name] = d[(mins >= 570) & (mins < 960)]["close"]   # NY session
    df = pd.DataFrame(px).dropna()
    return np.log(df)


def spread(lp: pd.DataFrame, a: str, b: str, beta_win=390, z_win=78):
    """log(A) - beta*log(B), z-scored. beta and z use only PAST bars."""
    x, y = lp[b], lp[a]
    # rolling hedge ratio, shifted so the current bar is never in its own fit
    cov = x.rolling(beta_win).cov(y).shift(1)
    var = x.rolling(beta_win).var().shift(1)
    beta = (cov / var).clip(0.2, 5.0)
    s = y - beta * x
    mu = s.rolling(z_win).mean().shift(1)
    sd = s.rolling(z_win).std().shift(1)
    z = (s - mu) / sd
    return s, z.replace([np.inf, -np.inf], np.nan)


def half_life(s: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life. Is the spread mean-reverting at all?"""
    s = s.dropna()
    ds = s.diff().dropna()
    lag = s.shift(1).dropna().loc[ds.index]
    b = np.polyfit(lag, ds, 1)[0]
    return -np.log(2) / b if b < 0 else np.inf


def backtest(lp, a, b, entry_z=2.0, exit_z=0.5, stop_z=4.0,
             max_hold=78, cost_bp=COST_BP):
    """Short the spread when rich, long when cheap. Flat by session end."""
    s, z = spread(lp, a, b)
    idx = lp.index
    zz, ss = z.values, s.values
    # P&L BUG: beta is re-estimated every bar, so (s[k] - s[i]) mixes real
    # price moves with changes in beta*log(price). A 0.01 beta drift moves it
    # ~850bp of pure artifact, which produced -103% "per trade". The position
    # holds its hedge ratio FIXED at entry, so P&L must too.
    la, lb = lp[a].values, lp[b].values
    bser = ((lp[b].rolling(390).cov(lp[a]).shift(1))
            / (lp[b].rolling(390).var().shift(1))).clip(0.2, 5.0).values
    rows = []
    i = 0
    n = len(idx)
    while i < n - 1:
        if not np.isfinite(zz[i]) or abs(zz[i]) < entry_z:
            i += 1
            continue
        sgn = -1 if zz[i] > 0 else 1        # fade the stretch
        entry_s = ss[i]
        out = None
        for k in range(i + 1, min(i + 1 + max_hold, n)):
            if idx[k].date() != idx[i].date():
                out = (k, "eod")
                break
            if abs(zz[k]) <= exit_z:
                out = (k, "revert")
                break
            if abs(zz[k]) >= stop_z and np.sign(zz[k]) == np.sign(zz[i]):
                out = (k, "stop")
                break
        if out is None:
            out = (min(i + max_hold, n - 1), "timeout")
        k, why = out
        # spread pnl in log terms -> basis points of notional, minus costs
        be = bser[i]
        if not np.isfinite(be):
            i = k + 1
            continue
        # return of the actual position: long/short A hedged with beta*B
        pnl_bp = sgn * ((la[k] - la[i]) - be * (lb[k] - lb[i])) * 1e4 - cost_bp
        rows.append(dict(ts=idx[i], pair=f"{a}/{b}", z=zz[i], bars=k - i,
                         why=why, pnl_bp=pnl_bp))
        i = k + 1
    return pd.DataFrame(rows)


def rep(x, label, ann=252 * 78):
    x = np.asarray(x, float)
    if len(x) < 10:
        print(f"  {label:<30} n={len(x)}")
        return np.nan
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    print(f"  {label:<30} n={len(x):<6} mean {m:+7.2f}bp  t={m/se:+6.2f}  "
          f"win {(x>0).mean()*100:5.1f}%  total {x.sum()/1e4*100:+7.1f}%")
    return m


if __name__ == "__main__":
    lp = align()
    print(f"aligned {len(lp):,} 5-min NY-session bars, "
          f"{lp.index.min():%Y-%m-%d} -> {lp.index.max():%Y-%m-%d}\n")

    print("=" * 76)
    print("STEP 1  is the spread even mean-reverting? (half-life in bars)")
    print("=" * 76)
    for a, b in PAIRS:
        s, z = spread(lp, a, b)
        # half-life on a FIXED-beta spread; a drifting beta contaminates it
        hl = half_life(lp[a] - np.polyfit(lp[b], lp[a], 1)[0] * lp[b])
        corr = lp[a].diff().corr(lp[b].diff())
        print(f"  {a+'/'+b:<20} return corr {corr:+.3f}   "
              f"half-life {hl:8.1f} bars ({hl*5/60:5.1f} h)")

    print("\n" + "=" * 76)
    print("STEP 2  trade it. DEV = 2022-2024, HOLDOUT = 2025-2026")
    print("        costs: 2.0bp round trip (two legs), deliberately pessimistic")
    print("=" * 76)
    allt = []
    for a, b in PAIRS:
        t = backtest(lp, a, b)
        if len(t):
            allt.append(t)
    T = pd.concat(allt, ignore_index=True)
    T["ts"] = pd.to_datetime(T["ts"])
    dev = T[T.ts < "2025-01-01"]
    hold = T[T.ts >= "2025-01-01"]

    print("\n  per pair (DEV only):")
    for p, g in dev.groupby("pair"):
        rep(g.pnl_bp, f"  {p}")
    print()
    rep(dev.pnl_bp, "DEV pooled")
    print()
    print("  per pair (HOLDOUT):")
    for p, g in hold.groupby("pair"):
        rep(g.pnl_bp, f"  {p}")
    print()
    rep(hold.pnl_bp, "HOLDOUT pooled")

    print("\n  exit reason mix (holdout):")
    print(hold.groupby("why").pnl_bp.agg(["count", "mean"]).round(2).to_string())

    print("\n" + "=" * 76)
    print("STEP 3  cost sensitivity -- how much edge survives real friction?")
    print("=" * 76)
    for c in (0.0, 1.0, 2.0, 3.0, 5.0):
        x = hold.pnl_bp + COST_BP - c
        rep(x, f"  holdout @ {c:.1f}bp round trip")
