"""COMBINING SIGNALS THAT CARRY INDEPENDENT INFORMATION.

Eight candlestick patterns combined give you nothing, because each carries
zero and they are all the same information -- price shape -- rearranged.

These four are different. Each comes from a distinct mechanism, so their
errors should be uncorrelated, and uncorrelated edges combine as roughly
Sharpe x sqrt(n):

  CALENDAR    turn-of-month + Monday. Pension and payroll money that moves on
              a schedule regardless of price. +9.55bp, t=3.39.
  OPEN_REV    fade the first 30 minutes in the second 30. Intraday liquidity,
              nothing to do with the calendar. +1.27 to +1.78bp.
  SPREAD      index-vs-index relative value. Cross-sectional, not directional.
  MOM_FADE    the failed JFE intraday-momentum effect, traded in reverse.

The test is not whether each is impressive alone -- they are not. It is
whether their CORRELATIONS are near zero, because that is what decides
whether combining them helps.

Costs 0.5bp per leg (real index-futures cost).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

COST_BP = 0.5
OPEN = 9*60+30
MK = {"S&P 500": "usa500idxusd", "NASDAQ": "usatechidxusd",
      "DOW": "usa30idxusd", "DAX": "deuidxeur"}


def load(slug):
    fs = [f for f in glob.glob(f"download/{slug}-m5-bid-*.csv") if "2026-07-24" in f]
    d = pd.read_csv(max(fs, key=os.path.getsize))
    d.columns = [c.lower() for c in d.columns]
    ts = d["timestamp"]
    idx = (pd.to_datetime(ts, unit="ms", utc=True)
           if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts, utc=True))
    d.index = idx.dt.tz_convert("America/New_York")
    d = d[["open", "high", "low", "close"]].astype(float).sort_index()
    assert d.index[0].year > 2000
    return d


def daily_legs(d):
    """Per-session pieces every signal is built from."""
    m = d.index.hour*60 + d.index.minute
    day = d.index.normalize().tz_localize(None)

    def seg(a, b):
        k = (m >= a) & (m < b)
        g = d[k].groupby(day[k])
        return g["open"].first(), g["close"].last()

    o1, c1 = seg(OPEN, OPEN+30)
    o2, c2 = seg(OPEN+30, OPEN+60)
    oS, cS = seg(OPEN, 16*60)
    L = pd.DataFrame({"leg1": (c1/o1-1)*1e4, "leg2": (c2/o2-1)*1e4,
                      "sess": (cS/oS-1)*1e4}).dropna()
    return L


def tag_calendar(idx):
    s = pd.DataFrame(index=idx)
    s["monday"] = idx.dayofweek == 0
    s["tom"] = False
    for _, g in pd.Series(idx, index=idx).groupby([idx.year, idx.month]):
        d = pd.DatetimeIndex(g.values)
        s.loc[d[:3], "tom"] = True
        s.loc[d[-1:], "tom"] = True
    return s


def build_signals():
    """Daily return series for each independent signal, per market."""
    out = {}
    for nm, slug in MK.items():
        d = load(slug)
        L = daily_legs(d)
        cal = tag_calendar(L.index)

        # 1. CALENDAR -- long the session on Monday / turn-of-month
        fire = (cal["monday"] | cal["tom"]).values
        calendar = np.where(fire, L["sess"].values - COST_BP, np.nan)

        # 2. OPEN_REV -- fade leg1 in leg2, every day
        openrev = -np.sign(L["leg1"].values) * L["leg2"].values - COST_BP

        # 3. MOM_FADE -- fade the opening hour over the rest of the session
        hour = L["leg1"].values + L["leg2"].values
        rest = L["sess"].values - hour
        momfade = -np.sign(hour) * rest - COST_BP

        out[nm] = pd.DataFrame({"calendar": calendar, "open_rev": openrev,
                                "mom_fade": momfade}, index=L.index)
    return out


def add_spread(S):
    """4. SPREAD -- fade the day's relative move between two indices."""
    a = S["NASDAQ"].index.intersection(S["S&P 500"].index)
    dn = load(MK["NASDAQ"]); ds = load(MK["S&P 500"])
    Ln, Ls = daily_legs(dn), daily_legs(ds)
    j = Ln.index.intersection(Ls.index)
    rel1 = (Ln.loc[j, "leg1"] - Ls.loc[j, "leg1"]).values      # divergence
    rel2 = (Ln.loc[j, "leg2"] - Ls.loc[j, "leg2"]).values      # reversion?
    sp = -np.sign(rel1) * rel2 - 2*COST_BP                     # two legs
    for nm in S:
        S[nm] = S[nm].join(pd.Series(sp, index=j, name="spread"), how="left")
    return S


def stats(x, per_year):
    x = pd.Series(x).dropna().values
    if len(x) < 30:
        return None
    m, sd = x.mean(), x.std(ddof=1)
    se = sd/np.sqrt(len(x))
    return dict(n=len(x), mean=m, t=m/se,
                sharpe=m/sd*np.sqrt(per_year),
                ann=m*per_year/100)


if __name__ == "__main__":
    S = add_spread(build_signals())
    # pool markets by stacking each signal's daily returns
    names = ["calendar", "open_rev", "mom_fade", "spread"]
    pooled = {}
    for k in names:
        parts = []
        for nm in MK:
            if k in S[nm]:
                parts.append(S[nm][k].rename(nm))
        pooled[k] = pd.concat(parts, axis=1).mean(axis=1)   # equal-weight markets

    P = pd.DataFrame(pooled).dropna(how="all")
    print("INDIVIDUAL SIGNALS (equal-weight across markets)\n")
    print(f"{'signal':<12}{'n':>6}{'mean bp':>10}{'t':>8}{'Sharpe':>9}{'ann %':>9}")
    print("-"*54)
    freq = {"calendar": 89, "open_rev": 252, "mom_fade": 252, "spread": 252}
    for k in names:
        s = stats(P[k], freq[k])
        if s:
            print(f"{k:<12}{s['n']:>6}{s['mean']:>10.2f}{s['t']:>8.2f}"
                  f"{s['sharpe']:>9.2f}{s['ann']:>9.2f}")

    print("\n" + "="*66)
    print("CORRELATION MATRIX -- the whole question")
    print("="*66)
    C = P[names].corr()
    print(C.round(3).to_string())
    off = C.values[np.triu_indices_from(C.values, 1)]
    print(f"\n  mean |correlation| off-diagonal: {np.abs(off).mean():.3f}")
    print("  (near 0 = independent = combining genuinely helps)")

    print("\n" + "="*66)
    print("COMBINED PORTFOLIO (equal risk, daily)")
    print("="*66)
    Z = P[names].copy()
    for k in names:                       # scale each to unit vol
        Z[k] = Z[k] / Z[k].std()
    combo = Z.mean(axis=1).dropna()
    s = stats(combo, 252)
    print(f"  n={s['n']}  Sharpe {s['sharpe']:+.2f}  t={s['t']:+.2f}")
    print("\n  vs best single signal:")
    best = max(names, key=lambda k: (stats(P[k], freq[k]) or {"sharpe": -9})["sharpe"])
    bs = stats(P[best], freq[best])
    print(f"    {best}: Sharpe {bs['sharpe']:+.2f}")
    print(f"\n  theory: {len(names)} uncorrelated signals -> "
          f"Sharpe x sqrt({len(names)}) = x{np.sqrt(len(names)):.2f}")
