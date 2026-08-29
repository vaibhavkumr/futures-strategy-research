"""STRESS-TESTING THE PRE-FOMC RESULT.

Event-level: n=35, +25.84bp, t=+2.69, positive in all 4 markets and all 5
years. Promising. But it failed Bonferroni across 15 tests, and the offset
(2 days before) does not match Lucca & Moench's published 24-hour window --
so it might be a real effect that migrated, or it might be the one test in
fifteen that came up.

Six tests, and the placebo is the one that matters:

  1. OFFSET PROFILE     day -6 .. +3. A real event effect should look like a
                        structured shape, not a lone spike.
  2. PLACEBO            35 RANDOM dates, same search over offsets, repeated
                        many times. This measures empirically how often a
                        t of 2.69 appears from nothing -- which is the honest
                        version of a multiple-comparison correction.
  3. OUTLIERS           median and trimmed mean. One 2022 crash day could
                        carry the whole thing.
  4. NON-EQUITY         gold and GBPUSD. Fed policy should move those too;
                        if the effect is equity-only that is informative
                        either way.
  5. SPLIT              first half vs second half of the sample.
  6. INDEPENDENCE       correlation with the calendar signal, since the whole
                        point is whether it adds to a portfolio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import structural_hunt as H

MK_ALL = dict(H.MK)
MK_ALL.update({"GOLD": "xauusd", "GBPUSD": "gbpusd"})


def event_returns(S, dates, offset):
    """One observation per event: equal-weight across markets."""
    tgt = pd.DatetimeIndex(dates) - pd.Timedelta(days=offset)
    cols = {}
    for nm, s in S.items():
        m = np.asarray(s.index.isin(tgt))
        if m.sum():
            cols[nm] = pd.Series(s.r.values[m] - H.COST_BP, index=s.index[m])
    if not cols:
        return pd.Series(dtype=float)
    D = pd.DataFrame(cols)
    return D.mean(axis=1).dropna()


def tstat(x):
    x = np.asarray(x, float)
    if len(x) < 8:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


if __name__ == "__main__":
    S = {nm: H.sessions(sl) for nm, sl in H.MK.items()}
    Sall = {nm: H.sessions(sl) for nm, sl in MK_ALL.items()}
    fomc = pd.DatetimeIndex([pd.Timestamp(x) for x in H.FOMC])

    print("=" * 72)
    print("1. OFFSET PROFILE -- is it a shape or a lone spike?")
    print("=" * 72)
    print(f"  {'offset':<12}{'n':>5}{'mean bp':>11}{'t':>8}{'win%':>8}")
    print("  " + "-" * 42)
    for off in range(6, -4, -1):
        ev = event_returns(S, fomc, off)
        if len(ev) < 8:
            continue
        lab = f"T-{off}" if off > 0 else ("T" if off == 0 else f"T+{-off}")
        print(f"  {lab:<12}{len(ev):>5}{ev.mean():>11.2f}{tstat(ev):>8.2f}"
              f"{(ev>0).mean()*100:>8.0f}")

    print("\n" + "=" * 72)
    print("2. PLACEBO -- how often does t>=2.69 appear from RANDOM dates?")
    print("=" * 72)
    rng = np.random.default_rng(7)
    alldays = S["S&P 500"].index
    best_ts = []
    for trial in range(300):
        fake = pd.DatetimeIndex(rng.choice(alldays, size=len(fomc), replace=False))
        # same search I did: try 10 offsets, keep the best
        ts = []
        for off in range(6, -4, -1):
            ev = event_returns(S, fake, off)
            if len(ev) >= 8:
                ts.append(abs(tstat(ev)))
        if ts:
            best_ts.append(max(ts))
    best_ts = np.array(best_ts)
    obs = 2.69
    print(f"  300 placebo runs, each searching 10 offsets")
    print(f"  best |t| from random dates: median {np.median(best_ts):.2f}, "
          f"90th pct {np.percentile(best_ts,90):.2f}, max {best_ts.max():.2f}")
    print(f"  fraction of placebos reaching |t| >= {obs}: "
          f"{(best_ts>=obs).mean()*100:.1f}%")
    print(f"  -> empirical p-value for the FOMC finding: "
          f"{(best_ts>=obs).mean():.3f}")

    print("\n" + "=" * 72)
    print("3. OUTLIERS -- is one day carrying it?")
    print("=" * 72)
    ev = event_returns(S, fomc, 2)
    x = np.sort(ev.values)
    print(f"  mean {ev.mean():+.2f}bp   median {np.median(x):+.2f}bp")
    print(f"  drop best event : {np.mean(x[:-1]):+.2f}bp")
    print(f"  drop best 3     : {np.mean(x[:-3]):+.2f}bp")
    print(f"  trimmed 10%     : {np.mean(x[2:-2]):+.2f}bp")
    print(f"  largest 3 moves : {x[-3:].round(1)}")

    print("\n" + "=" * 72)
    print("4. NON-EQUITY MARKETS")
    print("=" * 72)
    for nm in ("GOLD", "GBPUSD"):
        s = Sall[nm]
        m = np.asarray(s.index.isin(fomc - pd.Timedelta(days=2)))
        x = s.r.values[m] - H.COST_BP
        if len(x) >= 10:
            print(f"  {nm:<10} n={len(x):<4} {x.mean():+7.2f}bp  t={tstat(x):+5.2f}")

    print("\n" + "=" * 72)
    print("5. SPLIT-SAMPLE")
    print("=" * 72)
    mid = ev.index[len(ev)//2]
    for lab, g in (("first half", ev[ev.index <= mid]), ("second half", ev[ev.index > mid])):
        print(f"  {lab:<12} n={len(g):<4} {g.mean():+7.2f}bp  t={tstat(g):+5.2f}")

    print("\n" + "=" * 72)
    print("6. INDEPENDENCE from the calendar signal")
    print("=" * 72)
    cal_days = []
    for nm, s in S.items():
        idx = s.index
        tom = np.zeros(len(idx), bool)
        for _, g in pd.Series(idx, index=idx).groupby([idx.year, idx.month]):
            d = pd.DatetimeIndex(g.values)
            tom[idx.isin(d[:3])] = True
            tom[idx.isin(d[-1:])] = True
        cal_days.append(pd.Series(tom | (idx.dayofweek == 0), index=idx))
    cal = cal_days[0]
    overlap = cal.reindex(ev.index).fillna(False).mean()
    print(f"  share of pre-FOMC days that are ALSO calendar days: {overlap*100:.0f}%")
    print("  (low overlap = genuinely separate signal for a portfolio)")
