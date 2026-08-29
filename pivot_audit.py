"""AUDIT THE PIVOT FADE BEFORE BELIEVING IT.

+9.18bp DEV (t=8.53), +9.42bp HOLDOUT (t=6.49), 4/4 markets is exactly the
shape that has been a BUG every time in this session: the ladder leak, the
inverted stops, the Heikin-Ashi execution, the truncation lookahead, the
re-levering artifact. So it gets audited before it gets reported.

Five checks, each targeting a specific failure mode I have already hit:

  1. CORRELATED POOLING. These four indices correlate at 0.856. Pooling them
     and computing a t-stat on the average pretends there are 4x as many
     independent observations as there are. This inflated a t from 3.72 to a
     corrected 2.07 earlier. Per-market t-stats are the honest ones.
  2. TRADE ECONOMICS. Win rate, R:R, trade count. A 1:1 setup returning +9bp
     per trade implies a win rate well above anything else measured here,
     which would be remarkable and is more likely an accounting error.
  3. EXIT REALISM. The sim exits at the bar CLOSE when a level is crossed,
     not at the level. If price gaps past the target, that books a better
     fill than the target -- optimistic. Re-run filling AT the level.
  4. LOOKAHEAD. Recompute the pivot levels from a truncated series and check
     nothing changes.
  5. PLACEBO. Random levels of the same width, same engine. If random
     "pivots" pay similarly, the levels are not what is producing the edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from article_strats import MK, load, sessions, pivots, OPEN, CLOSE, COST_BP


def run_fade(d, S, fill_at_level=False, jitter=0.0, seed=0):
    PV = pivots(S)
    if jitter:
        rng = np.random.default_rng(seed)
        span = (PV.R1 - PV.S1).abs()
        shift = pd.Series(rng.normal(0, jitter, len(PV)), index=PV.index)*span
        PV = PV.add(shift, axis=0)
    m = d.index.hour*60 + d.index.minute
    day = d.index.normalize().tz_localize(None)
    rows = []
    for dt, lv in PV.dropna().iterrows():
        seg = d[(day == dt) & (m >= OPEN) & (m < CLOSE)]
        if len(seg) < 20:
            continue
        c = seg["close"].values
        entry = side = None
        for i in range(1, len(c)-1):
            if entry is None:
                if c[i] <= lv.S1:
                    entry, side = (lv.S1 if fill_at_level else c[i]), 1
                elif c[i] >= lv.R1:
                    entry, side = (lv.R1 if fill_at_level else c[i]), -1
                if entry is not None:
                    tgt, stop = lv.P, (lv.S2 if side > 0 else lv.R2)
            else:
                hit_t = (c[i] >= tgt) if side > 0 else (c[i] <= tgt)
                hit_s = (c[i] <= stop) if side > 0 else (c[i] >= stop)
                if hit_t or hit_s or i == len(c)-2:
                    px = c[i]
                    if fill_at_level:
                        px = tgt if hit_t else (stop if hit_s else c[i])
                    r = side*(px/entry - 1)*1e4 - COST_BP
                    rows.append(dict(ts=dt, r=r, win=r > 0))
                    entry = None
    return pd.DataFrame(rows)


def tstat(x):
    x = np.asarray(x, float)
    return x.mean()/(x.std(ddof=1)/np.sqrt(len(x))) if len(x) > 2 else np.nan


if __name__ == "__main__":
    D = {}
    for nm, slug in MK.items():
        d = load(slug)
        if d is not None:
            D[nm] = (d, sessions(d))

    print("=" * 78)
    print("1. PER-MARKET, NOT POOLED  (pooling correlated markets inflates t)")
    print("=" * 78)
    print(f"  {'market':<12}{'trades':>8}{'win%':>8}{'mean bp':>10}{'t':>8}")
    print("  " + "-" * 48)
    per = {}
    for nm, (d, S) in D.items():
        t = run_fade(d, S)
        per[nm] = t
        print(f"  {nm:<12}{len(t):>8}{t.win.mean()*100:>7.1f}%"
              f"{t.r.mean():>10.2f}{tstat(t.r):>8.2f}", flush=True)
    ts = [tstat(t.r) for t in per.values()]
    print(f"\n  per-market t range: {min(ts):.2f} to {max(ts):.2f}")
    daily = pd.DataFrame({k: v.groupby("ts").r.mean() for k, v in per.items()})
    C = daily.corr()
    off = C.values[np.triu_indices_from(C.values, 1)]
    print(f"  cross-market correlation: {np.abs(off).mean():.3f}")
    neff = len(D)/(1 + (len(D)-1)*np.abs(off).mean())
    print(f"  effective independent markets: {neff:.2f} of {len(D)}")

    print("\n" + "=" * 78)
    print("2. TRADE ECONOMICS")
    print("=" * 78)
    allt = pd.concat(per.values())
    w, l = allt[allt.win], allt[~allt.win]
    print(f"  total trades      {len(allt):,}")
    print(f"  win rate          {allt.win.mean()*100:.1f}%")
    print(f"  avg win           {w.r.mean():+.2f} bp")
    print(f"  avg loss          {l.r.mean():+.2f} bp")
    print(f"  payoff ratio      {abs(w.r.mean()/l.r.mean()):.2f} : 1")
    print(f"  expectancy        {allt.r.mean():+.2f} bp")
    be = abs(l.r.mean())/(abs(l.r.mean())+w.r.mean())
    print(f"  break-even win%   {be*100:.1f}%   (actual {allt.win.mean()*100:.1f}%)")

    print("\n" + "=" * 78)
    print("3. REALISTIC FILLS -- exit AT the level, not at the bar close")
    print("=" * 78)
    print(f"  {'market':<12}{'close-fill bp':>16}{'level-fill bp':>16}{'delta':>10}")
    print("  " + "-" * 56)
    for nm, (d, S) in D.items():
        a = run_fade(d, S).r.mean()
        b = run_fade(d, S, fill_at_level=True).r.mean()
        print(f"  {nm:<12}{a:>15.2f}{b:>16.2f}{b-a:>10.2f}", flush=True)

    print("\n" + "=" * 78)
    print("4. PLACEBO -- random levels of the same width")
    print("=" * 78)
    nm0 = list(D)[0]
    d0, S0 = D[nm0]
    real = run_fade(d0, S0).r.mean()
    fake = []
    for s in range(12):
        t = run_fade(d0, S0, jitter=0.25, seed=s)
        if len(t) > 50:
            fake.append(t.r.mean())
    fake = np.array(fake)
    print(f"  {nm0}: real pivots {real:+.2f} bp")
    print(f"  jittered levels: mean {fake.mean():+.2f} bp, "
          f"sd {fake.std():.2f}, max {fake.max():+.2f}")
    print(f"  random levels matching real: "
          f"{(fake >= real).sum()}/{len(fake)}")
