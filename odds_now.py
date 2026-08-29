"""HONEST ODDS FROM WHERE THE ACCOUNT ACTUALLY IS.

Every probability quoted so far assumed a $10,000 start. The account is at
$8,542.59 after the USO gap, so the target is 2.34x from here rather than
2.00x, and the floor is closer. Those are materially different odds and the
old numbers should not be reused.

Recomputed on the real system: weekly conviction-tilted momentum, 6% per
position stop, transaction costs, bootstrapped in 5-day blocks on 23 years of
daily returns. Horizon is the time REMAINING in the original 1-2 month window
plus longer horizons, since the hole changes what is reachable.
"""
import json

import numpy as np
import pandas as pd

import factor_lab as F

st = json.load(open("moonshot_state.json"))
EQ, LEV, TGT, FLOOR = st["equity"], st["lev"], st["target"], st["stop"]
PSTOP = st["pstop"]

px = F.universe(); wk = px.resample("W-FRI").last()
raw = px.pct_change(252).reindex(wk.index, method="ffill")
rd = px.pct_change(); COST = 10/1e4; K = 0.5
wkeys = [t for t in wk.index[:-1] if raw.loc[t].dropna().shape[0] >= 8]


def stream(pstop=PSTOP):
    out = []; prev = {}
    for i, t in enumerate(wkeys[:-1]):
        a = raw.loc[t].dropna()
        z = (a-a.mean())/a.std(ddof=1); w = np.exp(K*z); w = w/w.sum()
        seg = px.loc[(px.index > t) & (px.index <= wkeys[i+1])]
        if seg.empty: continue
        cols = [c for c in w.index if c in seg.columns]
        if not cols: continue
        w = w[cols]; entry = seg[cols].iloc[0]; live = pd.Series(True, index=cols)
        rets = []
        for d in range(len(seg)):
            r = rd.loc[seg.index[d], cols].fillna(0)
            rets.append(float((w*live.astype(float)*r).sum()))
            hit = live & ((seg[cols].iloc[d]/entry - 1) <= -pstop)
            if hit.any():
                rets[-1] -= float(w[hit].sum())*COST; live &= ~hit
        s = pd.Series(rets, index=seg.index)
        turn = sum(abs(w.get(x, 0)-prev.get(x, 0)) for x in set(w.index) | set(prev))
        s.iloc[0] -= turn*COST
        prev = w.to_dict(); out.append(s)
    return pd.concat(out).sort_index().values


DR = stream()
rng = np.random.default_rng(101)


def sim(eq0, lev, days, n=60000, block=5):
    nb = max(1, (days+block-1)//block)
    idx = rng.integers(0, len(DR)-block, size=(n, nb))
    p = np.concatenate([DR[idx+j] for j in range(block)], axis=1)[:, :days]*lev
    eq = np.full(n, float(eq0)); hit = np.zeros(n, bool); stp = np.zeros(n, bool)
    for d in range(days):
        liv = ~hit & ~stp
        eq[liv] *= (1+np.clip(p[liv, d], -0.999, None))
        hit |= liv & (eq >= TGT); eq[hit & (eq > TGT)] = TGT
        ns = liv & ~hit & (eq <= FLOOR); stp |= ns; eq[ns] = FLOOR
    return hit.mean(), stp.mean(), eq


print("="*80)
print(f"ODDS FROM ${EQ:,.2f} AT {LEV:.0f}x   (target ${TGT:,.0f} = "
      f"{TGT/EQ:.2f}x from here, floor ${FLOOR:,.0f})")
print("="*80)
print(f"  {'horizon':<20}{'P($20k)':>10}{'P(floor)':>10}{'median':>11}"
      f"{'p25':>10}{'p75':>10}")
print("  "+"-"*70)
for days, lab in ((37, "rest of the 2mo"), (63, "3 months"),
                  (126, "6 months"), (252, "1 year")):
    h, s, eq = sim(EQ, LEV, days)
    print(f"  {lab:<20}{h*100:>9.1f}%{s*100:>9.1f}%{np.median(eq):>11,.0f}"
          f"{np.percentile(eq,25):>10,.0f}{np.percentile(eq,75):>10,.0f}")

print("\n"+"="*80)
print("WHAT THE GAP COST IN PROBABILITY  (same 37 days remaining)")
print("="*80)
for eq0, lab in ((10000, "had USO not gapped"), (EQ, "actual, from here")):
    h, s, _ = sim(eq0, LEV, 37)
    print(f"  {lab:<24} start ${eq0:>9,.0f}   P($20k) {h*100:>5.1f}%   "
          f"P(floor) {s*100:>5.1f}%")

print("\n"+"="*80)
print("LEVERAGE OPTIONS FROM HERE  (rest of the 2 months, 37 days)")
print("="*80)
print(f"  {'lev':>5}{'P($20k)':>10}{'P(floor)':>10}{'median':>11}{'p10':>10}")
print("  "+"-"*48)
for lev in (10, 15, 20, 25, 30, 40):
    h, s, eq = sim(EQ, lev, 37)
    print(f"  {lev:>4}x{h*100:>9.1f}%{s*100:>9.1f}%{np.median(eq):>11,.0f}"
          f"{np.percentile(eq,10):>10,.0f}")
