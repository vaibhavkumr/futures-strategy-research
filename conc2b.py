"""TOP-3 CONCENTRATION -- verify it, then tune it.

conc1 found top-3 momentum at 32.7%/yr growth, Sharpe 0.94, survivorship-
controlled and stable across dev/holdout. That is now the headline claim of
this whole project, so it gets the same scrutiny that killed ~80 other
candidates before it -- starting with the control that has caught the most
false positives here: a placebo.

  1. PLACEBO. Random 3 names, same engine, same costs, 200 draws. With only
     3 positions the variance is enormous, so a lucky random draw can look
     spectacular. The question is what FRACTION of random triples match the
     momentum triple -- if it is more than ~5%, the signal is not the reason.

  2. ROBUSTNESS to the survivorship control. Sweep how many winners get
     dropped. A real effect degrades gently; an artifact collapses.

  3. CONVICTION WEIGHT k. With 3 names, does tilting toward the strongest
     help or is equal-weight better?

  4. REBALANCE FREQUENCY. Weekly was inherited from the ETF book and never
     tuned. Momentum is documented as a monthly-horizon effect.

  5. STOP WIDTH. At 3 names one position is a third of the book, so the stop
     matters far more than it did at 20.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import stocks as S
import stocks2 as S2
from conviction import growth


def rebal_backtest(px, score, top=3, k=0.5, pstop=0.06, cost=S.COST, freq="W-FRI"):
    """Same engine, but the rebalance frequency is a parameter."""
    wk = px.resample(freq).last()
    sc = score.reindex(wk.index, method="ffill")
    rd = px.pct_change()
    keys = list(wk.index[:-1])
    out, prev = [], {}
    for i, t in enumerate(keys[:-1]):
        a = sc.loc[t].dropna()
        if len(a) < top:
            continue
        pick = a.nlargest(top)
        sd = pick.std(ddof=1)
        z = (pick - pick.mean()) / sd if sd > 0 else pick * 0
        w = np.exp(k * z)
        w = w / w.sum()
        seg = px.loc[(px.index > t) & (px.index <= keys[i + 1])]
        cols = [c for c in w.index if c in seg.columns]
        if seg.empty or not cols:
            continue
        w = w[cols]
        entry = seg[cols].iloc[0]
        live = pd.Series(True, index=cols)
        rets = []
        for d in range(len(seg)):
            r = rd.loc[seg.index[d], cols].fillna(0)
            rets.append(float((w * live.astype(float) * r).sum()))
            hit = live & ((seg[cols].iloc[d] / entry - 1) <= -pstop)
            if hit.any():
                rets[-1] -= float(w[hit].sum()) * cost
                live &= ~hit
        s = pd.Series(rets, index=seg.index)
        turn = sum(abs(w.get(x, 0) - prev.get(x, 0))
                   for x in set(w.index) | set(prev))
        s.iloc[0] -= turn * cost
        prev = w.to_dict()
        out.append(s)
    return pd.concat(out).sort_index() if out else pd.Series(dtype=float)


def main_fast():
    px, vol = S.load(S.UNIV)
    total = (px.iloc[-1] / px.iloc[0] - 1).sort_values(ascending=False)
    clean = px[list(total.index[30:])]
    print(f"clean universe: {clean.shape[1]} stocks\n", flush=True)

    real = rebal_backtest(clean, S2.sc_mom12(clean), top=3)
    g_real = growth(real)
    return px, clean, total, real, g_real


def placebo(px, clean, total, real, g_real):
    print("=" * 80)
    print("1. PLACEBO -- random 3 names, 60 draws")
    print("=" * 80)
    print(f"  real momentum top 3: growth {g_real['growth']:.1f}%/yr, "
          f"Sharpe {g_real['sharpe']:.2f}\n", flush=True)
    rng = np.random.default_rng(17)
    beat_g, beat_s, gs = 0, 0, []
    N = 60
    for i in range(N):
        sc = pd.DataFrame(rng.standard_normal(clean.shape),
                          index=clean.index, columns=clean.columns)
        r = rebal_backtest(clean, sc, top=3)
        g = growth(r)
        if not g:
            continue
        gs.append(g["growth"])
        beat_g += g["growth"] >= g_real["growth"]
        beat_s += g["sharpe"] >= g_real["sharpe"]
    gs = np.array(gs)
    print(f"  random triples matching real GROWTH : {beat_g}/{len(gs)} = "
          f"{beat_g/len(gs)*100:.1f}%")
    print(f"  random triples matching real SHARPE : {beat_s}/{len(gs)} = "
          f"{beat_s/len(gs)*100:.1f}%")
    print(f"  random growth: mean {gs.mean():.1f}%, sd {gs.std():.1f}, "
          f"95th pct {np.percentile(gs,95):.1f}%")
    print(f"  -> under 5% means the momentum signal is doing the work")

    print("\n" + "=" * 80)
    print("2. ROBUSTNESS TO THE SURVIVORSHIP CONTROL")
    print("=" * 80)
    print(f"  {'drop top N winners':<24}{'GROWTH':>10}{'Sharpe':>9}{'maxDD':>9}")
    print("  " + "-" * 52)
    for drop in (0, 15, 30, 50, 75):
        sub = px[list(total.index[drop:])]
        g = growth(rebal_backtest(sub, S2.sc_mom12(sub), top=3))
        if g:
            print(f"  drop {drop:<19}{g['growth']:>9.1f}%{g['sharpe']:>9.2f}"
                  f"{g['dd']:>8.0f}%")

    print("\n" + "=" * 80)
    print("3. CONVICTION WEIGHT k  (0 = equal weight across the 3)")
    print("=" * 80)
    print(f"  {'k':<10}{'GROWTH':>10}{'vol':>8}{'maxDD':>9}{'Sharpe':>9}")
    print("  " + "-" * 48)
    for k in (0.0, 0.25, 0.5, 1.0, 2.0):
        g = growth(rebal_backtest(clean, S2.sc_mom12(clean), top=3, k=k))
        if g:
            print(f"  {k:<10.2f}{g['growth']:>9.1f}%{g['vol']:>7.0f}%"
                  f"{g['dd']:>8.0f}%{g['sharpe']:>9.2f}")

    print("\n" + "=" * 80)
    print("4. REBALANCE FREQUENCY")
    print("=" * 80)
    print(f"  {'frequency':<16}{'GROWTH':>10}{'vol':>8}{'maxDD':>9}{'Sharpe':>9}")
    print("  " + "-" * 54)
    for freq, lab in (("W-FRI", "weekly"), ("2W-FRI", "biweekly"),
                      ("ME", "monthly"), ("QE", "quarterly")):
        g = growth(rebal_backtest(clean, S2.sc_mom12(clean), top=3, freq=freq))
        if g:
            print(f"  {lab:<16}{g['growth']:>9.1f}%{g['vol']:>7.0f}%"
                  f"{g['dd']:>8.0f}%{g['sharpe']:>9.2f}")

    print("\n" + "=" * 80)
    print("5. STOP WIDTH  (one name is a third of the book here)")
    print("=" * 80)
    print(f"  {'stop':<16}{'GROWTH':>10}{'vol':>8}{'maxDD':>9}{'Sharpe':>9}")
    print("  " + "-" * 54)
    for ps, lab in ((0.03, "3%"), (0.06, "6%"), (0.10, "10%"),
                    (0.15, "15%"), (0.99, "none")):
        g = growth(rebal_backtest(clean, S2.sc_mom12(clean), top=3, pstop=ps))
        if g:
            print(f"  {lab:<16}{g['growth']:>9.1f}%{g['vol']:>7.0f}%"
                  f"{g['dd']:>8.0f}%{g['sharpe']:>9.2f}")
