"""IMPROVING THE STOCK MOMENTUM EDGE -- four documented, untested levers.

Baseline established in stocks.py / surv_test.py: ~22%/yr at Sharpe 0.88 once
survivorship bias is stripped out (vs 11.93% at 0.76 for the ETF book). Every
test below is measured against that, and every one carries the drop-winners
control so a bias artifact cannot masquerade as an improvement.

  1. VOL-SCALED MOMENTUM  Barroso & Santa-Clara (2015), "Momentum has its
     moments". Momentum's weakness is rare violent crashes (2009, 2020) when
     beaten-down names rip. Scaling exposure by momentum's OWN trailing
     realised vol sidesteps most of them. Published result: Sharpe roughly
     doubles. This is the single best-documented fix for the strategy's worst
     failure mode.

  2. RESIDUAL MOMENTUM  Blitz, Huij & Martens (2011). Rank on the part of the
     return NOT explained by market beta. Removes the "high beta rallied"
     component, which is what makes plain momentum crash on reversals.

  3. FACTOR STACK  momentum x low-volatility. Two documented effects with
     different mechanisms; the intersection is smaller but historically
     higher quality than either alone.

  4. SHORTER FORMATION  12-month momentum skipping the most recent month is
     the academic standard (the skip avoids 1-month reversal). Never tested
     here -- the live bot uses plain 12m with no skip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import stocks as S

COST = S.COST
K = S.K_TILT


def _weights(score_row, top, k=K):
    a = score_row.dropna()
    if len(a) < top:
        return None
    pick = a.nlargest(top)
    sd = pick.std(ddof=1)
    z = (pick - pick.mean())/sd if sd > 0 else pick*0
    w = np.exp(k*z)
    return w/w.sum()


def backtest(px, score, top=20, pstop=0.06, cost=COST, scale=None):
    """Weekly rebalance on an arbitrary score frame, with per-position stops.

    `scale` is an optional per-week exposure multiplier (used for vol scaling).
    """
    wk = px.resample("W-FRI").last()
    sc = score.reindex(wk.index, method="ffill")
    rd = px.pct_change()
    keys = list(wk.index[:-1])
    out, prev = [], {}
    for i, t in enumerate(keys[:-1]):
        w = _weights(sc.loc[t], top)
        if w is None:
            continue
        seg = px.loc[(px.index > t) & (px.index <= keys[i+1])]
        cols = [c for c in w.index if c in seg.columns]
        if seg.empty or not cols:
            continue
        w = w[cols]
        mult = 1.0 if scale is None else float(scale.get(t, 1.0))
        w = w*mult
        entry = seg[cols].iloc[0]
        live = pd.Series(True, index=cols)
        rets = []
        for d in range(len(seg)):
            r = rd.loc[seg.index[d], cols].fillna(0)
            rets.append(float((w*live.astype(float)*r).sum()))
            hit = live & ((seg[cols].iloc[d]/entry - 1) <= -pstop)
            if hit.any():
                rets[-1] -= float(w[hit].sum())*cost
                live &= ~hit
        s = pd.Series(rets, index=seg.index)
        turn = sum(abs(w.get(x, 0)-prev.get(x, 0)) for x in set(w.index) | set(prev))
        s.iloc[0] -= turn*cost
        prev = w.to_dict()
        out.append(s)
    return pd.concat(out).sort_index() if out else pd.Series(dtype=float)


# ------------------------------------------------------------------ scores
def sc_mom12(px):
    return px.pct_change(252)


def sc_mom12_skip1(px):
    """Academic standard: 12-month return skipping the most recent month."""
    return px.shift(21).pct_change(231)


def sc_residual(px, look=252):
    """Return not explained by the market -- Blitz/Huij/Martens."""
    r = px.pct_change()
    mkt = r.mean(axis=1)
    cov = r.rolling(look).cov(mkt)
    var = mkt.rolling(look).var()
    beta = cov.div(var, axis=0)
    resid = r.sub(beta.mul(mkt, axis=0))
    return resid.rolling(look).sum()


def sc_mom_lowvol(px, look=252, vlook=60):
    """Momentum ranked, then penalised by volatility -- the factor stack."""
    m = px.pct_change(look)
    v = px.pct_change().rolling(vlook).std()
    mz = m.rank(axis=1, pct=True)
    vz = (-v).rank(axis=1, pct=True)
    return mz + vz


def vol_scale(ret, target=0.15, look=126, cap=2.0, floor=0.25):
    """Barroso & Santa-Clara exposure multiplier, weekly."""
    rv = ret.rolling(look).std(ddof=1)*np.sqrt(252)
    lev = (target/rv).clip(lower=floor, upper=cap)
    wk = lev.resample("W-FRI").last().shift(1)
    return wk.dropna()


def line(lab, s, bench):
    st = S.stats(s)
    if not st:
        print(f"  {lab:<32} (insufficient)")
        return None
    b = S.stats(bench.reindex(pd.Series(s).dropna().index).fillna(0))
    ex = st["cagr"]-b["cagr"] if b else np.nan
    print(f"  {lab:<32}{st['cagr']:>8.2f}%{st['vol']:>7.0f}%{st['dd']:>7.0f}%"
          f"{st['sharpe']:>8.2f}{ex:>+9.2f}")
    return st


if __name__ == "__main__":
    px, vol = S.load(S.UNIV)
    print(f"universe: {px.shape[1]} stocks, {px.index.min():%Y-%m} -> "
          f"{px.index.max():%Y-%m}\n", flush=True)
    bench = px.pct_change().mean(axis=1).dropna()

    print("="*84)
    print("A. THE FOUR LEVERS  (top 20, weekly, 20bp costs, 6% stops)")
    print("="*84)
    print(f"  {'variant':<32}{'CAGR':>9}{'vol':>7}{'maxDD':>7}{'Sharpe':>8}{'vs B&H':>9}")
    print("  "+"-"*72)
    base = backtest(px, sc_mom12(px))
    line("baseline: mom 12m", base, bench)
    line("mom 12m, skip 1 month", backtest(px, sc_mom12_skip1(px)), bench)
    line("residual momentum", backtest(px, sc_residual(px)), bench)
    line("momentum x low-vol stack", backtest(px, sc_mom_lowvol(px)), bench)
    vs = vol_scale(base)
    line("VOL-SCALED momentum", backtest(px, sc_mom12(px), scale=vs), bench)

    print("\n"+"="*84)
    print("B. BEST COMBINATION")
    print("="*84)
    print(f"  {'variant':<32}{'CAGR':>9}{'vol':>7}{'maxDD':>7}{'Sharpe':>8}{'vs B&H':>9}")
    print("  "+"-"*72)
    r_res = backtest(px, sc_residual(px))
    line("residual + vol-scaled", backtest(px, sc_residual(px),
                                           scale=vol_scale(r_res)), bench)
    r_st = backtest(px, sc_mom_lowvol(px))
    line("stack + vol-scaled", backtest(px, sc_mom_lowvol(px),
                                        scale=vol_scale(r_st)), bench)

    print("\n"+"="*84)
    print("C. SURVIVORSHIP CONTROL -- same variants, biggest winners removed")
    print("="*84)
    total = (px.iloc[-1]/px.iloc[0] - 1).sort_values(ascending=False)
    sub = px[list(total.index[30:])]
    subb = sub.pct_change().mean(axis=1).dropna()
    print(f"  {'variant (drop top 30)':<32}{'CAGR':>9}{'vol':>7}{'maxDD':>7}"
          f"{'Sharpe':>8}{'vs B&H':>9}")
    print("  "+"-"*72)
    b2 = backtest(sub, sc_mom12(sub))
    line("mom 12m", b2, subb)
    line("residual momentum", backtest(sub, sc_residual(sub)), subb)
    line("momentum x low-vol", backtest(sub, sc_mom_lowvol(sub)), subb)
    line("VOL-SCALED momentum", backtest(sub, sc_mom12(sub),
                                         scale=vol_scale(b2)), subb)

    print("\n"+"="*84)
    print("D. DEV / HOLDOUT on the winner")
    print("="*84)
    cands = {"mom12": backtest(px, sc_mom12(px)),
             "residual": backtest(px, sc_residual(px)),
             "stack": backtest(px, sc_mom_lowvol(px))}
    cands["vol-scaled mom12"] = backtest(px, sc_mom12(px),
                                         scale=vol_scale(cands["mom12"]))
    best = max(cands.items(), key=lambda kv: (S.stats(kv[1]) or {}).get("sharpe", -9))
    print(f"  best by Sharpe: {best[0]}\n")
    for lab, sl in (("DEV  2010-2017", slice(None, S.DEV)),
                    ("HOLDOUT 2018-2026", slice(S.DEV, None))):
        a = S.stats(best[1].loc[sl])
        b = S.stats(bench.loc[sl])
        if a and b:
            print(f"  {lab:<20} {a['cagr']:>7.2f}%  vs B&H {b['cagr']:>7.2f}%"
                  f"   excess {a['cagr']-b['cagr']:>+7.2f}%   Sharpe {a['sharpe']:.2f}")
