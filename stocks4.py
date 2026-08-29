"""FACTORS BEYOND MOMENTUM -- and whether combining them beats momentum alone.

Best verified so far: plain 12m momentum on 202 large caps, ~21.5%/yr at
Sharpe 0.90 after survivorship control, stable across dev and holdout.
Everything layered on top has failed its controls.

The untested direction is a genuinely DIFFERENT factor rather than another
refinement of momentum. Diversification is the only free lunch available, and
it needs two effects that fail at different times -- momentum plus a
negatively-correlated factor should beat either alone.

All price-based, so no fundamentals and therefore no look-ahead from
restated financials or point-in-time data I do not have:

  52-WEEK HIGH      George & Hwang (2004). Rank on closeness to the 52-week
                    high rather than on trailing return. Documented to DOMINATE
                    conventional momentum -- the anchor matters more than the
                    path taken to it.
  LONG-TERM REVERSAL DeBondt & Thaler (1985). 5-year losers beat 5-year winners.
                    A price-only proxy for value, and importantly it is
                    NEGATIVELY correlated with 12-month momentum.
  IDIOSYNCRATIC VOL Ang, Hodrick, Xing & Zhang (2006). Low idio-vol names
                    outperform. Distinct from plain low-vol, which already
                    failed as a momentum filter.
  SEASONALITY       Heston & Sadka (2008). Stocks that did well in this
                    calendar month historically tend to repeat.

Then the combinations, which is the actual point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import stocks as S
import stocks2 as S2


def sc_52wk(px, look=252):
    """Proximity to the 52-week high -- George & Hwang."""
    return px/px.rolling(look).max() - 1


def sc_lt_reversal(px, look=1260, skip=252):
    """5-year return excluding the last year -- long-term reversal (value)."""
    return -(px.shift(skip).pct_change(look - skip))


def sc_idio_vol(px, look=126):
    """Negative idiosyncratic vol: residual vol after removing the market."""
    r = px.pct_change()
    mkt = r.mean(axis=1)
    cov = r.rolling(look).cov(mkt)
    var = mkt.rolling(look).var()
    beta = cov.div(var, axis=0)
    resid = r.sub(beta.mul(mkt, axis=0))
    return -resid.rolling(look).std()


def sc_seasonal(px):
    """Same-calendar-month historical performance -- Heston & Sadka."""
    r = px.pct_change(21)
    out = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    m = px.index.month
    for mo in range(1, 13):
        mask = m == mo
        hist = r[mask].expanding().mean().shift(1)
        out.loc[mask] = hist
    return out


def combine(px, scores, weights):
    """Rank-combine several score frames into one."""
    tot = None
    for sc, wt in zip(scores, weights):
        rk = sc.rank(axis=1, pct=True)
        tot = rk*wt if tot is None else tot + rk*wt
    return tot


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
    print(f"universe: {px.shape[1]} stocks\n", flush=True)
    bench = px.pct_change().mean(axis=1).dropna()

    # survivorship-controlled universe is the one that decides anything
    total = (px.iloc[-1]/px.iloc[0] - 1).sort_values(ascending=False)
    sub = px[list(total.index[30:])]
    subb = sub.pct_change().mean(axis=1).dropna()

    mom = S2.sc_mom12(px)
    scores = {"mom 12m (baseline)": mom,
              "52-week high": sc_52wk(px),
              "long-term reversal": sc_lt_reversal(px),
              "low idiosyncratic vol": sc_idio_vol(px),
              "monthly seasonality": sc_seasonal(px)}

    print("="*86)
    print("A. EACH FACTOR ALONE  (full universe, top 20, weekly, 20bp, 6% stops)")
    print("="*86)
    print(f"  {'factor':<32}{'CAGR':>9}{'vol':>7}{'maxDD':>7}{'Sharpe':>8}{'vs B&H':>9}")
    print("  "+"-"*72)
    series = {}
    for lab, sc in scores.items():
        r = S2.backtest(px, sc)
        series[lab] = r
        line(lab, r, bench)

    print("\n"+"="*86)
    print("B. CORRELATION -- do any of them diversify momentum?")
    print("="*86)
    base = series["mom 12m (baseline)"]
    print(f"  {'factor':<32}{'corr vs momentum':>20}")
    print("  "+"-"*54)
    for lab, r in series.items():
        if lab.startswith("mom 12m"):
            continue
        j = base.index.intersection(r.index)
        print(f"  {lab:<32}{base[j].corr(r[j]):>+20.3f}")

    print("\n"+"="*86)
    print("C. COMBINATIONS")
    print("="*86)
    print(f"  {'blend':<32}{'CAGR':>9}{'vol':>7}{'maxDD':>7}{'Sharpe':>8}{'vs B&H':>9}")
    print("  "+"-"*72)
    line("mom only", base, bench)
    line("mom + 52wk high",
         S2.backtest(px, combine(px, [mom, sc_52wk(px)], [1, 1])), bench)
    line("mom + LT reversal",
         S2.backtest(px, combine(px, [mom, sc_lt_reversal(px)], [1, 1])), bench)
    line("mom + seasonality",
         S2.backtest(px, combine(px, [mom, sc_seasonal(px)], [1, 1])), bench)
    line("mom + 52wk + LT rev",
         S2.backtest(px, combine(px, [mom, sc_52wk(px), sc_lt_reversal(px)],
                                 [2, 1, 1])), bench)
    # portfolio of separate sleeves, not a blended score
    r52 = series["52-week high"]
    j = base.index.intersection(r52.index)
    line("50/50 SLEEVES: mom + 52wk", 0.5*base[j] + 0.5*r52[j], bench)

    print("\n"+"="*86)
    print("D. SURVIVORSHIP CONTROL -- top candidates, biggest winners removed")
    print("="*86)
    print(f"  {'config (drop top 30)':<32}{'CAGR':>9}{'vol':>7}{'maxDD':>7}"
          f"{'Sharpe':>8}{'vs B&H':>9}")
    print("  "+"-"*72)
    line("mom 12m", S2.backtest(sub, S2.sc_mom12(sub)), subb)
    line("52-week high", S2.backtest(sub, sc_52wk(sub)), subb)
    line("mom + 52wk high",
         S2.backtest(sub, combine(sub, [S2.sc_mom12(sub), sc_52wk(sub)],
                                  [1, 1])), subb)
    line("mom + LT reversal",
         S2.backtest(sub, combine(sub, [S2.sc_mom12(sub), sc_lt_reversal(sub)],
                                  [1, 1])), subb)

    print("\n"+"="*86)
    print("E. DEV / HOLDOUT on the bias-controlled winner")
    print("="*86)
    cands = {
        "mom 12m": S2.backtest(sub, S2.sc_mom12(sub)),
        "52-week high": S2.backtest(sub, sc_52wk(sub)),
        "mom + 52wk": S2.backtest(sub, combine(sub, [S2.sc_mom12(sub),
                                                     sc_52wk(sub)], [1, 1])),
    }
    for lab, r in cands.items():
        a = S.stats(r.loc[:S.DEV])
        b = S.stats(r.loc[S.DEV:])
        ba = S.stats(subb.loc[:S.DEV])
        bb = S.stats(subb.loc[S.DEV:])
        if not (a and b and ba and bb):
            continue
        print(f"  {lab:<20} DEV {a['cagr']-ba['cagr']:>+7.2f}%   "
              f"HOLDOUT {b['cagr']-bb['cagr']:>+7.2f}%   "
              f"{'STABLE' if (a['cagr']-ba['cagr'])*(b['cagr']-bb['cagr'])>0 else 'UNSTABLE'}")
