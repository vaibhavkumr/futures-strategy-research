"""Niche / 'hidden' strategies — put through the same gauntlet.

Testable with free data:
  1. tax_loss     - buy December's biggest losers, hold through January
  2. micro_mom    - momentum in small names (survivorship-caveated)
  3. pead         - post-earnings announcement drift (needs earnings data)
  4. gap_reversal - overnight gap fades
  5. seasonal_sc  - small-cap January / turn-of-year effect

NOT testable free (stated honestly rather than faked):
  - index add/delete: needs point-in-time index membership (paid)
  - merger arb: needs historical deal announcements + terms (paid)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

COST_BPS = 10.0


def _wrap(w: pd.DataFrame, px: pd.DataFrame, cost=COST_BPS) -> pd.Series:
    w = w.shift(1).reindex(px.index).fillna(0.0)
    rets = px.pct_change(fill_method=None).fillna(0.0)
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    return (w * rets).sum(axis=1) - turn * cost / 1e4


def sharpe(r):
    if r is None or len(r) < 100 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(252)


def strat_tax_loss(px: pd.DataFrame, decile=0.2) -> pd.Series:
    """Buy the year's biggest losers in mid-December, exit end of January."""
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for yr in sorted(set(px.index.year))[:-1]:
        try:
            entry = px.loc[f"{yr}-12-10":f"{yr}-12-20"].index[0]
            exit_ = px.loc[f"{yr+1}-01-25":f"{yr+1}-02-05"].index[0]
        except IndexError:
            continue
        ytd = px.loc[entry] / px.loc[px.index[px.index.year == yr][0]] - 1
        ytd = ytd.dropna()
        if len(ytd) < 10:
            continue
        losers = ytd.nsmallest(max(3, int(len(ytd) * decile))).index
        w.loc[entry:exit_, losers] = 1.0 / len(losers)
    return _wrap(w, px)


def strat_seasonal_sc(px: pd.DataFrame) -> pd.Series:
    """Turn-of-year: long everything equally from Dec 20 to Jan 10."""
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    m, d = px.index.month, px.index.day
    on = ((m == 12) & (d >= 20)) | ((m == 1) & (d <= 10))
    w.loc[on, :] = 1.0 / px.shape[1]
    return _wrap(w, px)


def strat_gap_reversal(px_open: pd.DataFrame, px: pd.DataFrame,
                       thresh=0.02) -> pd.Series:
    """Fade large overnight gaps: buy big gap-downs at the open, exit at close."""
    gap = px_open / px.shift(1) - 1
    sig = (-np.sign(gap) * (gap.abs() > thresh)).fillna(0.0)
    n = sig.abs().sum(axis=1).replace(0, np.nan)
    w = sig.div(n, axis=0).fillna(0.0)
    # intraday: open -> close return
    intr = (px / px_open - 1).fillna(0.0)
    turn = w.abs().sum(axis=1) * 2
    return (w * intr).sum(axis=1) - turn * COST_BPS / 1e4


def strat_micro_mom(px: pd.DataFrame, top=10) -> pd.Series:
    """12-month momentum within a small-cap universe."""
    r = px.pct_change(252, fill_method=None)
    rank = r.rank(axis=1, ascending=False)
    w = (rank <= top).astype(float)
    w = w.div(w.sum(axis=1), axis=0).fillna(0.0)
    mask = pd.Series(False, index=px.index)
    mask[px.resample("W-FRI").last().index.intersection(px.index)] = True
    return _wrap(w.where(mask).ffill().fillna(0.0), px)
