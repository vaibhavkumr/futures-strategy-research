"""Candidate alpha signals — the 'throw everything at it' library.

Every signal is a function: (px, vol, ctx) -> DataFrame of scores in [-1, 1]
aligned to px. Positive = long, negative = short/avoid, 0 = flat.

NOTHING here is trusted until it passes screen.py. A signal in this file is a
HYPOTHESIS, not an edge. Most of these will be deleted after screening --
that is the point of having many.

All signals are causal: they use only information available at the close of
the bar they are computed on, and screening shifts them before trading.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _z(df: pd.DataFrame, win: int = 252) -> pd.DataFrame:
    """Cross-sectionally rank then scale to [-1,1]; robust to outliers."""
    r = df.rank(axis=1, pct=True)
    return (r - 0.5) * 2


def _clip(x):
    return x.clip(-1, 1)


# ---------------- momentum / trend family ---------------------------------

def mom_12m(px, vol=None, ctx=None):
    """Classic 12-month momentum (Jegadeesh & Titman)."""
    return _z(px.pct_change(252))


def mom_6m(px, vol=None, ctx=None):
    return _z(px.pct_change(126))


def mom_3m(px, vol=None, ctx=None):
    return _z(px.pct_change(63))


def mom_12m_skip1m(px, vol=None, ctx=None):
    """12-month momentum skipping the last month — the standard academic
    construction, which avoids short-term reversal contamination."""
    return _z(px.pct_change(252).shift(21))


def tsmom(px, vol=None, ctx=None):
    """Time-series momentum: own trailing return sign (Moskowitz/Ooi/Pedersen)."""
    return _clip(np.sign(px.pct_change(126)))


def trend_ma(px, vol=None, ctx=None):
    """Price vs 200d MA, normalized by volatility."""
    ma = px.rolling(200).mean()
    sd = px.pct_change().rolling(200).std() * np.sqrt(200)
    return _clip((px / ma - 1) / (sd + 1e-9))


def donchian(px, vol=None, ctx=None):
    hi, lo = px.rolling(100).max(), px.rolling(50).min()
    s = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    s[px >= hi] = 1.0
    s[px <= lo] = -1.0
    return s.replace(0.0, np.nan).ffill().fillna(0.0)


# ---------------- reversal / value family ---------------------------------

def rev_1w(px, vol=None, ctx=None):
    """Short-term reversal: buy last week's losers."""
    return -_z(px.pct_change(5))


def rev_1m(px, vol=None, ctx=None):
    return -_z(px.pct_change(21))


def dd_recovery(px, vol=None, ctx=None):
    """Distance below 1-year high — buying deep drawdowns."""
    return _z(px / px.rolling(252).max() - 1)


# ---------------- volatility / risk family --------------------------------

def lowvol(px, vol=None, ctx=None):
    """Low-volatility anomaly: prefer calmer assets."""
    return -_z(px.pct_change().rolling(63).std())


def vol_breakout(px, vol=None, ctx=None):
    """Range expansion after a quiet period."""
    r = px.pct_change()
    short, long = r.rolling(10).std(), r.rolling(100).std()
    return _clip(np.sign(px.pct_change(10)) * _z(short / (long + 1e-9)))


def risk_parity(px, vol=None, ctx=None):
    """Inverse-vol weighting — a sizing signal, always long."""
    iv = 1 / (px.pct_change().rolling(63).std() + 1e-9)
    return iv.div(iv.sum(axis=1), axis=0) * 2 - 0  # scaled later


# ---------------- volume family -------------------------------------------

def vol_surge(px, vol=None, ctx=None):
    """Volume spike with price confirmation."""
    if vol is None:
        return pd.DataFrame(0.0, index=px.index, columns=px.columns)
    rel = vol / (vol.rolling(63).mean() + 1e-9)
    return _clip(np.sign(px.pct_change(5)) * _z(rel))


def obv_trend(px, vol=None, ctx=None):
    """On-balance-volume slope."""
    if vol is None:
        return pd.DataFrame(0.0, index=px.index, columns=px.columns)
    obv = (np.sign(px.diff()) * vol).fillna(0).cumsum()
    return _z(obv.diff(63))


# ---------------- seasonality family --------------------------------------

def turn_of_month(px, vol=None, ctx=None):
    """Long the last/first few trading days of the month."""
    dom = pd.Series(px.index.day, index=px.index)
    on = ((dom >= 28) | (dom <= 3)).astype(float)
    return pd.DataFrame(np.tile(on.values[:, None], (1, px.shape[1])),
                        index=px.index, columns=px.columns)


def day_of_week(px, vol=None, ctx=None):
    """Monday effect (documented historically, widely believed decayed)."""
    dow = pd.Series(px.index.dayofweek, index=px.index)
    on = (dow == 0).astype(float) * 2 - 1
    return pd.DataFrame(np.tile(on.values[:, None], (1, px.shape[1])),
                        index=px.index, columns=px.columns)


# ---------------- cross-asset / macro regime ------------------------------

def risk_on_off(px, vol=None, ctx=None):
    """Credit + bond signal: when HYG beats LQD and stocks beat bonds, risk-on.
    ctx must contain the tickers; falls back to flat if unavailable."""
    if ctx is None:
        return pd.DataFrame(0.0, index=px.index, columns=px.columns)
    need = {"HYG", "LQD", "SPY", "IEF"}
    if not need.issubset(set(ctx.columns)):
        return pd.DataFrame(0.0, index=px.index, columns=px.columns)
    credit = (ctx["HYG"] / ctx["LQD"]).pct_change(63)
    equity = (ctx["SPY"] / ctx["IEF"]).pct_change(63)
    regime = _clip(np.sign(credit) * 0.5 + np.sign(equity) * 0.5)
    return pd.DataFrame(np.tile(regime.values[:, None], (1, px.shape[1])),
                        index=px.index, columns=px.columns)


def dispersion(px, vol=None, ctx=None):
    """Cross-sectional dispersion regime: momentum works better when assets
    move apart, mean reversion when they move together."""
    d = px.pct_change().rolling(21).std().mean(axis=1)
    reg = _clip((d / d.rolling(252).mean() - 1) * 3)
    return pd.DataFrame(np.tile(reg.values[:, None], (1, px.shape[1])),
                        index=px.index, columns=px.columns)


REGISTRY = {
    "mom_12m": mom_12m, "mom_6m": mom_6m, "mom_3m": mom_3m,
    "mom_12m_skip1m": mom_12m_skip1m, "tsmom": tsmom, "trend_ma": trend_ma,
    "donchian": donchian, "rev_1w": rev_1w, "rev_1m": rev_1m,
    "dd_recovery": dd_recovery, "lowvol": lowvol, "vol_breakout": vol_breakout,
    "vol_surge": vol_surge, "obv_trend": obv_trend,
    "turn_of_month": turn_of_month, "day_of_week": day_of_week,
    "risk_on_off": risk_on_off, "dispersion": dispersion,
}
