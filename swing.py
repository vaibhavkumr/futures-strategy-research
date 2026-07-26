"""Swing / position-trading research harness.

Horizon: days to weeks, rebalanced weekly. Free daily data, decades deep,
diversified ETF universe. Reports MONTHLY results, since that's how you
want to measure.

Same discipline as the intraday work:
  - costs modelled (spread + commission per trade)
  - signals use only past data (shift(1) before trading)
  - DEV / HOLDOUT split by date; holdout opened once
  - random-signal control: does the logic beat coin flips?
  - benchmark vs buy & hold, because beating cash is not the bar

Strategies implemented are DOCUMENTED effects, not invented ones:
  tsmom     - time-series momentum / trend following (Moskowitz, Ooi &
              Pedersen 2012). Long when own past return is positive.
  xsmom     - cross-sectional momentum (Jegadeesh & Titman 1993). Hold the
              strongest N of the universe.
  donchian  - classic breakout trend following (Turtle-style).
  meanrev   - short-term reversal; buy weakness, documented but decayed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Diversified, liquid, long-history ETFs across asset classes. Using ETFs
# (not single stocks) avoids most survivorship bias -- these all still exist
# and were selected for asset-class coverage, not past performance.
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM",      # equities
            "TLT", "IEF", "LQD", "HYG",             # bonds
            "GLD", "SLV", "DBC", "USO",             # commodities
            "XLE", "XLF", "XLK", "XLV", "XLU"]      # sectors

COST_BPS = 10.0        # round-trip cost per trade, basis points (spread+comm)
DEV_END = "2018-01-01"  # research on data before this; holdout after


def fetch(tickers=UNIVERSE, start="2004-01-01") -> pd.DataFrame:
    import yfinance as yf
    px = yf.download(tickers, start=start, interval="1d",
                     progress=False, auto_adjust=True)["Close"]
    return px.dropna(how="all").ffill()


# ---------------- signals (all causal: computed on data up to t-1) --------

def sig_tsmom(px: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    """Long when the asset's own trailing return is positive, else flat."""
    return (px.pct_change(lookback) > 0).astype(float)


def sig_xsmom(px: pd.DataFrame, lookback: int = 126, top: int = 5) -> pd.DataFrame:
    """Hold the top-N assets by trailing return, equally weighted."""
    r = px.pct_change(lookback)
    rank = r.rank(axis=1, ascending=False)
    return (rank <= top).astype(float)


def sig_donchian(px: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """Long when price makes a new N-day high; exit on new N/2-day low."""
    hi = px.rolling(lookback).max()
    lo = px.rolling(max(lookback // 2, 5)).min()
    raw = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    raw[px >= hi] = 1.0
    raw[px <= lo] = 0.0
    return raw.ffill().fillna(0.0)


def sig_meanrev(px: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Buy the weakest half over the last week (short-term reversal)."""
    r = px.pct_change(lookback)
    return (r.rank(axis=1, ascending=True) <= px.shape[1] / 2).astype(float)


SIGNALS = {"tsmom": sig_tsmom, "xsmom": sig_xsmom,
           "donchian": sig_donchian, "meanrev": sig_meanrev}


# ---------------- backtest -----------------------------------------------

def backtest(px: pd.DataFrame, weights: pd.DataFrame, rebal: str = "W-FRI",
             cost_bps: float = COST_BPS) -> pd.Series:
    """Weights are target exposures; rebalance on `rebal`, hold in between.
    Signals are shifted one day so we never trade on same-day information."""
    w = weights.shift(1).reindex(px.index).ffill()
    # normalize to full investment across held names (equal weight)
    tot = w.sum(axis=1).replace(0, np.nan)
    w = w.div(tot, axis=0).fillna(0.0)
    # only change positions on rebalance dates
    mask = pd.Series(False, index=px.index)
    mask[px.resample(rebal).last().index.intersection(px.index)] = True
    w = w.where(mask).ffill().fillna(0.0)
    rets = px.pct_change().fillna(0.0)
    gross = (w * rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turnover * cost_bps / 1e4


def stats(r: pd.Series, label: str = "") -> dict:
    if r.std() == 0 or len(r) < 60:
        return {}
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252)
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    m = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    return {"label": label, "CAGR": cagr, "Sharpe": sharpe, "maxDD": dd,
            "pos_months": (m > 0).mean(), "worst_month": m.min(),
            "best_month": m.max(), "months": len(m), "final": eq.iloc[-1]}


def show(rows: list[dict], title: str):
    print(f"\n--- {title} ---")
    d = pd.DataFrame([r for r in rows if r])
    if d.empty:
        print("  (nothing)")
        return
    d = d.sort_values("Sharpe", ascending=False)
    for c, f in [("CAGR", "{:+.1%}"), ("maxDD", "{:.1%}"),
                 ("pos_months", "{:.0%}"), ("worst_month", "{:+.1%}"),
                 ("best_month", "{:+.1%}"), ("Sharpe", "{:.2f}")]:
        d[c] = d[c].map(f.format)
    print(d[["label", "CAGR", "Sharpe", "maxDD", "pos_months",
             "worst_month", "best_month", "months"]].to_string(index=False))
