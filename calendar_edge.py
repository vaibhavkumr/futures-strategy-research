"""CALENDAR EDGE -- the one intraday effect in this project that survived.

Trade the regular session open-to-close on days when the flow is structural
rather than predictive:

  TURN OF MONTH   last trading day + first 3 of the next month.
                  Documented since Ariel (1987) and Lakonishok & Smidt (1988):
                  pension and payroll inflows are mechanical, not a forecast.
                  Genuinely pre-registered -- not discovered by searching here.

  MONDAY          weaker evidence and CONTRARY to the classic weekend effect,
                  which says Monday returns are negative. Strongly positive on
                  2022-2026, so it is very possibly regime-specific. It is kept
                  as a separate, separately-tracked sleeve so forward data can
                  kill it without touching the turn-of-month sleeve.

Measured, net of 2bp round-trip costs, equal-weight across four indices:
    +9.55bp on selected days vs -0.01bp on every other day (t=+3.58, p=0.0004)
    +6.9%/yr unleveraged, Sharpe 0.88, max drawdown -10.2%
    dev 2022-23 and holdout 2024-26 agree; all four markets positive

That drawdown is the problem to solve, not the return. A prop account's
trailing limit is ~5%; 10.2% breaches it before the edge can pay. Everything
here is therefore built to be volatility-targeted and sleeve-diversified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COST_BP = 2.0


def tag_days(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Label each trading day with its sleeve membership. Uses only the
    calendar, so there is nothing to leak -- the schedule is known in advance,
    which is exactly why this effect is tradeable."""
    s = pd.DataFrame(index=idx)
    s["dow"] = idx.dayofweek
    s["monday"] = s.dow == 0
    s["tom"] = False
    grp = pd.Series(idx, index=idx).groupby([idx.year, idx.month])
    for _, g in grp:
        d = pd.DatetimeIndex(g.values)
        s.loc[d[:3], "tom"] = True      # first 3 trading days of the month
        s.loc[d[-1:], "tom"] = True     # last trading day
    return s


def sessions(df5: pd.DataFrame, open_min=570, close_min=960) -> pd.DataFrame:
    """One row per day: the regular-session open-to-close return."""
    m = df5.index.hour * 60 + df5.index.minute
    d = df5[(m >= open_min) & (m < close_min)]
    day = d.index.normalize()
    o = d.groupby(day)["open"].first()
    c = d.groupby(day)["close"].last()
    h = d.groupby(day)["high"].max()
    lo = d.groupby(day)["low"].min()
    out = pd.DataFrame({"open": o, "close": c, "high": h, "low": lo})
    out["r"] = out["close"] / out["open"] - 1
    return out


def signal_for(day: pd.Timestamp, calendar: pd.DataFrame):
    """Which sleeves fire on this date? Known BEFORE the open."""
    if day not in calendar.index:
        return []
    row = calendar.loc[day]
    out = []
    if bool(row["tom"]):
        out.append("TOM")
    if bool(row["monday"]):
        out.append("MON")
    return out


def backtest(df5: pd.DataFrame, sleeves=("TOM", "MON"), cost_bp=COST_BP,
             vol_target=None, vol_win=60):
    """Long the session open, flat at the close, on selected days.

    vol_target scales position by recent realised volatility so a calm month
    and a violent one contribute equally -- this is the drawdown lever, and
    it is what makes the strategy prop-survivable rather than just profitable.
    """
    S = sessions(df5)
    cal = tag_days(S.index)
    fire = pd.Series(False, index=S.index)
    if "TOM" in sleeves:
        fire |= cal["tom"]
    if "MON" in sleeves:
        fire |= cal["monday"]

    r = S["r"] - cost_bp / 1e4
    if vol_target is not None:
        vol = S["r"].rolling(vol_win).std().shift(1)      # past only
        size = (vol_target / vol).clip(0.2, 3.0).fillna(1.0)
    else:
        size = pd.Series(1.0, index=S.index)

    out = pd.DataFrame({"r_raw": S["r"], "net": r * size, "size": size,
                        "fire": fire, "tom": cal["tom"], "mon": cal["monday"]})
    return out[out.fire]


def stats(net: pd.Series, per_year: float | None = None) -> dict:
    x = net.dropna().values
    if len(x) < 20:
        return {}
    n_yr = per_year or (len(x) / ((net.index[-1] - net.index[0]).days / 365.25))
    eq = (1 + x).cumprod()
    dd = eq / np.maximum.accumulate(eq) - 1
    return dict(n=len(x), mean_bp=x.mean() * 1e4,
                t=x.mean() / (x.std(ddof=1) / np.sqrt(len(x))),
                win=(x > 0).mean() * 100,
                ann=x.mean() * n_yr * 100,
                sharpe=x.mean() / x.std(ddof=1) * np.sqrt(n_yr),
                maxdd=dd.min() * 100, total=(eq[-1] - 1) * 100)


def show(s: dict, label: str):
    if not s:
        print(f"  {label:<30} (insufficient data)")
        return
    print(f"  {label:<30} n={s['n']:<5} {s['mean_bp']:+6.2f}bp  t={s['t']:+5.2f}  "
          f"win {s['win']:4.1f}%  ann {s['ann']:+5.1f}%  "
          f"Sharpe {s['sharpe']:+4.2f}  maxDD {s['maxdd']:5.1f}%")
