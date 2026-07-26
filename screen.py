"""The gauntlet. Every candidate signal must survive this or be deleted.

Tests, in order of how easy they are to fake:
  1. DEV performance          - does it make money 2004-2017?
  2. Random control           - does it beat 20 random signals of the same
                                shape? (kills anything that is just market beta)
  3. HOLDOUT                  - does it survive 2018-2026, never searched?
  4. Sign stability           - same direction in both halves of dev?

A signal passes only if it clears ALL of them. Passing 1 alone is worthless:
with 18 candidates, several will look good on dev by pure chance.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import alpha_signals as A

COST_BPS = 10.0
DEV_END = "2018-01-01"
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG",
            "GLD", "SLV", "DBC", "USO", "XLE", "XLF", "XLK", "XLV", "XLU",
            "XLP", "XLY", "XLI", "XLB", "EWJ", "FXI", "VNQ", "BTC-USD"]


def fetch(tickers=UNIVERSE, start="2004-01-01"):
    import yfinance as yf
    d = yf.download(tickers, start=start, interval="1d",
                    progress=False, auto_adjust=True)
    px = d["Close"].dropna(how="all").ffill()
    vol = d["Volume"].reindex_like(px).ffill()
    return px, vol


def to_weights(score: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """Turn signal scores into portfolio weights.

    Two kinds of signal need different handling:
      - CROSS-SECTIONAL (values differ across assets): demean to get a
        market-neutral long/short book.
      - MARKET TIMING (same value for every asset, e.g. seasonality, macro
        regime): demeaning would zero it out entirely. Instead scale the
        equal-weight portfolio by the signal -- long when +1, flat at 0.
    """
    s = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cross_sectional = (s.std(axis=1) > 1e-12).mean() > 0.5
    if cross_sectional:
        s = s.sub(s.mean(axis=1), axis=0)
        denom = s.abs().sum(axis=1).replace(0, np.nan)
        return s.div(denom, axis=0).fillna(0.0) * gross
    # timing signal: uniform exposure to the equal-weight basket
    n = s.shape[1]
    return (s / n).fillna(0.0) * gross


def run(px: pd.DataFrame, score: pd.DataFrame, rebal="W-FRI",
        cost_bps=COST_BPS) -> pd.Series:
    w = to_weights(score).shift(1).reindex(px.index).ffill()
    mask = pd.Series(False, index=px.index)
    mask[px.resample(rebal).last().index.intersection(px.index)] = True
    w = w.where(mask).ffill().fillna(0.0)
    rets = px.pct_change().fillna(0.0)
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    return (w * rets).sum(axis=1) - turn * cost_bps / 1e4


def sharpe(r: pd.Series) -> float:
    if r is None or len(r) < 100 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(252)


def random_control(px, score, n=20, seed=0):
    """Same weight magnitudes, shuffled across assets each rebalance."""
    rng = np.random.default_rng(seed)
    out = []
    vals = score.values
    for i in range(n):
        perm = vals.copy()
        for row in range(perm.shape[0]):
            perm[row] = rng.permutation(perm[row])
        out.append(sharpe(run(px, pd.DataFrame(perm, index=score.index,
                                               columns=score.columns))))
    return np.array([o for o in out if not np.isnan(o)])


def screen(px, vol, verbose=True):
    dev = px[px.index < DEV_END]
    hold = px[px.index >= DEV_END]
    vdev = vol[vol.index < DEV_END]
    vhold = vol[vol.index >= DEV_END]
    h1 = dev[dev.index < "2011-01-01"]
    h2 = dev[dev.index >= "2011-01-01"]
    rows = []
    for name, fn in A.REGISTRY.items():
        try:
            s_dev = fn(dev, vdev, dev)
            r_dev = run(dev, s_dev)
            sh_dev = sharpe(r_dev)
            ctrl = random_control(dev, s_dev, n=20)
            pct = (sh_dev > ctrl).mean() * 100 if len(ctrl) else np.nan
            sh_h1 = sharpe(run(h1, fn(h1, vol.loc[h1.index], h1)))
            sh_h2 = sharpe(run(h2, fn(h2, vol.loc[h2.index], h2)))
            sh_hold = sharpe(run(hold, fn(hold, vhold, hold)))
            stable = (np.sign(sh_h1) == np.sign(sh_h2)) and sh_dev > 0
            passed = (sh_dev > 0.3) and (pct >= 90) and stable and (sh_hold > 0.2)
            rows.append({"signal": name, "dev": sh_dev, "vs_random_pct": pct,
                         "h1": sh_h1, "h2": sh_h2, "HOLDOUT": sh_hold,
                         "stable": stable, "PASS": passed})
        except Exception as e:
            rows.append({"signal": name, "dev": np.nan, "err": str(e)[:40]})
    df = pd.DataFrame(rows).sort_values("HOLDOUT", ascending=False)
    if verbose:
        d = df.copy()
        for c in ["dev", "h1", "h2", "HOLDOUT"]:
            d[c] = d[c].astype(float).round(2)
        d["vs_random_pct"] = d["vs_random_pct"].astype(float).round(0)
        print(d[["signal", "dev", "vs_random_pct", "h1", "h2", "HOLDOUT",
                 "stable", "PASS"]].to_string(index=False))
    return df


if __name__ == "__main__":
    px, vol = fetch()
    print(f"universe {px.shape[1]} assets, {len(px)} days, "
          f"{px.index[0].date()} -> {px.index[-1].date()}")
    print(f"DEV < {DEV_END}   HOLDOUT >= {DEV_END}\n")
    res = screen(px, vol)
    res.to_csv("signal_screen.csv", index=False)
    n = int(res["PASS"].fillna(False).sum())
    print(f"\n{n} of {len(res)} signals passed the full gauntlet.")
