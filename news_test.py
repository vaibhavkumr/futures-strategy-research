"""Does PRE-MARKET NEWS SENTIMENT predict the trading day?

The first input in this project not derived from price. Everything else --
TJR's rules, 40 stacked filters, gradient boosting, a CNN+GRU reading raw
charts, order flow -- was a rearrangement of price history, and all of it
landed on fair odds.

Design, fixed before looking at any result:

  SIGNAL   mean GDELT tone across ECON-themed articles, sampled 13:00 UTC
           (08:00 ET). Complete 90 minutes BEFORE the US cash open, so it is
           tradeable at the open with no lookahead.
  TARGET   that day's regular-session open-to-close return.
  DIRECTION not pre-committed -- news tone could plausibly be momentum
           (good news -> up) or contrarian (good news already priced ->
           fade). So the test is two-sided and the p-value is doubled to
           pay for that.
  BAR      must clear its own shuffled null, on markets not used to pick
           anything, and then exceed 2bp round-trip costs.

Four markets. If tone predicts returns, it should show in more than one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from duka import load

MK = {"NASDAQ": "usatechidxusd", "S&P 500": "usa500idxusd",
      "DOW": "usa30idxusd", "DAX": "deuidxeur"}
COST_BP = 2.0


def daily_returns(slug: str) -> pd.DataFrame:
    d = load(slug, "m5")
    m = d.index.hour * 60 + d.index.minute
    d = d[(m >= 570) & (m < 960)]                 # 09:30-16:00 ET
    day = d.index.normalize().tz_localize(None)
    o = d.groupby(day)["open"].first()
    c = d.groupby(day)["close"].last()
    return pd.DataFrame({"r": (c / o - 1) * 1e4})  # bp


def align(news: pd.DataFrame) -> dict:
    news = news.copy()
    news.index = pd.DatetimeIndex(news.index).tz_localize(None).normalize()
    # signal features, all from the pre-market snapshot only
    news["tone_chg"] = news["tone_econ"].diff()
    news["tone_z"] = ((news["tone_econ"] - news["tone_econ"].rolling(20).mean())
                      / news["tone_econ"].rolling(20).std())
    news["vol_z"] = ((news["n_econ"] - news["n_econ"].rolling(20).mean())
                     / news["n_econ"].rolling(20).std())
    out = {}
    for name, slug in MK.items():
        r = daily_returns(slug)
        j = news.join(r, how="inner").dropna(
            subset=["tone_econ", "tone_chg", "tone_z", "r"])
        out[name] = j
    return out


def test(x, y, label, n_null=2000, seed=0):
    """Correlation vs a shuffled null. Two-sided, because direction was not
    pre-committed."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 60:
        print(f"  {label:<26} n={len(x)} (too few)")
        return
    r = np.corrcoef(x, y)[0, 1]
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rng.permutation(x), y)[0, 1]
                     for _ in range(n_null)])
    p = 2 * min((null >= r).mean(), (null <= r).mean())
    z = (r - null.mean()) / null.std(ddof=1)
    print(f"  {label:<26} n={len(x):<5} corr {r:+.4f}  z={z:+5.2f}  "
          f"p={p:.3f}  {'SIGNAL' if p < 0.05 else 'noise'}")
    return r, p


if __name__ == "__main__":
    news = pd.read_pickle("gkg_news.pkl")
    print(f"news: {len(news)} days  {news.index.min():%Y-%m-%d} -> "
          f"{news.index.max():%Y-%m-%d}")
    print(news[["tone_econ", "n_econ", "econ_share"]].describe().round(3).to_string())
    A = align(news)

    print("\n" + "=" * 72)
    print("DOES PRE-MARKET NEWS TONE PREDICT THE SESSION? (vs shuffled null)")
    print("=" * 72)
    for feat in ("tone_econ", "tone_chg", "tone_z", "vol_z"):
        print(f"\n  --- {feat} ---")
        for name in MK:
            j = A[name]
            test(j[feat], j["r"], name)

    print("\n" + "=" * 72)
    print("TRADE IT: long when tone above median, short below, net of costs")
    print("=" * 72)
    for feat in ("tone_z", "tone_chg"):
        pooled = []
        for name in MK:
            j = A[name].dropna(subset=[feat])
            sig = np.sign(j[feat] - j[feat].median())
            pooled.append(sig * j["r"] - COST_BP)
        R = np.concatenate([p.values for p in pooled])
        m, se = R.mean(), R.std(ddof=1) / np.sqrt(len(R))
        print(f"  {feat:<12} n={len(R):<5} {m:+6.2f}bp  t={m/se:+5.2f}  "
              f"win {(R>0).mean()*100:5.1f}%  ann {m*252/1e4*100:+5.1f}%")
