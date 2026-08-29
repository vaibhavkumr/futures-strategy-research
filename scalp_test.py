"""SCALPING HORIZONS -- the regime I never tested.

Everything before this ran on 5-minute bars. The trading being described --
stays small, never scales, capacity-constrained -- lives at seconds, not
hours. That is a genuinely different regime and I argued for three days
without checking it.

Two things are true at this timescale and they pull in opposite directions:

  FOR   short-horizon predictability is REAL and well documented. Order flow
        autocorrelates, and price mean-reverts around the microstructure
        noise. There is more signal per bar here than at 5 minutes.
  AGAINST the SPREAD is charged on every round trip and does not shrink with
        your holding period. A 1bp edge over 5 minutes is the same 1bp over
        5 seconds -- but at 5 seconds you pay the spread 60x more often.

So the question is precise: does predictability at N seconds exceed the
round-trip cost of capturing it? That is measurable and this file measures it.

Costs modelled honestly:
  spread   BTC/ETH on Binance ~1bp; you cross half of it entering and half
           exiting, so ~1bp round trip if you take liquidity
  fees     0.10% taker per side = 20bp round trip
           0.075% maker per side with BNB = 15bp
           VIP/rebate tiers can approach 0-4bp; modelled as a best case
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd

COLS = ["ot", "o", "h", "l", "c", "v", "ct", "qv", "ntrades",
        "tbbav", "tbqav", "ig"]


def load_1s(sym: str, max_rows: int | None = None) -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(f"binance1s/{sym}-1s-*.csv")):
        d = pd.read_csv(f, header=None, names=COLS,
                        usecols=["ot", "o", "h", "l", "c", "v", "ntrades", "tbbav"])
        mx = d["ot"].max()
        if mx > 1e15:
            d["ot"] //= 1000
        parts.append(d)
    d = pd.concat(parts, ignore_index=True).drop_duplicates("ot").sort_values("ot")
    d.index = pd.to_datetime(d["ot"], unit="ms", utc=True)
    d = d.drop(columns=["ot"]).astype(float)
    if d.index.min().year < 2015:
        raise ValueError("bad timestamp units")
    if max_rows:
        d = d.iloc[:max_rows]
    return d


def features(d: pd.DataFrame) -> pd.DataFrame:
    """Microstructure features, all strictly backward-looking."""
    c = d["c"]
    f = pd.DataFrame(index=d.index)
    ret = c.pct_change()
    vol = ret.rolling(300).std()
    for k in (1, 3, 10, 30, 60):
        f[f"r{k}"] = (c / c.shift(k) - 1) / vol
    # order flow: aggressor imbalance at second resolution
    buy = d["tbbav"]
    sell = d["v"] - buy
    delta = buy - sell
    for k in (5, 30, 120):
        f[f"imb{k}"] = (delta.rolling(k).sum()
                        / d["v"].rolling(k).sum().replace(0, np.nan)).clip(-1, 1)
    f["ntr_z"] = ((d["ntrades"] - d["ntrades"].rolling(300).mean())
                  / d["ntrades"].rolling(300).std())
    f["vol_z"] = ((d["v"] - d["v"].rolling(300).mean())
                  / d["v"].rolling(300).std())
    f["spread_px"] = (d["h"] - d["l"]) / c * 1e4      # per-second range, bp
    return f


def horizon_scan(d: pd.DataFrame, label: str):
    """For each holding period: how big is the predictable move, in bp?"""
    c = d["c"]
    F = features(d)
    print(f"\n  {label}")
    print(f"    {'hold':<8}{'|autocorr|':>12}{'best feat IC':>14}"
          f"{'implied move':>15}{'vs 1bp spread':>16}")
    print("    " + "-" * 65)
    for h in (1, 5, 15, 60, 300):
        fwd = (c.shift(-h) / c - 1) * 1e4                # bp
        ac = c.pct_change().autocorr(lag=h)
        best_ic, best_name = 0.0, ""
        for col in F.columns:
            x, y = F[col], fwd
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 5000:
                continue
            r = np.corrcoef(x[m], y[m])[0, 1]
            if abs(r) > abs(best_ic):
                best_ic, best_name = r, col
        # implied capturable move = IC * sigma(fwd), the standard approximation
        sd = fwd.std()
        implied = abs(best_ic) * sd
        print(f"    {str(h)+'s':<8}{abs(ac):>12.4f}{best_ic:>+14.4f}"
              f"{implied:>13.2f}bp{implied/1.0:>15.1f}x")
    return F


def trade_sim(d, F, hold, thresh_q, cost_bp):
    """Trade the best single feature. Deliberately simple -- the question is
    whether ANY edge clears costs, not whether a clever model can be fit."""
    c = d["c"].values
    fwd = (np.roll(c, -hold) / c - 1) * 1e4
    best, name = 0.0, None
    for col in F.columns:
        x = F[col].values
        m = np.isfinite(x) & np.isfinite(fwd)
        if m.sum() < 5000:
            continue
        r = np.corrcoef(x[m], fwd[m])[0, 1]
        if abs(r) > abs(best):
            best, name = r, col
    x = F[name].values
    sgn = np.sign(best)
    thr = np.nanquantile(np.abs(x), thresh_q)
    take = np.isfinite(x) & np.isfinite(fwd) & (np.abs(x) >= thr)
    take[-hold:] = False
    # non-overlapping
    idx = np.where(take)[0]
    keep, last = [], -10**9
    for i in idx:
        if i - last >= hold:
            keep.append(i); last = i
    keep = np.array(keep)
    if len(keep) < 200:
        return None
    R = sgn * np.sign(x[keep]) * fwd[keep] - cost_bp
    return name, best, R


if __name__ == "__main__":
    print("=" * 76)
    print("SCALPING TEST -- 1-SECOND BARS, BTC (dev) and ETH (holdout)")
    print("=" * 76)
    D = {}
    for s in ("BTCUSDT", "ETHUSDT"):
        D[s] = load_1s(s)
        print(f"  {s:<9} {len(D[s]):>10,} one-second bars  "
              f"{D[s].index.min():%Y-%m-%d} -> {D[s].index.max():%Y-%m-%d}")

    print("\n" + "=" * 76)
    print("STEP 1  how much predictable movement exists at each horizon?")
    print("=" * 76)
    F = {}
    for s in D:
        F[s] = horizon_scan(D[s], s)

    print("\n" + "=" * 76)
    print("STEP 2  trade it, at each realistic cost level")
    print("=" * 76)
    for hold in (5, 15, 60):
        print(f"\n  --- holding {hold}s, top 10% of signal ---")
        for cost, clab in ((0.0, "0bp  (impossible)"),
                           (1.0, "1bp  (spread only, maker rebate tier)"),
                           (15.0, "15bp (maker, BNB discount)"),
                           (20.0, "20bp (taker, what you'd actually pay)")):
            for s in D:
                out = trade_sim(D[s], F[s], hold, 0.90, cost)
                if out is None:
                    continue
                name, ic_, R = out
                m, se = R.mean(), R.std(ddof=1) / np.sqrt(len(R))
                tag = "DEV " if s == "BTCUSDT" else "HOLD"
                print(f"    {clab:<38}[{tag}] n={len(R):<6} "
                      f"{m:+7.2f}bp  t={m/se:+6.2f}")
            print()
