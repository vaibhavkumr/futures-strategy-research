"""WOULD A BIGGER / BETTER-TRAINED AI FIND IT?

This is answerable, and not by argument. In machine learning the learning
curve is the standard diagnostic:

  * If a model is limited by CAPACITY, train score is low too -- it cannot
    even fit what it has seen. Fix: a bigger model.
  * If limited by DATA, test score climbs steadily as samples increase.
    Fix: more data, more years, more markets.
  * If limited by INFORMATION, train score climbs (it memorises fine) while
    test score sits at zero no matter how much you add. Nothing fixes this,
    because there is nothing there to learn.

So: train identical models on 5k, 10k, 25k, 50k, 100k and 200k samples, and
at three capacities. Report TRAIN and TEST together. Six data sizes times
three capacities is eighteen points on the curve, and the shape answers the
question definitively.

A control is included: the same sweep on data whose labels are shuffled. That
shows what "provably no information" looks like on this exact plot, so the
real curve can be compared against it rather than against intuition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import flow_edge as F

HORIZON = 12
PURGE = 36


def ic(p, y):
    if len(p) < 50 or np.std(p) == 0:
        return 0.0
    return float(np.corrcoef(p, y)[0, 1])


def curve(X, cols, sizes, capacities, shuffle=False, seed=0):
    """Train on the first `n` samples, test on a fixed later block."""
    X = X.sort_index()
    n_all = len(X)
    test_start = int(n_all * 0.75)
    Xte = X[cols].values[test_start:]
    yte = X["y"].values[test_start:]
    rng = np.random.default_rng(seed)
    rows = []
    for cap_name, params in capacities.items():
        for n in sizes:
            end = min(test_start - PURGE, n)
            if end < 1000:
                continue
            Xtr = X[cols].values[:end]
            ytr = X["y"].values[:end]
            if shuffle:
                ytr = rng.permutation(ytr)
            m = HistGradientBoostingRegressor(random_state=0, **params)
            m.fit(Xtr, ytr)
            rows.append(dict(cap=cap_name, n=end,
                             train_ic=ic(m.predict(Xtr), ytr),
                             test_ic=ic(m.predict(Xte), yte)))
    return pd.DataFrame(rows)


CAPS = {
    "small  (depth 3, 100 trees)":
        dict(max_depth=3, max_iter=100, learning_rate=0.05, min_samples_leaf=200),
    "medium (depth 6, 400 trees)":
        dict(max_depth=6, max_iter=400, learning_rate=0.05, min_samples_leaf=50),
    "large  (depth 12, 1500 trees)":
        dict(max_depth=12, max_iter=1500, learning_rate=0.05, min_samples_leaf=10,
             l2_regularization=0.0),
}
SIZES = [5_000, 10_000, 25_000, 50_000, 100_000, 200_000]


def show(df, title):
    print(f"\n{title}")
    print(f"  {'capacity':<30}{'samples':>9}{'TRAIN ic':>11}{'TEST ic':>10}")
    print("  " + "-" * 60)
    for cap, g in df.groupby("cap", sort=False):
        for _, r in g.iterrows():
            print(f"  {cap:<30}{int(r.n):>9,}{r.train_ic:>11.4f}{r.test_ic:>10.4f}")
        print()


if __name__ == "__main__":
    X = F.build("BTCUSDT")
    cols = [c for c in X.columns if c not in ("y", "close", "sym")]
    print(f"BTCUSDT  {len(X):,} samples, {len(cols)} features "
          f"(price + order flow)")
    print("Train on the first n, test on a fixed later 25% block.\n")
    print("=" * 66)

    real = curve(X, cols, SIZES, CAPS)
    show(real, "REAL DATA")

    ctrl = curve(X, cols, SIZES, {k: v for k, v in list(CAPS.items())[1:2]},
                 shuffle=True, seed=1)
    show(ctrl, "CONTROL -- labels shuffled (provably no information)")

    print("=" * 66)
    print("HOW TO READ IT")
    print("=" * 66)
    big = real[real.cap.str.startswith("large")]
    print(f"  large model, 5k samples : train {big.train_ic.iloc[0]:+.4f}  "
          f"test {big.test_ic.iloc[0]:+.4f}")
    print(f"  large model, most data  : train {big.train_ic.iloc[-1]:+.4f}  "
          f"test {big.test_ic.iloc[-1]:+.4f}")
    print()
    print("  TRAIN ic high  -> capacity is NOT the limit; it fits fine.")
    print("  TEST ic flat   -> more data does NOT help.")
    print("  Both together  -> the limit is INFORMATION, and no architecture,")
    print("                    no amount of years, and no 'sense for the")
    print("                    market' recovers a signal that is not there.")
