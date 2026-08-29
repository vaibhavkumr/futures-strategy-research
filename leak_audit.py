"""FEATURE LEAK AUDIT -- run this before trusting any model in this repo.

A feature is only legitimate if its value at bar i is unchanged when every
bar after i is deleted. Recompute the whole feature frame on a truncated
series and compare, column by column. Anything that moves is reading the
future, and will look like alpha until it is traded.

Three separate leaks in this project were this exact shape:
  #8   session high/low readable before the session closed
  #11  overnight hold disguised as a time stop
  OR   opening range broadcast to every bar of the day via transform("max")
"""
import numpy as np
import pandas as pd

from duka import load
import edge500 as E


def audit(slug="usatechidxusd", n_probe=12, seed=3):
    d = load(slug, "m5")
    corr = load("usa500idxusd", "m5")
    full = E.features(d, corr)
    rng = np.random.default_rng(seed)
    # probe well inside the series so warmup windows are satisfied
    spots = rng.choice(np.arange(5000, len(d) - 5), size=n_probe, replace=False)
    bad = {}
    for s in sorted(spots):
        ts = d.index[s]
        cut = E.features(d.loc[:ts], corr.loc[:ts])
        a, b = full.loc[ts], cut.loc[ts]
        for col in full.columns:
            x, y = a[col], b[col]
            if pd.isna(x) and pd.isna(y):
                continue
            if pd.isna(x) != pd.isna(y) or not np.isclose(
                    float(x), float(y), rtol=1e-6, atol=1e-9):
                bad[col] = bad.get(col, 0) + 1
    print(f"probed {n_probe} timestamps on {slug}, {len(full.columns)} features\n")
    if not bad:
        print("  CLEAN -- every feature reproduces with the future deleted")
        return True
    print("  LEAKING FEATURES (value changes when future bars are removed):")
    for k, v in sorted(bad.items(), key=lambda x: -x[1]):
        print(f"    {k:<16} differs at {v}/{n_probe} probes")
    return False


if __name__ == "__main__":
    ok = audit()
    print()
    print("VERDICT:", "safe to model" if ok else "FIX THE FEATURES FIRST")
