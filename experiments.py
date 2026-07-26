"""Systematic filter search with anti-data-mining discipline.

THE RULE: search only on DEV (2022-2024). HOLDOUT (2025-2026) is opened
exactly once, for the single best config. If it dies there, it was noise.

Every config reports a t-stat. Remember: when you test N configs, the best
one's t-stat is inflated by selection. A t of 2 across 300 configs is
expected by chance -- we account for that explicitly at the end.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import data as datamod
from strategy import add_indicators
import fastcore as fc

DEV_END = "2025-01-01"
CSV = "download/usatechidxusd-m5-bid-2022-01-01-2026-07-24.csv"
_prep_cache: dict = {}


def get_prep(df, key, swing_lb, zones):
    ck = (key, swing_lb, tuple(zones))
    if ck not in _prep_cache:
        _prep_cache[ck] = fc.prep(df, swing_lb, zones)
    return _prep_cache[ck]


def evaluate(df, key, min_n=30, **p):
    zones = p.pop("zones", ("ny_am",))
    P = get_prep(df, key, p.get("swing_lb", 2), zones)
    R = fc.simulate_fast(P, fc.signals_fast(P, **p))
    s = fc.stats(R, min_n)
    s.update(p); s["zones"] = "+".join(zones)
    return s


def grid(df, key, gd, min_n=30):
    keys = list(gd)
    rows = [evaluate(df, key, min_n, **dict(zip(keys, c)))
            for c in itertools.product(*(gd[k] for k in keys))]
    return pd.DataFrame(rows).sort_values("expR", ascending=False, na_position="last")


def show(d, cols, n=10, title=""):
    print(f"\n--- {title} ---")
    d = d.dropna(subset=["expR"]).head(n).copy()
    if d.empty:
        print("  (nothing met the minimum trade count)")
        return
    for c in ["win", "expR", "pf", "ddR", "t", "totR"]:
        d[c] = d[c].astype(float).round(2)
    print(d[cols + ["n", "win", "expR", "pf", "ddR", "t"]].to_string(index=False))


if __name__ == "__main__":
    df = add_indicators(datamod.load_csv(CSV)).between_time("09:30", "16:00")
    dev = df[df.index < DEV_END]
    hold = df[df.index >= DEV_END]
    print(f"DEV     {len(dev):,} bars  {dev.index[0].date()} -> {dev.index[-1].date()}")
    print(f"HOLDOUT {len(hold):,} bars  {hold.index[0].date()} -> {hold.index[-1].date()}  [LOCKED]")

    base = evaluate(dev, "dev", rr=2.0)
    print("\nBASELINE (dev):", {k: round(v, 3) for k, v in base.items()
                               if k in ("n", "win", "expR", "pf", "ddR", "t")})

    all_res = []

    g = {"rr": [1.0, 1.5, 2.0, 2.5, 3.0], "swing_lb": [2, 3, 4, 5],
         "fvg_window": [6, 12, 20, 30]}
    r = grid(dev, "dev", g); all_res.append(r)
    show(r, list(g), title="core params")

    g = {"zones": [("ny_am",), ("london",), ("ny_pm",), ("ny_am", "ny_pm"),
                   ("london", "ny_am"), ("london", "ny_am", "ny_pm")],
         "rr": [1.5, 2.0, 2.5]}
    r = grid(dev, "dev", g); all_res.append(r)
    show(r, ["zones", "rr"], title="killzones")

    g = {"rr": [1.5, 2.0, 2.5], "trend_filter": [False, True],
         "min_gap_atr": [0.0, 0.25, 0.5, 1.0],
         "max_risk_atr": [0.0, 2.0, 3.0, 5.0]}
    r = grid(dev, "dev", g); all_res.append(r)
    show(r, ["rr", "trend_filter", "min_gap_atr", "max_risk_atr"],
         title="trend / gap / risk-cap filters")

    g = {"rr": [1.5, 2.0, 2.5],
         "atr_rank_min": [0.0, 0.2, 0.4, 0.6], "atr_rank_max": [0.6, 0.8, 1.0]}
    r = grid(dev, "dev", g); all_res.append(r)
    show(r, ["rr", "atr_rank_min", "atr_rank_max"], title="volatility regime")

    res = pd.concat(all_res, ignore_index=True)
    res.to_csv("experiment_results.csv", index=False)
    ok = res.dropna(subset=["expR"])
    print(f"\n{len(res)} configs tested, {len(ok)} with >=30 trades.")
    print(f"best dev expectancy: {ok.expR.max():.3f}R   best t: {ok.t.max():.2f}")
    print("HOLDOUT untouched. Run holdout.py to spend the one look.")
