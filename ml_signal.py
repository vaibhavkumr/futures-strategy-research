"""The 'AI / pattern recognition' component — done the legitimate way.

The wrong way (how most people do it, and why their bots fail live):
  feed raw prices to a neural net, let it find patterns, marvel at the
  in-sample accuracy. It memorizes noise. It always looks brilliant.

The right way, implemented here:
  - features are the ECONOMICALLY MEANINGFUL signals we already built,
    not raw prices
  - target is cross-sectional forward return rank (relative, not absolute)
  - WALK-FORWARD: train only on the past, predict the future, retrain
    periodically. The model never sees data from after its prediction.
  - graded on the same gauntlet as every other signal

If the model can't beat plain 12-month momentum out-of-sample, it doesn't
ship. Complexity has to earn its place too.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import alpha_signals as A
import screen as S

FEATURES = ["mom_12m", "mom_6m", "mom_3m", "mom_12m_skip1m", "trend_ma",
            "rev_1w", "rev_1m", "dd_recovery", "lowvol", "vol_breakout",
            "obv_trend", "dispersion"]
FWD = 21          # predict 21-day forward relative return
RETRAIN_EVERY = 252


def build_panel(px: pd.DataFrame, vol: pd.DataFrame):
    """Long-format panel: one row per (date, asset) with features + target."""
    feats = {}
    for f in FEATURES:
        feats[f] = A.REGISTRY[f](px, vol, px)
    fwd = px.pct_change(FWD).shift(-FWD)
    fwd_rank = fwd.rank(axis=1, pct=True) - 0.5      # cross-sectional target
    frames = []
    for asset in px.columns:
        d = pd.DataFrame({f: feats[f][asset] for f in FEATURES})
        d["y"] = fwd_rank[asset]
        d["asset"] = asset
        d["date"] = px.index
        frames.append(d)
    panel = pd.concat(frames, ignore_index=True)
    return panel.dropna(subset=FEATURES, how="all")


def walk_forward_predict(panel: pd.DataFrame, px: pd.DataFrame,
                         start_date: str, retrain=RETRAIN_EVERY):
    """Train on everything strictly before each retrain point; predict forward.
    Returns a score DataFrame shaped like px."""
    dates = px.index[px.index >= start_date]
    out = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    anchors = list(range(0, len(dates), retrain))
    for k, a in enumerate(anchors):
        t0 = dates[a]
        t1 = dates[anchors[k + 1]] if k + 1 < len(anchors) else dates[-1]
        # training data: everything that was fully observable before t0.
        # subtract FWD days so the target of the last training row does not
        # peek past t0 -- this is the step people forget.
        cutoff = t0 - pd.Timedelta(days=int(FWD * 1.6))
        tr = panel[(panel.date < cutoff)].dropna(subset=["y"])
        tr = tr.dropna(subset=FEATURES)
        if len(tr) < 2000:
            continue
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                          learning_rate=0.05,
                                          min_samples_leaf=200,
                                          random_state=0)
        m.fit(tr[FEATURES].values, tr["y"].values)
        te = panel[(panel.date >= t0) & (panel.date <= t1)].dropna(subset=FEATURES)
        if te.empty:
            continue
        pred = m.predict(te[FEATURES].values)
        for (dt, asset), p in zip(zip(te.date, te.asset), pred):
            out.at[dt, asset] = p
    return out.ffill()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    px, vol = S.fetch()
    panel = build_panel(px, vol)
    print(f"panel: {len(panel):,} rows, {len(FEATURES)} features")

    scores = walk_forward_predict(panel, px, start_date="2010-01-01")
    dev_mask = scores.index < S.DEV_END
    r_dev = S.run(px[dev_mask], scores[dev_mask])
    r_hold = S.run(px[~dev_mask], scores[~dev_mask])

    mom_dev = S.run(px[dev_mask], A.mom_12m(px[dev_mask]))
    mom_hold = S.run(px[~dev_mask], A.mom_12m(px[~dev_mask]))

    print("\n              DEV Sharpe   HOLDOUT Sharpe")
    print(f"  ML model     {S.sharpe(r_dev):>8.2f}   {S.sharpe(r_hold):>12.2f}")
    print(f"  mom_12m      {S.sharpe(mom_dev):>8.2f}   {S.sharpe(mom_hold):>12.2f}")
    ctrl = S.random_control(px[~dev_mask], scores[~dev_mask], n=20)
    print(f"\n  ML vs 20 random controls on holdout: beats "
          f"{(S.sharpe(r_hold) > ctrl).mean()*100:.0f}%")
    verdict = "SHIPS" if S.sharpe(r_hold) > S.sharpe(mom_hold) else "DOES NOT SHIP"
    print(f"\n  verdict: ML {verdict} (must beat plain momentum out-of-sample)")
