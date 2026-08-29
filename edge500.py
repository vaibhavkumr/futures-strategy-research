"""EDGE500 -- a forecasting bot, not another pattern bot.

Every previous build asked "does this pattern predict direction?" and the
answer was always no: win rate tracked fair odds to a fraction of a percent
at every R:R. Bolting more filters onto a coin flip cannot help, because a
filter only changes WHICH coin flips you take.

So this asks a different question. Predict the next hour's move directly
from a wide feature set, and only trade when the model is confident. If the
prediction carries information, expectancy beats fair odds. If it does not,
this tells us so instead of hiding it behind a tuned threshold.

Validation is the point of this file:
  - WALK-FORWARD retrain, never fitting a period it then trades
  - PURGED gap between train and test so overlapping targets cannot leak
  - HOLDOUT markets the model never saw
  - SHUFFLED-label control (must collapse to zero)
  - honest costs: slippage + commission subtracted from every trade

Run:  python edge500.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from duka import load

MK = {"NASDAQ": "usatechidxusd", "S&P 500": "usa500idxusd",
      "DOW": "usa30idxusd", "DAX": "deuidxeur"}
DEV_MK = ("NASDAQ", "DAX")          # model is built here
HOLD_MK = ("S&P 500", "DOW")        # and never sees these until the end

HORIZON = 12          # predict 12 five-minute bars ahead (1 hour)
PURGE = HORIZON * 2   # bars dropped between train and test


# ----------------------------------------------------------------- features
def features(df: pd.DataFrame, corr: pd.DataFrame | None = None) -> pd.DataFrame:
    """Everything is backward-looking. Anything computed from bar i uses only
    bars <= i, so a row can be evaluated the moment bar i closes."""
    o, h, l, c = (df[x] for x in ("open", "high", "low", "close"))
    f = pd.DataFrame(index=df.index)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    f["atr"] = atr

    # --- momentum at several horizons, normalised by volatility ------------
    for k in (1, 3, 6, 12, 24, 48, 96):
        f[f"ret{k}"] = (c - c.shift(k)) / atr

    # --- mean reversion / stretch -----------------------------------------
    for k in (12, 48):
        ma = c.rolling(k).mean()
        f[f"dist_ma{k}"] = (c - ma) / atr
        f[f"z{k}"] = (c - ma) / c.rolling(k).std()

    # --- volatility state --------------------------------------------------
    f["atr_pct"] = atr / c * 1e4
    f["atr_ratio"] = atr / atr.rolling(96).mean()
    f["range_pos"] = (c - l.rolling(24).min()) / (
        h.rolling(24).max() - l.rolling(24).min())

    # --- candle / microstructure shape -------------------------------------
    body = (c - o).abs()
    f["body_frac"] = body / (h - l).replace(0, np.nan)
    f["upper_wick"] = (h - np.maximum(o, c)) / atr
    f["lower_wick"] = (np.minimum(o, c) - l) / atr
    f["gap"] = (o - c.shift()) / atr

    # --- session structure (what the day has done so far) ------------------
    day = df.index.normalize()
    f["day_hi_dist"] = (c - h.groupby(day).cummax()) / atr
    f["day_lo_dist"] = (c - l.groupby(day).cummin()) / atr
    f["bars_into_day"] = df.groupby(day).cumcount()
    # opening range: first hour of the NY session
    mins = df.index.hour * 60 + df.index.minute
    # LOOKAHEAD TRAP (same class as the session-range bug that cost 83% of an
    # earlier result): groupby(day).transform("max") broadcasts the WHOLE day's
    # opening range onto every bar of that day -- including 02:00, hours before
    # the range exists. That alone drove IC to +0.50 (t=206), which is not an
    # edge, it is reading the future. Build it with a running max INSIDE the
    # window, then make it readable only once the window has CLOSED.
    inor = (mins >= 570) & (mins < 630)
    orh = h.where(inor).groupby(day).cummax().groupby(day).ffill()
    orl = l.where(inor).groupby(day).cummin().groupby(day).ffill()
    closed = mins >= 630                       # OR is only known after 10:30
    orh = orh.where(closed)
    orl = orl.where(closed)
    f["or_pos"] = (c - orl) / (orh - orl).replace(0, np.nan)
    f["above_or"] = (c > orh).astype(float).where(closed)
    f["below_or"] = (c < orl).astype(float).where(closed)

    # --- prior day ---------------------------------------------------------
    pdh = h.groupby(day).max().shift(1)
    pdl = l.groupby(day).min().shift(1)
    pdc = c.groupby(day).last().shift(1)
    f["pdh_dist"] = (c - pd.Series(day, index=df.index).map(pdh)) / atr
    f["pdl_dist"] = (c - pd.Series(day, index=df.index).map(pdl)) / atr
    f["pdc_dist"] = (c - pd.Series(day, index=df.index).map(pdc)) / atr

    # --- time of day / week ------------------------------------------------
    f["tod"] = mins / 1440.0
    f["tod_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    f["tod_cos"] = np.cos(2 * np.pi * mins / 1440.0)
    f["dow"] = df.index.dayofweek.astype(float)

    # --- relative strength vs the correlated index (the SMT idea, as a
    #     continuous feature rather than a binary trigger) -------------------
    if corr is not None:
        cc = corr["close"].reindex(df.index, method="ffill")
        catr = (corr["high"] - corr["low"]).rolling(14).mean().reindex(
            df.index, method="ffill")
        for k in (6, 24):
            f[f"rs{k}"] = ((c - c.shift(k)) / atr) - ((cc - cc.shift(k)) / catr)

    return f


def target(df: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """Forward move over `horizon` bars, in ATR units. Volatility-normalised
    so one high-vol week cannot dominate the fit."""
    c = df["close"]
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - c.shift()).abs(),
                    (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    return (c.shift(-horizon) - c) / atr


def build(name: str) -> pd.DataFrame:
    d = load(MK[name], "m5")
    partner = "usa500idxusd" if MK[name] != "usa500idxusd" else "usatechidxusd"
    X = features(d, load(partner, "m5"))
    X["y"] = target(d)
    X["close"] = d["close"]
    X["high"] = d["high"]
    X["low"] = d["low"]
    X["mkt"] = name
    # only trade when a real session is running
    mins = d.index.hour * 60 + d.index.minute
    X = X[(mins >= 120) & (mins < 960)]
    return X.replace([np.inf, -np.inf], np.nan).dropna()


# ------------------------------------------------------- walk-forward model
def walk_forward(X: pd.DataFrame, n_folds: int = 8, shuffle_y: bool = False,
                 seed: int = 0):
    """Expanding-window retrain. Each fold trains ONLY on data before its test
    block, with a PURGE gap so the 12-bar-ahead target of the last training row
    cannot overlap the first test row."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    feat = [c for c in X.columns
            if c not in ("y", "close", "high", "low", "mkt")]
    X = X.sort_index()
    idx = np.arange(len(X))
    bounds = np.linspace(len(X) * 0.35, len(X), n_folds + 1).astype(int)
    rng = np.random.default_rng(seed)
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        tr = idx[: max(a - PURGE, 1)]
        te = idx[a:b]
        if len(tr) < 2000 or len(te) < 50:
            continue
        ytr = X["y"].values[tr]
        if shuffle_y:                      # control: destroy the signal only
            ytr = rng.permutation(ytr)
        m = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=250,
            min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
        m.fit(X[feat].values[tr], ytr)
        p = m.predict(X[feat].values[te])
        out.append(pd.DataFrame({"pred": p, "y": X["y"].values[te]},
                                index=X.index[te]))
    return pd.concat(out) if out else pd.DataFrame()


def ic(df: pd.DataFrame) -> tuple[float, float]:
    """Information coefficient + its t-stat. This is the honest first look:
    if predictions carry no information, everything downstream is noise."""
    if len(df) < 30:
        return float("nan"), float("nan")
    r = np.corrcoef(df["pred"], df["y"])[0, 1]
    t = r * np.sqrt(max(len(df) - 2, 1)) / np.sqrt(max(1 - r ** 2, 1e-12))
    return r, t


# -------------------------------------------------------------- trade sim
def simulate(X: pd.DataFrame, pred: pd.Series, thresh: float,
             stop_atr: float = 1.0, tgt_atr: float = 1.5,
             max_hold: int = 24, slip_atr: float = 0.05,
             comm_R: float = 0.028) -> pd.DataFrame:
    """Trade only when |prediction| clears `thresh`. Costs are real: slippage
    on the stop plus MNQ round-trip commission as a fraction of R."""
    X = X.loc[pred.index]
    c, h, l, a = (X[x].values for x in ("close", "high", "low", "atr"))
    ts = X.index
    p = pred.values
    rows = []
    busy = -1
    for i in range(len(X) - 1):
        if i < busy or abs(p[i]) < thresh:
            continue
        sgn = 1 if p[i] > 0 else -1
        entry = c[i]
        risk = stop_atr * a[i]
        if risk <= 0:
            continue
        stop = entry - sgn * risk
        tgt = entry + sgn * tgt_atr * risk
        R = None
        for k in range(i + 1, min(i + 1 + max_hold, len(X))):
            if ts[k].date() != ts[i].date():
                R = (c[k - 1] - entry) * sgn / risk
                busy = k
                break
            hs = (l[k] <= stop) if sgn > 0 else (h[k] >= stop)
            ht = (h[k] >= tgt) if sgn > 0 else (l[k] <= tgt)
            if hs:                       # ambiguous bar resolves against us
                R = -1.0 - slip_atr * a[i] / risk
                busy = k
                break
            if ht:
                R = tgt_atr
                busy = k
                break
        if R is None:
            k = min(i + max_hold, len(X) - 1)
            R = (c[k] - entry) * sgn / risk
            busy = k
        rows.append(dict(ts=ts[i], side="long" if sgn > 0 else "short",
                         pred=p[i], R=R - comm_R))
    return pd.DataFrame(rows)


def stat(R, label):
    R = np.asarray(R, float)
    if len(R) < 5:
        print(f"{label:<34} n={len(R)}")
        return float("nan")
    m, se = R.mean(), R.std(ddof=1) / np.sqrt(len(R))
    print(f"{label:<34} n={len(R):<6} win {(R>0).mean()*100:5.1f}%  "
          f"expR {m:+.3f}  t={m/se:+6.2f}  CI[{m-1.96*se:+.3f},{m+1.96*se:+.3f}]")
    return m


if __name__ == "__main__":
    print("=" * 78)
    print("EDGE500 -- forecast the next hour, trade only when confident")
    print("=" * 78)

    data = {}
    for name in MK:
        data[name] = build(name)
        print(f"  {name:<9} {len(data[name]):>7,} rows  "
              f"{len([c for c in data[name].columns if c not in ('y','close','high','low','mkt')])} features")

    # ---------- STEP 1: does the prediction carry ANY information? ----------
    print("\n" + "-" * 78)
    print("STEP 1  information coefficient (walk-forward, purged, per market)")
    print("-" * 78)
    preds = {}
    for name in MK:
        p = walk_forward(data[name])
        preds[name] = p
        r, t = ic(p)
        tag = "DEV" if name in DEV_MK else "HOLDOUT"
        print(f"  {name:<9} [{tag:<7}] n={len(p):>7,}  IC {r:+.4f}  t={t:+6.2f}")

    print("\n  CONTROL -- same model, SHUFFLED training labels (must be ~0):")
    for name in DEV_MK:
        p = walk_forward(data[name], shuffle_y=True, seed=1)
        r, t = ic(p)
        print(f"  {name:<9} [shuffled] n={len(p):>7,}  IC {r:+.4f}  t={t:+6.2f}")

    # ---------- STEP 2: does that information survive into trades? ---------
    print("\n" + "-" * 78)
    print("STEP 2  trade simulation, costs included (slip + $1.28 RT on MNQ)")
    print("-" * 78)
    for tag, group in (("DEV", DEV_MK), ("HOLDOUT", HOLD_MK)):
        print(f"\n  --- {tag} markets ---")
        for th in (0.0, 0.10, 0.20, 0.30, 0.50):
            allR = []
            for name in group:
                tr = simulate(data[name], preds[name]["pred"], th)
                if len(tr):
                    allR.append(tr.R.values)
            if not allR:
                continue
            R = np.concatenate(allR)
            stat(R, f"  threshold |pred| > {th:.2f}")

    print("\n" + "=" * 78)
    print("Read STEP 1 first. If IC is indistinguishable from the shuffled")
    print("control, nothing in STEP 2 is real no matter how it looks.")
    print("=" * 78)
