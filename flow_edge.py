"""ORDER FLOW -- the one input that is not derivable from OHLC.

Everything disproven so far was some version of "predict direction from price
history": hand-built features, 4,000 learned boosting rules, a CNN+GRU that
read raw bars and matched its own noise-trained twin, and a failed replication
of a published JFE effect. Three independent routes, all zero.

Order flow is genuinely different information. A candle tells you price went
up. It does NOT tell you whether buyers lifted offers aggressively or sellers
simply stepped away -- and that distinction is what every ICT concept is
actually a claim ABOUT. "Liquidity sweep", "order block", "smart money": all
of them are assertions about who transacted against whom. Until now I could
only ever test the shadow those claims cast on bars.

Binance publishes it free, forever, at data.binance.vision:
    taker_buy_base_volume  -> volume that AGGRESSED into the offer
    volume - taker_buy     -> volume that AGGRESSED into the bid
    number_of_trades       -> participation
    volume / trades        -> average size, i.e. size of participant

The decisive test is not "does a flow model make money". It is whether the
SAME model, on the SAME bars, does better WITH flow than WITHOUT it. That
isolates the contribution of the new information and cannot be gamed by
tuning, because both arms are tuned identically.
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
DEV_SYMS = ("BTCUSDT", "ETHUSDT")
HOLD_SYMS = ("SOLUSDT", "XRPUSDT")

RAW_COLS = ["ot", "o", "h", "l", "c", "v", "ct", "qv", "ntrades",
            "tbbav", "tbqav", "ig"]


def load_symbol(sym: str) -> pd.DataFrame:
    """Handles both the preprocessed cache and the raw 12-column archive."""
    parts = []
    for f in sorted(glob.glob(f"binance/{sym}-5m-*.csv")):
        head = open(f).readline()
        if head.startswith("ot,"):
            d = pd.read_csv(f)
        else:
            d = pd.read_csv(f, header=None, names=RAW_COLS)
        d = d[["ot", "o", "h", "l", "c", "v", "ntrades", "tbbav"]].copy()
        # BUG: Binance switched open_time from MILLIseconds to MICROseconds
        # partway through 2025. Detecting the unit once for the concatenated
        # series parsed every older file as us -> 1970-01-20, which silently
        # destroyed the time ordering and therefore the walk-forward split.
        # Detect PER FILE and normalise to ms.
        mx = d["ot"].max()
        if mx > 1e15:
            d["ot"] = d["ot"] // 1000          # us -> ms
        elif mx < 1e11:
            d["ot"] = d["ot"] * 1000           # s  -> ms
        parts.append(d)
    if not parts:
        raise FileNotFoundError(sym)
    d = pd.concat(parts, ignore_index=True)
    d = d.drop_duplicates("ot").sort_values("ot")
    d.index = pd.to_datetime(d["ot"], unit="ms", utc=True)
    d = d.drop(columns=["ot"]).astype(float)
    d = d[~d.index.duplicated()]
    if d.index.min().year < 2015:
        raise ValueError(f"{sym}: timestamps parsed to {d.index.min()} -- bad units")
    return d


def flow_features(d: pd.DataFrame) -> pd.DataFrame:
    """ONLY the columns OHLC cannot give you. Kept separate from price
    features so the with/without comparison is clean."""
    f = pd.DataFrame(index=d.index)
    v = d["v"].replace(0, np.nan)
    buy = d["tbbav"]
    sell = d["v"] - buy

    # aggressor imbalance: who is crossing the spread
    delta = buy - sell
    f["imb"] = (delta / v).clip(-1, 1)
    for k in (3, 12, 48):
        f[f"imb{k}"] = (delta.rolling(k).sum() / d["v"].rolling(k).sum()).clip(-1, 1)
    # cumulative delta relative to its own recent range
    cd = delta.cumsum()
    f["cd_z"] = (cd - cd.rolling(96).mean()) / cd.rolling(96).std()

    # participation and participant SIZE
    f["ntr_z"] = (d["ntrades"] - d["ntrades"].rolling(96).mean()) / \
        d["ntrades"].rolling(96).std()
    avg = v / d["ntrades"].replace(0, np.nan)
    f["avgsz_z"] = (avg - avg.rolling(96).mean()) / avg.rolling(96).std()
    f["vol_z"] = (v - v.rolling(96).mean()) / v.rolling(96).std()

    # DIVERGENCE: price makes a new extreme, flow does not confirm it.
    # This is the SMT / "smart money" claim stated in the only terms that
    # can actually be measured.
    ret = d["c"].pct_change()
    f["div"] = np.sign(ret.rolling(12).sum()) * -np.sign(delta.rolling(12).sum())
    f["flow_vs_ret"] = (delta.rolling(12).sum() / d["v"].rolling(12).sum()) - \
        np.sign(ret.rolling(12).sum()) * 0.5
    return f


def price_features(d: pd.DataFrame) -> pd.DataFrame:
    """The usual OHLC set, so the with/without-flow arms differ ONLY by flow."""
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    f = pd.DataFrame(index=d.index)
    for k in (1, 3, 6, 12, 24, 48):
        f[f"ret{k}"] = (c - c.shift(k)) / atr
    f["atr_ratio"] = atr / atr.rolling(96).mean()
    f["rng"] = (h - l) / atr
    f["body"] = (c - o) / atr
    f["clpos"] = (c - l) / (h - l).replace(0, np.nan)
    ma = c.rolling(48).mean()
    f["dist_ma"] = (c - ma) / atr
    f["z48"] = (c - ma) / c.rolling(48).std()
    f["hi_dist"] = (c - h.rolling(48).max()) / atr
    f["lo_dist"] = (c - l.rolling(48).min()) / atr
    return f


def build(sym: str, horizon: int = 12) -> pd.DataFrame:
    d = load_symbol(sym)
    P, F = price_features(d), flow_features(d)
    c = d["c"]
    tr = pd.concat([d["h"] - d["l"], (d["h"] - c.shift()).abs(),
                    (d["l"] - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    X = pd.concat([P.add_prefix("p_"), F.add_prefix("f_")], axis=1)
    X["y"] = (c.shift(-horizon) - c) / atr
    X["close"] = c
    X["sym"] = sym
    return X.replace([np.inf, -np.inf], np.nan).dropna()


HORIZON, PURGE = 12, 36


def walk(X, cols, folds=6, shuffle=False, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = X.sort_index()
    n = len(X)
    bounds = np.linspace(n * 0.35, n, folds + 1).astype(int)
    rng = np.random.default_rng(seed)
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        tr = slice(0, max(a - PURGE, 1))
        ytr = X["y"].values[tr]
        if len(ytr) < 3000:
            continue
        if shuffle:
            ytr = rng.permutation(ytr)
        m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                          max_iter=250, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=seed)
        m.fit(X[cols].values[tr], ytr)
        p = m.predict(X[cols].values[a:b])
        out.append(pd.DataFrame({"pred": p, "y": X["y"].values[a:b]},
                                index=X.index[a:b]))
    return pd.concat(out) if out else pd.DataFrame()


def ic(df):
    if len(df) < 50:
        return np.nan, np.nan
    r = np.corrcoef(df["pred"], df["y"])[0, 1]
    return r, r * np.sqrt(len(df) - 2) / np.sqrt(max(1 - r ** 2, 1e-12))


if __name__ == "__main__":
    D = {}
    for s in SYMS:
        D[s] = build(s)
        print(f"  {s:<9} {len(D[s]):>8,} bars  "
              f"{D[s].index.min():%Y-%m-%d} -> {D[s].index.max():%Y-%m-%d}")
    cols = [c for c in D[SYMS[0]].columns if c not in ("y", "close", "sym")]
    PCOLS = [c for c in cols if c.startswith("p_")]
    FCOLS = [c for c in cols if c.startswith("f_")]
    print(f"\n  {len(PCOLS)} price features, {len(FCOLS)} FLOW features")

    print("\n" + "=" * 78)
    print("THE DECISIVE TEST: same model, same bars, WITH vs WITHOUT flow")
    print("=" * 78)
    print(f"{'symbol':<10}{'':<8}{'price only':>16}{'price + FLOW':>18}{'flow adds':>12}")
    print("-" * 78)
    for s in SYMS:
        tag = "DEV" if s in DEV_SYMS else "HOLD"
        r1, t1 = ic(walk(D[s], PCOLS))
        r2, t2 = ic(walk(D[s], cols))
        print(f"{s:<10}[{tag:<4}]  IC {r1:+.4f} t={t1:+5.2f}   "
              f"IC {r2:+.4f} t={t2:+5.2f}   {r2-r1:+.4f}")

    print("\n  FLOW FEATURES ALONE (no price at all):")
    for s in SYMS:
        r, t = ic(walk(D[s], FCOLS))
        tag = "DEV" if s in DEV_SYMS else "HOLD"
        print(f"    {s:<9}[{tag:<4}] IC {r:+.4f}  t={t:+6.2f}")

    print("\n  CONTROL -- shuffled labels, all features (must be ~0):")
    for s in DEV_SYMS:
        r, t = ic(walk(D[s], cols, shuffle=True, seed=1))
        print(f"    {s:<9}[shuf] IC {r:+.4f}  t={t:+6.2f}")
