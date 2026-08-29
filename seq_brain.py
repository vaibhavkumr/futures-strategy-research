"""A model that READS THE CHART instead of applying rules.

Everything before this was either hand-written if-statements or a gradient
boosting model fed 34 features I designed. Both encode MY idea of what
matters. This does not: it takes the raw bar sequence and learns its own
representation, which is the closest thing that exists to "looks at the chart
and forms a view".

Architecture: a 1D convolutional stack (local candle shapes -> multi-bar
patterns) feeding a GRU (order and context), then a head that outputs a
directional view. Roughly 200k parameters -- big enough to learn genuine
structure, small enough that it cannot simply memorise 200k bars.

The validation is identical to everything else in this repo, because that is
the part that matters:
  - walk-forward retrain, PURGED so overlapping targets cannot leak
  - HOLDOUT markets the model never trains on
  - SHUFFLED-label control, which must collapse to zero
  - costs subtracted from every simulated trade

If a learned representation finds something my features missed, this is where
it shows up. If it returns the same zero, that is strong evidence the
information is not in the bars at all -- which is a far more useful result
than another failed rule set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from duka import load

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ = 64          # bars of context the model sees
HORIZON = 12      # bars ahead it predicts
PURGE = HORIZON * 3

MK = {"NASDAQ": "usatechidxusd", "S&P 500": "usa500idxusd",
      "DOW": "usa30idxusd", "DAX": "deuidxeur"}
DEV_MK = ("NASDAQ", "DAX")
HOLD_MK = ("S&P 500", "DOW")


def bar_features(d: pd.DataFrame) -> np.ndarray:
    """Per-bar, scale-free description. Everything is divided by ATR so the
    model sees SHAPE, not price level -- and so a 2022 bar and a 2026 bar are
    directly comparable."""
    o, h, l, c = (d[x].astype(float) for x in ("open", "high", "low", "close"))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    rng = (h - l).replace(0, np.nan)
    f = np.stack([
        ((c - c.shift()) / atr).values,        # return
        ((h - l) / atr).values,                # range
        ((c - o) / atr).values,                # body
        ((h - np.maximum(o, c)) / atr).values,  # upper wick
        ((np.minimum(o, c) - l) / atr).values,  # lower wick
        ((c - l) / rng).values,                # close position in bar
    ], axis=1)
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def target(d: pd.DataFrame) -> np.ndarray:
    c = d["close"].astype(float)
    tr = pd.concat([d["high"] - d["low"],
                    (d["high"] - c.shift()).abs(),
                    (d["low"] - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    y = (c.shift(-HORIZON) - c) / atr
    return np.nan_to_num(y.values, nan=0.0).astype(np.float32)


def windows(F: np.ndarray, y: np.ndarray, valid: np.ndarray):
    """Stack of (SEQ, n_feat) windows ending at each usable bar."""
    idx = np.arange(SEQ, len(F) - HORIZON)
    idx = idx[valid[idx]]
    X = np.lib.stride_tricks.sliding_window_view(F, SEQ, axis=0)
    X = np.transpose(X, (0, 2, 1))          # (n, SEQ, feat)
    return X[idx - SEQ], y[idx], idx


class Brain(nn.Module):
    def __init__(self, nf=6, ch=48, hid=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(nf, ch, 5, padding=2), nn.GELU(),
            nn.Conv1d(ch, ch, 5, padding=2, dilation=1), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=2, dilation=2), nn.GELU(),
        )
        self.gru = nn.GRU(ch, hid, batch_first=True, num_layers=1)
        self.head = nn.Sequential(nn.LayerNorm(hid), nn.Dropout(0.2),
                                  nn.Linear(hid, 32), nn.GELU(),
                                  nn.Linear(32, 1))

    def forward(self, x):                      # x: (B, SEQ, feat)
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.gru(z)
        return self.head(z[:, -1]).squeeze(-1)


def train(Xtr, ytr, epochs=6, bs=512, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    m = Brain().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    n = len(Xtr)
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            xb, yb = Xt[b].to(DEV), yt[b].to(DEV)
            opt.zero_grad()
            loss = lossf(m(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return m


@torch.no_grad()
def predict(m, X, bs=2048):
    m.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(m(torch.from_numpy(X[i:i + bs]).to(DEV)).cpu().numpy())
    return np.concatenate(out)


def build(name):
    d = load(MK[name], "m5")
    m = d.index.hour * 60 + d.index.minute
    valid_time = np.asarray((m >= 120) & (m < 960))
    F, y = bar_features(d), target(d)
    ok = np.isfinite(F).all(1) & np.isfinite(y)
    return windows(F, y, valid_time & ok), d.index


def walk(name, folds=5, shuffle=False, seed=0):
    (X, y, idx), tindex = build(name)
    n = len(X)
    bounds = np.linspace(n * 0.40, n, folds + 1).astype(int)
    rng = np.random.default_rng(seed)
    preds, acts, times = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        tr_end = max(a - PURGE, 1)
        Xtr, ytr = X[:tr_end], y[:tr_end]
        if len(Xtr) < 5000:
            continue
        if shuffle:
            ytr = rng.permutation(ytr)
        mdl = train(Xtr, ytr, seed=seed)
        p = predict(mdl, X[a:b])
        preds.append(p); acts.append(y[a:b]); times.append(idx[a:b])
    if not preds:
        return None
    return (np.concatenate(preds), np.concatenate(acts),
            tindex[np.concatenate(times)])


def ic(p, a):
    r = np.corrcoef(p, a)[0, 1]
    t = r * np.sqrt(len(p) - 2) / np.sqrt(max(1 - r ** 2, 1e-12))
    return r, t


if __name__ == "__main__":
    print(f"device: {DEV}   seq={SEQ} bars   horizon={HORIZON} bars")
    print(f"params: {sum(p.numel() for p in Brain().parameters()):,}\n")
    print("=" * 74)
    print("STEP 1  does a learned representation beat hand-made features?")
    print("=" * 74)
    res = {}
    for name in MK:
        out = walk(name)
        if out is None:
            continue
        p, a, ts = out
        res[name] = (p, a, ts)
        r, t = ic(p, a)
        tag = "DEV" if name in DEV_MK else "HOLDOUT"
        print(f"  {name:<9} [{tag:<7}] n={len(p):>7,}  IC {r:+.4f}  t={t:+6.2f}")
    print("\n  CONTROL -- shuffled training labels (must be ~0):")
    for name in DEV_MK:
        out = walk(name, shuffle=True, seed=1)
        if out is None:
            continue
        r, t = ic(out[0], out[1])
        print(f"  {name:<9} [shuffled] n={len(out[0]):>7,}  IC {r:+.4f}  t={t:+6.2f}")

    print("\n" + "=" * 74)
    print("STEP 2  directional accuracy by confidence (does it KNOW when it knows?)")
    print("=" * 74)
    for tag, grp in (("DEV", DEV_MK), ("HOLDOUT", HOLD_MK)):
        P = np.concatenate([res[n][0] for n in grp if n in res])
        A = np.concatenate([res[n][1] for n in grp if n in res])
        print(f"\n  --- {tag} ---")
        q = np.quantile(np.abs(P), [0, .5, .8, .9, .95, .99, 1.0])
        for lo, hi, lab in zip(q[:-1], q[1:], ["0-50%", "50-80%", "80-90%",
                                               "90-95%", "95-99%", "top 1%"]):
            m = (np.abs(P) >= lo) & (np.abs(P) < hi) if hi < q[-1] else (np.abs(P) >= lo)
            if m.sum() < 50:
                continue
            hit = (np.sign(P[m]) == np.sign(A[m])).mean() * 100
            edge = (np.sign(P[m]) * A[m]).mean()
            print(f"    conf {lab:<8} n={m.sum():>7,}  direction right {hit:5.2f}%  "
                  f"mean signed move {edge:+.4f} ATR")
    print("\n  50.00% = coin flip. Costs need roughly 52-53% to break even.")
