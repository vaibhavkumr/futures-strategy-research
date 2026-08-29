"""CONFLUENCE SYSTEM -- Riley's one instruction that actually measured positive.

His four patterns individually: -0.086 to -0.134R, all significantly negative,
win rates within 0.2pts of fair odds. But requiring TWO to agree moved it to
+0.041R. That is his stated rule -- "don't use this by itself, couple it with
other indications" -- and it is the only thing in three sources that crossed
zero.

The problem was sample: only 234 qualifying setups, because I had coded 4
patterns and one barely fired. This adds the classic reversal set so
confluence has more chances to form.

VALIDATION IS PRE-REGISTERED, because "add patterns until it works" is exactly
how the 40-filter stack produced a beautiful backtest and a dead holdout:
    DEV     = S&P 500 + GOLD, 2022-2024   -> choose the confluence threshold
    HOLDOUT = NASDAQ + DOW, 2025-2026     -> looked at ONCE
Costs: 0.5bp round trip (real index-futures cost).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

COST_BP = 0.5
MK = {"S&P 500": "usa500idxusd", "NASDAQ": "usatechidxusd",
      "DOW": "usa30idxusd", "GOLD": "xauusd", "GBPUSD": "gbpusd"}


def load(slug):
    fs = [f for f in glob.glob(f"download/{slug}-m5-bid-*.csv") if "2026-07-24" in f]
    d = pd.read_csv(max(fs, key=os.path.getsize))
    d.columns = [c.lower() for c in d.columns]
    ts = d["timestamp"]
    idx = (pd.to_datetime(ts, unit="ms", utc=True)
           if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts, utc=True))
    d.index = idx.dt.tz_convert("America/New_York")
    d = d[["open", "high", "low", "close"]].astype(float).sort_index()
    assert d.index[0].year > 2000
    return d


def prep(d):
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    tr = pd.concat([d["high"] - d["low"],
                    (d["high"] - d["close"].shift()).abs(),
                    (d["low"] - d["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().bfill().values
    return o, h, l, c, atr


# ---------------------------------------------------------------- patterns
# Each returns +1 (bullish), -1 (bearish), 0 (nothing). All look BACKWARD only.

def p_engulfing(o, h, l, c, a, i):
    if i < 2:
        return 0
    if c[i] > o[i] and c[i-1] < o[i-1] and c[i] > o[i-1] and o[i] < c[i-1]:
        return 1 if (c[i]-o[i]) > 0.6*a[i] else 0
    if c[i] < o[i] and c[i-1] > o[i-1] and c[i] < o[i-1] and o[i] > c[i-1]:
        return -1 if (o[i]-c[i]) > 0.6*a[i] else 0
    return 0


def p_pinbar(o, h, l, c, a, i):
    rng = max(h[i]-l[i], 1e-9)
    body = abs(c[i]-o[i])
    if body/rng > 0.35 or rng < 0.6*a[i]:
        return 0
    up = h[i] - max(o[i], c[i])
    dn = min(o[i], c[i]) - l[i]
    if dn > 2.0*body and dn/rng > 0.55:
        return 1
    if up > 2.0*body and up/rng > 0.55:
        return -1
    return 0


def p_star(o, h, l, c, a, i):
    """Morning / evening star: big candle, small body, big opposite candle."""
    if i < 3:
        return 0
    b1, b2, b3 = c[i-2]-o[i-2], abs(c[i-1]-o[i-1]), c[i]-o[i]
    if b1 < -0.7*a[i] and b2 < 0.3*a[i] and b3 > 0.7*a[i] and c[i] > (o[i-2]+c[i-2])/2:
        return 1
    if b1 > 0.7*a[i] and b2 < 0.3*a[i] and b3 < -0.7*a[i] and c[i] < (o[i-2]+c[i-2])/2:
        return -1
    return 0


def p_tweezer(o, h, l, c, a, i):
    if i < 2:
        return 0
    tol = 0.10*a[i]
    if abs(l[i]-l[i-1]) < tol and c[i] > o[i] and c[i-1] < o[i-1]:
        return 1
    if abs(h[i]-h[i-1]) < tol and c[i] < o[i] and c[i-1] > o[i-1]:
        return -1
    return 0


def p_three_line_strike(o, h, l, c, a, i):
    for k in (3, 4, 5):
        if i-k < 0:
            continue
        legs = [c[j]-o[j] for j in range(i-k, i)]
        if all(x < 0 for x in legs) and c[i] > o[i] and c[i] > o[i-k] and (c[i]-o[i]) > 1.0*a[i]:
            return 1
        if all(x > 0 for x in legs) and c[i] < o[i] and c[i] < o[i-k] and (o[i]-c[i]) > 1.0*a[i]:
            return -1
    return 0


def p_trap(o, h, l, c, a, i, look=20):
    if i < look+2:
        return 0
    hi, lo = np.max(h[i-look:i]), np.min(l[i-look:i])
    rng = max(h[i]-l[i], 1e-9)
    if h[i] > hi and c[i] < hi and (h[i]-hi) > 0.3*a[i] and (h[i]-c[i])/rng > 0.5:
        return -1
    if l[i] < lo and c[i] > lo and (lo-l[i]) > 0.3*a[i] and (c[i]-l[i])/rng > 0.5:
        return 1
    return 0


def p_head_shoulders(o, h, l, c, a, i, w=4):
    if i < 6*w:
        return 0
    hh = lambda x, y: np.max(h[x:y])
    ll = lambda x, y: np.min(l[x:y])
    h1, h2, h3 = hh(i-6*w, i-4*w), hh(i-4*w, i-2*w), hh(i-2*w, i)
    if h2 > h1 and h2 > h3 and (h2-max(h1, h3)) > 0.4*a[i] and c[i] < min(h1, h3):
        return -1
    l1, l2, l3 = ll(i-6*w, i-4*w), ll(i-4*w, i-2*w), ll(i-2*w, i)
    if l2 < l1 and l2 < l3 and (min(l1, l3)-l2) > 0.4*a[i] and c[i] > max(l1, l3):
        return 1
    return 0


def p_at_extreme(o, h, l, c, a, i, look=48):
    """Location context: at a multi-hour extreme. His 'major zone'."""
    if i < look:
        return 0
    if l[i] <= np.min(l[i-look:i]):
        return 1
    if h[i] >= np.max(h[i-look:i]):
        return -1
    return 0


PATTERNS = {"engulfing": p_engulfing, "pinbar": p_pinbar, "star": p_star,
            "tweezer": p_tweezer, "three_line_strike": p_three_line_strike,
            "trap": p_trap, "head_shoulders": p_head_shoulders,
            "at_extreme": p_at_extreme}


def signals(d, sess=(570, 960)):
    """One row per bar with a confluence score."""
    o, h, l, c, a = prep(d)
    idx = d.index
    m = idx.hour*60 + idx.minute
    n = len(d)
    rows = []
    for i in range(60, n-2):
        if not (sess[0] <= m[i] < sess[1]):
            continue
        s = [f(o, h, l, c, a, i) for f in PATTERNS.values()]
        pos, neg = sum(1 for x in s if x > 0), sum(1 for x in s if x < 0)
        if pos == 0 and neg == 0:
            continue
        if pos and neg:                     # conflicting -> no trade
            continue
        rows.append(dict(i=i, sgn=1 if pos else -1, score=max(pos, neg), ts=idx[i]))
    return rows


def simulate(d, rows, min_score, tmult=2.0):
    o, h, l, c, a = prep(d)
    n = len(d)
    out = []
    busy = -1
    for r in rows:
        i, sgn = r["i"], r["sgn"]
        if i < busy or r["score"] < min_score:
            continue
        e = c[i]
        stop = (min(l[i-1], l[i])-0.1*a[i]) if sgn > 0 else (max(h[i-1], h[i])+0.1*a[i])
        risk = abs(e-stop)
        if risk < 0.25*a[i] or risk > 3*a[i]:
            continue
        tgt = e + sgn*tmult*risk
        R = None
        for k in range(i+1, min(i+80, n)):
            if (l[k] <= stop) if sgn > 0 else (h[k] >= stop):
                R = -1.0 - 0.05*a[i]/risk
                busy = k
                break
            if (h[k] >= tgt) if sgn > 0 else (l[k] <= tgt):
                R = tmult
                busy = k
                break
        if R is None:
            continue
        out.append(dict(R=R-(COST_BP/1e4)*e/risk, ts=r["ts"], score=r["score"]))
    return pd.DataFrame(out)


def st(x, lab):
    x = np.asarray(x, float)
    if len(x) < 25:
        print(f"  {lab:<30} n={len(x)}  (too few)")
        return None
    m, se = x.mean(), x.std(ddof=1)/np.sqrt(len(x))
    print(f"  {lab:<30} n={len(x):<6} win {(x>0).mean()*100:5.1f}%  "
          f"expR {m:+.3f}  t={m/se:+6.2f}")
    return m


if __name__ == "__main__":
    DEV, HOLD = ("S&P 500", "GOLD"), ("NASDAQ", "DOW")
    print(f"{len(PATTERNS)} patterns coded\n")
    D, S = {}, {}
    for nm, slug in MK.items():
        D[nm] = load(slug)
        S[nm] = signals(D[nm])
        print(f"  {nm:<9} {len(S[nm]):>7,} raw signals")

    print("\n" + "="*70)
    print("DEV (S&P + GOLD, 2022-2024) -- choose the confluence threshold here")
    print("="*70)
    for ms in (1, 2, 3, 4):
        pool = []
        for nm in DEV:
            t = simulate(D[nm], S[nm], ms)
            if len(t):
                t = t[pd.to_datetime(t.ts) < "2025-01-01"]
                pool.append(t.R.values)
        if pool:
            st(np.concatenate(pool), f"score >= {ms}")

    print("\n" + "="*70)
    print("HOLDOUT (NASDAQ + DOW, 2025-2026) -- single look, all shown")
    print("="*70)
    for ms in (1, 2, 3, 4):
        pool = []
        for nm in HOLD:
            t = simulate(D[nm], S[nm], ms)
            if len(t):
                t = t[pd.to_datetime(t.ts) >= "2025-01-01"]
                pool.append(t.R.values)
        if pool:
            st(np.concatenate(pool), f"score >= {ms}")
    print("\n  fair odds at a 2R target = 33.3% win rate")
