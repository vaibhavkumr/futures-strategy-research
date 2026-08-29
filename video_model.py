"""THE VIDEO'S MODEL, IMPLEMENTED EXACTLY AS SPECIFIED.

Transcribed from the 28-minute video, the rules are precise enough to code
without interpretation:

  1. 1-minute chart, from the 09:30 ET open
  2. CHANGE OF CHARACTER: a candle CLOSES above the recent swing high
     (bullish) or below the recent swing low (bearish)
  3. FAIR VALUE GAP: three candles where candle 1's wick does not overlap
     candle 3's wick   ->  bullish: high[i-2] < low[i]
  4. ENTRY: limit at the 50% midpoint of that gap. A limit only fills if
     price actually trades there -- never assumed filled.
  5. STOP: just outside the FVG-producing (middle) candle, plus a buffer
  6. BREAK-EVEN: when a candle CLOSES past the prior swing high, move the
     stop to entry. He is explicit that a WICK past the high is not enough,
     and that using wicks would scratch winners early.
  7. TARGET: fixed 1:4

Two of these I never tested before: the close-confirmed break-even rule, and
the 1:4 target (my earlier replay used 2R). Both change the payoff geometry,
so the earlier -0.100R result does not settle this.

His arithmetic is right: at 1:4 the break-even win rate is 1/(1+4) = 20%.
At 30% it is +0.5R/trade, at 40% +1.0R/trade. So the ONLY question that
matters is the realised win rate.

CONSERVATIVE CHOICES, so a bug cannot flatter the result:
  - within a bar, the STOP is checked before the target
  - entry requires price to trade through the midpoint
  - costs charged on every trade
  - break-even exits book exactly 0 minus costs
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

COST_BP = 0.5
OPEN_MIN, CLOSE_MIN = 9*60+30, 16*60
MK = {"S&P 500": "usa500idxusd", "NASDAQ": "usatechidxusd",
      "DOW": "usa30idxusd", "DAX": "deuidxeur"}


def load_m1(slug):
    fs = glob.glob(f"download/{slug}-m1-bid-*.csv")
    if not fs:
        return None
    d = pd.read_csv(max(fs, key=os.path.getsize))
    d.columns = [c.lower() for c in d.columns]
    ts = d["timestamp"]
    idx = (pd.to_datetime(ts, unit="ms", utc=True)
           if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts, utc=True))
    d.index = idx.dt.tz_convert("America/New_York")
    d = d[["open", "high", "low", "close"]].astype(float).sort_index()
    assert d.index[0].year > 2000, "timestamp parse failed"
    return d


def swings(h, l, k=3):
    """Fractal swing highs/lows: extreme of a 2k+1 window, centred."""
    n = len(h)
    sh = np.zeros(n, bool)
    sl = np.zeros(n, bool)
    for i in range(k, n-k):
        if h[i] == max(h[i-k:i+k+1]):
            sh[i] = True
        if l[i] == min(l[i-k:i+k+1]):
            sl[i] = True
    return sh, sl


def run_day(o, h, l, c, tmult=4.0, buf=0.10, be_on_close=True):
    """One session. Returns a list of trade R-multiples."""
    n = len(c)
    if n < 30:
        return []
    sh, sl = swings(h, l)
    out = []
    i = 8
    while i < n - 2:
        # --- 1. change of character: close beyond the last confirmed swing
        last_sh = last_sl = None
        for j in range(i-1, max(i-60, 2), -1):
            if last_sh is None and sh[j] and j + 3 <= i:
                last_sh = h[j]
            if last_sl is None and sl[j] and j + 3 <= i:
                last_sl = l[j]
            if last_sh is not None and last_sl is not None:
                break
        side = 0
        if last_sh is not None and c[i] > last_sh:
            side = 1
        elif last_sl is not None and c[i] < last_sl:
            side = -1
        if side == 0:
            i += 1
            continue

        # --- 2. first fair value gap after the CHoCH
        fvg = None
        for j in range(i+1, min(i+40, n-1)):
            if side > 0 and h[j-2] < l[j]:
                fvg = (h[j-2], l[j], j-1)      # gap edges, producing candle
                break
            if side < 0 and l[j-2] > h[j]:
                fvg = (h[j], l[j-2], j-1)
                break
        if fvg is None:
            i += 1
            continue
        lo_g, hi_g, prod = fvg
        mid = (lo_g + hi_g)/2
        rng = abs(h[prod] - l[prod])
        stop = (l[prod] - buf*rng) if side > 0 else (h[prod] + buf*rng)
        risk = abs(mid - stop)
        if risk <= 0 or rng <= 0:
            i += 1
            continue
        tgt = mid + side*tmult*risk

        # --- 3. limit fill at the midpoint, then manage
        entry_k = None
        for k in range(prod+2, min(prod+60, n)):
            if entry_k is None:
                touched = (l[k] <= mid) if side > 0 else (h[k] >= mid)
                if touched:
                    entry_k = k
                continue
            # break-even once a candle CLOSES past the swing being broken
            if be_on_close and stop != mid:
                ref = last_sh if side > 0 else last_sl
                broke = (c[k] > ref) if side > 0 else (c[k] < ref)
                if broke:
                    stop = mid
            hit_s = (l[k] <= stop) if side > 0 else (h[k] >= stop)
            hit_t = (h[k] >= tgt) if side > 0 else (l[k] <= tgt)
            if hit_s:                       # stop checked FIRST, conservative
                r = (stop - mid)*side/risk
                out.append(r - COST_BP/1e4*mid/risk)
                i = k + 1
                break
            if hit_t:
                out.append(tmult - COST_BP/1e4*mid/risk)
                i = k + 1
                break
        else:
            i += 1
            continue
        if entry_k is None:
            i += 1
    return out


def backtest(d, **kw):
    m = d.index.hour*60 + d.index.minute
    k = (m >= OPEN_MIN) & (m < CLOSE_MIN)
    dd = d[k]
    day = dd.index.normalize().tz_localize(None)
    rows = []
    for dt, g in dd.groupby(day):
        rs = run_day(g["open"].values, g["high"].values,
                     g["low"].values, g["close"].values, **kw)
        for r in rs:
            rows.append(dict(ts=dt, R=r))
    return pd.DataFrame(rows)


def summary(t, lab):
    if len(t) < 40:
        print(f"  {lab:<22} n={len(t)} (too few)")
        return None
    R = t.R.values
    win = (R > tmult_eps).mean()*100
    be = (np.abs(R) < 0.05).mean()*100
    m, se = R.mean(), R.std(ddof=1)/np.sqrt(len(R))
    print(f"  {lab:<22}{len(R):>7}{win:>8.1f}%{be:>8.1f}%{m:>+9.3f}{m/se:>8.2f}")
    return dict(n=len(R), win=win, mean=m, t=m/se)


tmult_eps = 0.05

if __name__ == "__main__":
    D = {}
    for nm, slug in MK.items():
        d = load_m1(slug)
        if d is not None:
            D[nm] = d
            print(f"{nm}: {len(d):,} 1-min bars, "
                  f"{d.index.min():%Y-%m-%d} -> {d.index.max():%Y-%m-%d}")
    print()

    print("=" * 74)
    print("THE VIDEO'S MODEL, AS SPECIFIED  (1:4 target, BE on close)")
    print("=" * 74)
    print(f"  {'market':<22}{'trades':>7}{'win%':>8}{'BE%':>8}{'expR':>9}{'t':>8}")
    print("  " + "-" * 62)
    allt = []
    for nm, d in D.items():
        t = backtest(d)
        if len(t):
            summary(t, nm)
            t["mkt"] = nm
            allt.append(t)
    if allt:
        A = pd.concat(allt, ignore_index=True)
        A.to_pickle("video_trades.pkl")
        print("  " + "-" * 62)
        summary(A, "POOLED")
        need = 1/(1+4.0)
        print(f"\n  break-even win rate at 1:4 = {need*100:.1f}%")
        print(f"  his claim: 30-40%.  measured above.")
