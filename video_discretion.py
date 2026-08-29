"""CODIFYING THE "COMMON SENSE" -- can discretion lift this past fair odds?

The raw model lands at 20.4% win against a 20.0% break-even -- exactly fair
odds. He says the missing ingredient is judgement: "use common sense", "give
it a little bit of space", "this is kind of already broken into its own range,
so it's a little open to interpretation", and above all "be a little bit more
selective and truly find the best trade setups".

Those are not mystical. Each maps to something measurable about the setup at
the moment of entry, using only information available then:

  fvg_atr      size of the fair value gap in ATR units. A bigger gap is a
               stronger displacement -- his "momentum" argument.
  risk_atr     stop distance in ATR. Too tight = stopped by noise, which is
               exactly the case where he says to give it space. Too wide = the
               1:4 target becomes unreachable.
  choch_str    how decisively the change-of-character candle closed beyond the
               swing, in ATR. A marginal poke is not a real break.
  extended     how far price has already travelled from the CHoCH level when
               the gap forms. His "already broken into its own range" objection.
  minutes      minutes since the 09:30 open. He trades right out of the open
               and is often "done in 15 minutes".
  disp         body size of the gap-producing candle in ATR -- displacement
               strength.
  htf_align    is the 15-minute trend in the same direction as the trade.

Each is tested as a standalone filter, on DEV, then anything that helps is
re-checked on HOLDOUT. With ~20 filters tested, one crossing break-even on DEV
by luck is expected -- the holdout is what decides.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from video_model import load_m1, swings, MK, COST_BP, OPEN_MIN, CLOSE_MIN

DEV_END = "2025-10-01"


def atr(h, l, c, n=14):
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    a = np.full(len(c), np.nan)
    if len(tr) >= n:
        a[n:] = pd.Series(tr).rolling(n).mean().values[n-1:]
    return pd.Series(a).bfill().ffill().values


def run_day(o, h, l, c, mins, tmult=4.0, buf=0.10, k_sw=3):
    """Same model, but every trade carries its setup features."""
    n = len(c)
    if n < 30:
        return []
    sh, sl = swings(h, l, k_sw)
    A = atr(h, l, c)
    out = []
    i = 8
    while i < n - 2:
        last_sh = last_sl = None
        for j in range(i-1, max(i-60, 2), -1):
            if last_sh is None and sh[j] and j + k_sw <= i:
                last_sh = h[j]
            if last_sl is None and sl[j] and j + k_sw <= i:
                last_sl = l[j]
            if last_sh is not None and last_sl is not None:
                break
        side = 0
        if last_sh is not None and c[i] > last_sh:
            side, ref = 1, last_sh
        elif last_sl is not None and c[i] < last_sl:
            side, ref = -1, last_sl
        if side == 0:
            i += 1
            continue
        a0 = max(A[i], 1e-9)
        choch_str = abs(c[i] - ref)/a0

        fvg = None
        for j in range(i+1, min(i+40, n-1)):
            if side > 0 and h[j-2] < l[j]:
                fvg = (h[j-2], l[j], j-1)
                break
            if side < 0 and l[j-2] > h[j]:
                fvg = (h[j], l[j-2], j-1)
                break
        if fvg is None:
            i += 1
            continue
        lo_g, hi_g, prod = fvg
        mid = (lo_g + hi_g)/2
        gap = hi_g - lo_g
        rng = abs(h[prod] - l[prod])
        stop0 = (l[prod] - buf*rng) if side > 0 else (h[prod] + buf*rng)
        risk = abs(mid - stop0)
        if risk <= 0 or rng <= 0:
            i += 1
            continue
        tgt = mid + side*tmult*risk
        stop = stop0
        push = h[i:prod+2].max() if side > 0 else l[i:prod+2].min()

        feat = dict(fvg_atr=gap/a0, risk_atr=risk/a0, choch_str=choch_str,
                    extended=abs(push - ref)/a0,
                    minutes=float(mins[prod] - OPEN_MIN),
                    disp=abs(c[prod] - o[prod])/a0, side=side)

        entry_k = None
        done = False
        for kk in range(prod+2, min(prod+90, n)):
            if entry_k is None:
                if (l[kk] <= mid) if side > 0 else (h[kk] >= mid):
                    entry_k = kk
                continue
            if stop != mid:
                if (c[kk] > push) if side > 0 else (c[kk] < push):
                    stop = mid
            hit_s = (l[kk] <= stop) if side > 0 else (h[kk] >= stop)
            hit_t = (h[kk] >= tgt) if side > 0 else (l[kk] <= tgt)
            if hit_s:
                out.append(dict(R=(stop-mid)*side/risk - COST_BP/1e4*mid/risk,
                                **feat))
                i = kk + 1
                done = True
                break
            if hit_t:
                out.append(dict(R=tmult - COST_BP/1e4*mid/risk, **feat))
                i = kk + 1
                done = True
                break
        if not done:
            i += 1
    return out


def backtest(d):
    m = d.index.hour*60 + d.index.minute
    dd = d[(m >= OPEN_MIN) & (m < CLOSE_MIN)]
    day = dd.index.normalize().tz_localize(None)
    mm = (dd.index.hour*60 + dd.index.minute).values
    rows = []
    for dt, g in dd.groupby(day):
        idx = dd.index.get_indexer(g.index)
        for r in run_day(g["open"].values, g["high"].values, g["low"].values,
                         g["close"].values, mm[idx]):
            r["ts"] = dt
            rows.append(r)
    return pd.DataFrame(rows)


def show(lab, t, need=20.0):
    if len(t) < 60:
        print(f"  {lab:<34} n={len(t):<6} (too few)")
        return None
    R = t.R.values
    win = (R > 2.0).mean()*100
    m, se = R.mean(), R.std(ddof=1)/np.sqrt(len(R))
    ok = m > 0
    print(f"  {lab:<34}{len(R):>7}{win:>7.1f}%{m:>+9.3f}{m/se:>7.2f}"
          f"{'  POSITIVE' if ok else '':>10}")
    return dict(n=len(R), win=win, mean=m, t=m/se)


if __name__ == "__main__":
    print("building trade set with setup features...", flush=True)
    parts = []
    for nm, slug in MK.items():
        d = load_m1(slug)
        if d is None:
            continue
        t = backtest(d)
        t["mkt"] = nm
        parts.append(t)
        print(f"  {nm}: {len(t):,} trades", flush=True)
    T = pd.concat(parts, ignore_index=True)
    T.to_pickle("video_feat.pkl")
    T["ts"] = pd.to_datetime(T.ts)
    dev, hold = T[T.ts < DEV_END], T[T.ts >= DEV_END]
    print(f"\n{len(T):,} trades   dev {len(dev):,}   holdout {len(hold):,}\n")

    print("=" * 78)
    print("BASELINE, NO DISCRETION")
    print("=" * 78)
    print(f"  {'filter':<34}{'n':>7}{'win%':>7}{'expR':>9}{'t':>7}")
    print("  " + "-" * 64)
    show("all trades (dev)", dev)

    print("\n" + "=" * 78)
    print("EACH 'COMMON SENSE' FILTER, ON DEV")
    print("=" * 78)
    print(f"  {'filter':<34}{'n':>7}{'win%':>7}{'expR':>9}{'t':>7}")
    print("  " + "-" * 64)
    cands = {}
    for col, qs in (("fvg_atr", [0.5, 0.7]), ("risk_atr", [0.3, 0.5]),
                    ("choch_str", [0.5, 0.7]), ("disp", [0.5, 0.7])):
        for q in qs:
            thr = dev[col].quantile(q)
            f = dev[dev[col] >= thr]
            r = show(f"{col} >= p{int(q*100)}", f)
            if r and r["mean"] > 0:
                cands[(col, q, "hi")] = thr
    for col, qs in (("risk_atr", [0.5, 0.7]), ("extended", [0.3, 0.5]),
                    ("minutes", [0.25, 0.5])):
        for q in qs:
            thr = dev[col].quantile(q)
            f = dev[dev[col] <= thr]
            r = show(f"{col} <= p{int(q*100)}", f)
            if r and r["mean"] > 0:
                cands[(col, q, "lo")] = thr

    print("\n" + "=" * 78)
    print("HOLDOUT CHECK ON ANYTHING POSITIVE IN DEV")
    print("=" * 78)
    if not cands:
        print("  nothing was positive on DEV -- no filter to check.")
    else:
        print(f"  {'filter':<34}{'n':>7}{'win%':>7}{'expR':>9}{'t':>7}")
        print("  " + "-" * 64)
        for (col, q, dirn), thr in cands.items():
            f = hold[hold[col] >= thr] if dirn == "hi" else hold[hold[col] <= thr]
            show(f"{col} {'>=' if dirn=='hi' else '<='} p{int(q*100)}", f)
