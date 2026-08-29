"""THE VIDEO'S MODEL -- corrected reference level, plus a fairness sweep.

Two fixes over the first pass:

  BREAK-EVEN REFERENCE. He is specific: break-even triggers on a candle
  closing past "the push up before a pull back down into our fair value gap"
  -- i.e. the local extreme made AFTER the change of character and BEFORE
  price retraces into the gap. My first version used the original CHoCH
  swing, which is a different (and higher) level. Corrected here.

  FAIRNESS SWEEP. Several rules are left to discretion in the video --
  "comfortably outside" the candle, "recent" swing, how far to trail. If the
  model only fails at one arbitrary parameter choice, that is my failure and
  not the strategy's. So the stop buffer, target multiple, swing lookback,
  and break-even on/off are all swept. The strategy gets its best case.

His arithmetic: at 1:4 break-even is 20% win rate; he claims 30-40%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from video_model import load_m1, swings, MK, COST_BP, OPEN_MIN, CLOSE_MIN


def run_day(o, h, l, c, tmult=4.0, buf=0.10, k_sw=3, be=True):
    n = len(c)
    if n < 30:
        return []
    sh, sl = swings(h, l, k_sw)
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
            side = 1
        elif last_sl is not None and c[i] < last_sl:
            side = -1
        if side == 0:
            i += 1
            continue

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
        rng = abs(h[prod] - l[prod])
        stop0 = (l[prod] - buf*rng) if side > 0 else (h[prod] + buf*rng)
        risk = abs(mid - stop0)
        if risk <= 0 or rng <= 0:
            i += 1
            continue
        tgt = mid + side*tmult*risk
        stop = stop0

        # THE PUSH: local extreme between the CHoCH and the retrace into the gap.
        # This is the level he watches for the break-of-structure close.
        push = h[i:prod+2].max() if side > 0 else l[i:prod+2].min()

        entry_k = None
        done = False
        for kk in range(prod+2, min(prod+90, n)):
            if entry_k is None:
                touched = (l[kk] <= mid) if side > 0 else (h[kk] >= mid)
                if touched:
                    entry_k = kk
                continue
            if be and stop != mid:
                broke = (c[kk] > push) if side > 0 else (c[kk] < push)
                if broke:
                    stop = mid
            hit_s = (l[kk] <= stop) if side > 0 else (h[kk] >= stop)
            hit_t = (h[kk] >= tgt) if side > 0 else (l[kk] <= tgt)
            if hit_s:
                out.append((stop - mid)*side/risk - COST_BP/1e4*mid/risk)
                i = kk + 1
                done = True
                break
            if hit_t:
                out.append(tmult - COST_BP/1e4*mid/risk)
                i = kk + 1
                done = True
                break
        if not done:
            i += 1
    return out


def backtest(d, **kw):
    m = d.index.hour*60 + d.index.minute
    dd = d[(m >= OPEN_MIN) & (m < CLOSE_MIN)]
    day = dd.index.normalize().tz_localize(None)
    rows = []
    for dt, g in dd.groupby(day):
        for r in run_day(g["open"].values, g["high"].values,
                         g["low"].values, g["close"].values, **kw):
            rows.append(dict(ts=dt, R=r))
    return pd.DataFrame(rows)


def line(lab, t, tmult):
    if len(t) < 40:
        print(f"  {lab:<28} n={len(t)} (too few)")
        return None
    R = t.R.values
    win = (R > tmult*0.5).mean()*100
    bee = (np.abs(R) < 0.06).mean()*100
    m, se = R.mean(), R.std(ddof=1)/np.sqrt(len(R))
    need = 1/(1+tmult)*100
    flag = "  BEATS BE" if win > need and m > 0 else ""
    print(f"  {lab:<28}{len(R):>7}{win:>7.1f}%{bee:>7.1f}%{need:>7.1f}%"
          f"{m:>+9.3f}{m/se:>7.2f}{flag}")
    return dict(win=win, mean=m, t=m/se)


if __name__ == "__main__":
    D = {nm: load_m1(s) for nm, s in MK.items()}
    D = {k: v for k, v in D.items() if v is not None}
    print(f"{len(D)} markets, 1-minute bars 2024-2026\n")

    print("=" * 84)
    print("A. CORRECTED BREAK-EVEN REFERENCE  (close past the post-CHoCH push)")
    print("=" * 84)
    print(f"  {'market':<28}{'trades':>7}{'win%':>7}{'BE%':>7}{'need':>7}"
          f"{'expR':>9}{'t':>7}")
    print("  " + "-" * 72)
    allt = []
    for nm, d in D.items():
        t = backtest(d)
        if len(t):
            line(nm, t, 4.0)
            allt.append(t)
    A = pd.concat(allt, ignore_index=True)
    print("  " + "-" * 72)
    line("POOLED", A, 4.0)

    print("\n" + "=" * 84)
    print("B. FAIRNESS SWEEP -- every discretionary parameter, best case")
    print("=" * 84)
    d0 = D["S&P 500"]
    print(f"  {'variant':<28}{'trades':>7}{'win%':>7}{'BE%':>7}{'need':>7}"
          f"{'expR':>9}{'t':>7}")
    print("  " + "-" * 72, flush=True)
    for tm in (2.0, 3.0, 4.0, 6.0):
        line(f"target 1:{tm:.0f}", backtest(d0, tmult=tm), tm)
    print()
    for bf in (0.05, 0.25, 0.50, 1.00):
        line(f"stop buffer {bf:.2f}x range", backtest(d0, buf=bf), 4.0)
    print()
    for ks in (2, 3, 5):
        line(f"swing lookback {ks}", backtest(d0, k_sw=ks), 4.0)
    print()
    line("NO break-even rule", backtest(d0, be=False), 4.0)
