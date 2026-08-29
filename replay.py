"""Replay the LIVE bot logic over recent real NQ data.

Mirrors live_paper.py exactly: same session windows, same swing/FVG
detection, same one-position-at-a-time rule, same two targets, same four
accounts, same risk controls (3-loss daily limit, de-risk at -20%,
halt at -40%).

Purpose: see what the bot would have done last week / last month, day by
day, before waiting for it to happen live.

    python replay.py --days 7
    python replay.py --days 30 --verbose
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from live_paper import (STRATS, SWING_LB, FVG_WINDOW, STOP_SLIP,
                        MAX_LOSSES_PER_DAY, DERISK_DD, HALT_DD,
                        in_session, acct_names, fetch)

# replay.py predates the STRATS refactor; rebuild the views it expects
TARGETS = {k: v["tmult"] for k, v in STRATS.items()}
ACCOUNTS = {k: v["risk"] for k, v in STRATS.items()}


def gen_signals(df: pd.DataFrame):
    """Same detection as live_paper.find_signal, but emitted in one pass."""
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(df)
    w = 2 * SWING_LB + 1
    is_h = (df["high"].rolling(w, center=True).max() == df["high"]).to_numpy()
    is_l = (df["low"].rolling(w, center=True).min() == df["low"]).to_numpy()
    is_h[:SWING_LB] = is_h[n - SWING_LB:] = False
    is_l[:SWING_LB] = is_l[n - SWING_LB:] = False

    lsh = lsl = np.nan
    armed, sweep, expiry = 0, np.nan, -1
    out = []
    for i in range(SWING_LB, n):
        conf = i - SWING_LB
        if is_h[conf]:
            lsh = h[conf]
        if is_l[conf]:
            lsl = l[conf]
        if not np.isnan(lsl) and l[i] < lsl:
            armed, sweep, expiry = 1, l[i], i + FVG_WINDOW
        elif not np.isnan(lsh) and h[i] > lsh:
            armed, sweep, expiry = -1, h[i], i + FVG_WINDOW
        if armed != 0 and i > expiry:
            armed = 0
        if armed == 0 or not in_session(df.index[i]):
            continue
        if armed == 1 and h[i - 2] < l[i]:
            entry, stop = l[i], sweep
            risk = entry - stop
            if risk > max(1e-6, 0.0005 * entry):
                out.append((i, "long", entry, stop, risk))
                armed = 0
        elif armed == -1 and l[i - 2] > h[i]:
            entry, stop = h[i], sweep
            risk = stop - entry
            if risk > max(1e-6, 0.0005 * entry):
                out.append((i, "short", entry, stop, risk))
                armed = 0
    return out


def replay(df: pd.DataFrame, verbose=False):
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(df)
    acc = {k: 10000.0 for k in acct_names()}
    peak = dict(acc)
    maxdd = {k: 0.0 for k in acct_names()}
    curve = {k: [10000.0] for k in acct_names()}
    trades, daily = [], {}
    sigs = gen_signals(df)
    busy_until = -1
    day, losses = None, 0

    for (i, side, entry, stop, risk) in sigs:
        if i <= busy_until:
            continue                      # one position at a time, as live
        d = df.index[i].date()
        if d != day:
            day, losses = d, 0
        if losses >= MAX_LOSSES_PER_DAY:
            continue

        scale = {}
        for k in acct_names():
            dd = 1 - acc[k] / max(peak[k], 1e-9)
            scale[k] = 0.0 if dd >= HALT_DD else (0.5 if dd >= DERISK_DD else 1.0)

        sgn = 1 if side == "long" else -1
        tgts = {v: entry + sgn * m * risk for v, m in TARGETS.items()}
        done = {v: False for v in TARGETS}
        filled = False
        last_j = i

        for j in range(i + 1, min(i + 61, n)):
            if not filled:
                if l[j] <= entry <= h[j]:
                    filled = True
                else:
                    continue
                # A 5-min bar cannot tell us whether the target was reached
                # before or after the limit filled. Assuming "after" was worth
                # ~$24k of fake profit, so exits start on the NEXT bar.
                continue
            for v in TARGETS:
                if done[v]:
                    continue
                R = None
                if side == "long":
                    if l[j] <= stop:
                        R = (stop - STOP_SLIP - entry) / risk
                    elif h[j] >= tgts[v]:
                        R = (tgts[v] - entry) / risk
                else:
                    if h[j] >= stop:
                        R = (entry - stop - STOP_SLIP) / risk
                    elif l[j] <= tgts[v]:
                        R = (entry - tgts[v]) / risk
                if R is None:
                    continue
                for sz, frac in ACCOUNTS.items():
                    k = f"{v}_{sz}"
                    acc[k] *= (1 + R * frac * scale[k])
                    peak[k] = max(peak[k], acc[k])
                    # TRUE max drawdown: worst peak-to-trough seen, not the
                    # drawdown as of the final bar.
                    maxdd[k] = max(maxdd[k], 1 - acc[k] / peak[k])
                    curve[k].append(acc[k])
                done[v] = True
                if R < 0 and v == "r10":
                    losses += 1
                trades.append({"time": df.index[i], "day": d, "variant": v,
                               "side": side, "R": round(R, 3)})
                last_j = j
            if all(done.values()):
                break
        busy_until = last_j
        daily.setdefault(d, []).append(1)

    return pd.DataFrame(trades), acc, maxdd, curve


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    import yfinance as yf
    raw = yf.download("NQ=F", period="60d", interval="5m",
                      progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    idx = pd.to_datetime(raw.index, utc=True).tz_convert("America/New_York")
    df = raw.set_index(idx).sort_index()[["open", "high", "low", "close"]].astype(float)
    cutoff = df.index[-1] - pd.Timedelta(days=a.days)
    df = df[df.index >= cutoff]

    print(f"REPLAY: NQ 5m, {df.index[0]:%Y-%m-%d %H:%M} -> {df.index[-1]:%Y-%m-%d %H:%M}"
          f"  ({len(df)} bars)\n")
    tr, acc, maxdd, curve = replay(df, a.verbose)
    if tr.empty:
        print("no signals in this window.")
        raise SystemExit

    for v in TARGETS:
        s = tr[tr.variant == v]
        print(f"  {v}: {len(s)} trades  win {(s.R>0).mean()*100:.0f}%  "
              f"total {s.R.sum():+.2f}R  avg {s.R.mean():+.3f}R")

    print(f"\n  {'account':<22}{'balance':>12}{'P&L':>10}{'maxDD':>9}")
    for k in acct_names():
        print(f"  {k:<22}{acc[k]:>12,.0f}{acc[k]-10000:>+10,.0f}"
              f"{maxdd[k]*100:>8.1f}%")

    days = tr.day.nunique()
    print(f"\n  {days} trading days, {len(tr[tr.variant=='r10'])} setups "
          f"({len(tr[tr.variant=='r10'])/days:.1f}/day)")
    for k in ("r05_conservative", "r05_aggressive"):
        print(f"  {k}: ${(acc[k]-10000)/days:+,.0f}/day")

    if a.verbose:
        print("\n  DAY BY DAY (0.5R conservative):")
        for d, g in tr[tr.variant == "r05"].groupby("day"):
            print(f"    {d}  {len(g)} trades  {g.R.sum():+.2f}R  "
                  f"{''.join('W' if r>0 else 'L' for r in g.R)}")
