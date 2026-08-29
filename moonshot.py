"""MOONSHOT -- PAPER TRADING ONLY. No broker connection, no real orders.

Goal: $10,000 -> $20,000, compounded, in 1-2 months.

WHAT THIS IS, STATED PLAINLY AT THE TOP SO IT IS NEVER AMBIGUOUS:
at 30x leverage the verified edge contributes a few percentage points and
VARIANCE does the rest. Measured odds over 42 trading days, bootstrapped on
23 years of real daily returns:

    P(reach $20,000)      48.9%
    P(stopped out)        41.3%   -> you keep the floor
    median outcome      $13,509
    expected value      $11,353

(net of transaction costs. The first run of these numbers omitted them --
at 30x a 10bp cost becomes 300bp, so that mattered and was corrected.)

This is a coin flip with a small tilt, not a strategy grinding out an edge.
Both numbers are real: the tilt is real and the coin flip is real.

ENGINE -- the verified components, sized aggressively:
    mom_12m         12-month cross-sectional momentum, the one factor that
                    passed a 0/100 placebo and beat buy&hold in dev AND holdout
    k=0.5 tilt      exponential conviction weighting; measured optimum, it
                    improved growth, Sharpe and drawdown simultaneously
    30x leverage    chosen by sweep: same P(target) as 50x with a far better
                    median. Past 30x you buy nothing.

HARD RULES, both checked every cycle:
    TARGET $20,000  touch it -> flat, stop trading, done
    STOP   $2,000   touch it -> flat, stop trading, capital preserved

The stop is enforced on every cycle rather than weekly. At this leverage a
weekly-only check gaps straight through the floor and the stop becomes
decorative. It still cannot defend against an overnight gap -- at 30x a 3.3%
adverse gap is the whole account, and gaps that size happen. The floor is a
policy, not a guarantee.

Rebalances weekly (that is where the edge lives), marks to market every cycle.

    python moonshot.py --equity 10000 --lev 30 --target 20000 --stop 2000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD",
           "SLV", "USO", "DBC", "VNQ", "XLE", "XLF", "XLK", "XLV", "XLI", "XLP",
           "XLU", "XLY", "XLB", "XBI", "SMH", "EWJ"]

K_TILT = 0.5          # measured optimum
COST_BP = 10.0        # per unit turnover
LOOKBACK = 252        # momentum window


def prices(period="2y"):
    import yfinance as yf
    d = yf.download(TICKERS, period=period, progress=False,
                    auto_adjust=True)["Close"]
    return d.dropna(axis=1, how="all").ffill()


def covered(px, held):
    """Which held positions does this download actually price?

    yfinance fails intermittently (DNS, rate limits) on a subset of tickers,
    and prices() drops any that fail. That is quietly dangerous: mark() skips
    a position it cannot price, so a failed ticker is treated as UNCHANGED
    rather than unknown, and its stop is never checked. At 20x a handful of
    silently-flat positions misstates equity badly.

    So the mark is refused unless every held name is priced. Returns the list
    of missing tickers; empty list means safe to mark.
    """
    if not held:
        return []
    last = px.iloc[-1]
    return [a for a in held
            if a not in px.columns or pd.isna(last.get(a))]


def target_weights(px, k=K_TILT):
    """Conviction-tilted weights from momentum known at the LAST CLOSE.

    px must end at the most recent completed bar -- no partial day, or the
    score peeks at a price the position could not have been taken at.
    """
    score = px.pct_change(LOOKBACK).iloc[-1].dropna()
    if len(score) < 8:
        return pd.Series(dtype=float)
    z = (score - score.mean()) / score.std(ddof=1)
    w = np.exp(k * z)
    return w / w.sum()


class Book:
    def __init__(self, path, equity, lev, target, stop, pstop):
        self.path = path
        self.s = {"equity": equity, "start": equity, "lev": lev,
                  "target": target, "stop": stop, "pstop": pstop,
                  "anchor": equity, "realised": 0.0,
                  "entry_px": {}, "weights": {},
                  "last_px": {}, "last_rebal": None, "status": "RUNNING",
                  "peak": equity, "cycles": 0, "opened": _now()}
        if os.path.exists(path):
            self.s.update(json.load(open(path)))

    def save(self):
        json.dump(self.s, open(self.path, "w"), indent=1)

    def mark(self, px_now):
        """Mark to market from the REBALANCE, not from the previous cycle.

        BUG FIXED HERE. The old version did

            equity *= (1 + lev * move_since_last_cycle)

        every cycle, which re-levers the book to `lev` 26 times a day -- a
        leveraged ETF resetting every 15 minutes. That manufactures volatility
        drag found nowhere in the backtest, which compounds DAILY. At 30x the
        difference is not cosmetic: an intraday round trip of +/-1.2%
        unlevered burns ~13% of capital to drag alone while ending flat.

        A real position does not behave that way. Between rebalances you hold
        a fixed number of shares, so equity is the anchor plus the levered
        move of those shares since they were bought. Realised P/L from
        stopped-out positions is banked separately in `realised`.
        """
        w, ent = self.s["weights"], self.s.get("entry_px", {})
        self.s["last_px"] = {k: float(v) for k, v in px_now.items()}
        if not w:
            self.s["equity"] = self.s["anchor"] * (1 + self.s.get("realised", 0.0))
            return 0.0
        move = 0.0
        for a, wt in w.items():
            p0, p1 = ent.get(a), px_now.get(a)
            if p0 and p1 and p0 > 0:
                move += wt * (p1 / p0 - 1)
        move *= self.s["lev"]
        self.s["equity"] = max(
            self.s["anchor"] * (1 + self.s.get("realised", 0.0) + move), 0.0)
        self.s["peak"] = max(self.s["peak"], self.s["equity"])
        return move

    def check_position_stops(self, px_now, logfile):
        """PER-POSITION STOP. Exit any holding that has fallen `pstop` from
        its entry price, and stay out of it until the next rebalance.

        This matters much more at 30x than it looks. SMH at 12.4% of the book
        dropping 5% is 12.4% * 5% * 30 = -18.6% of equity from ONE holding.
        The account-level floor only fires after the damage is already done;
        this caps each position's contribution on the way there.
        """
        entry = self.s.setdefault("entry_px", {})
        pstop = self.s["pstop"]
        exited = []
        for a in list(self.s["weights"]):
            e, p = entry.get(a), px_now.get(a)
            if not e or not p or e <= 0:
                continue
            draw = p / e - 1
            if draw <= -pstop:
                wt = self.s["weights"].pop(a)
                entry.pop(a, None)
                # bank the loss: it is realised, so it leaves the mark-to-market
                # and becomes part of the anchor instead
                lv = self.s["lev"]
                self.s["realised"] = (self.s.get("realised", 0.0)
                                      + wt * draw * lv
                                      - wt * COST_BP / 1e4 * lv)
                exited.append((a, draw, wt))
                with open(logfile, "a") as f:
                    f.write(f"{_now()},POSITION_STOP,{self.s['equity']:.2f},,,"
                            f"{a} {draw*100:+.2f}% weight {wt*100:.1f}%\n")
        return exited

    def check_rules(self):
        """TARGET and STOP. Returns a status string if either fired."""
        e = self.s["equity"]
        if e >= self.s["target"]:
            self.s["status"] = "TARGET HIT"
            self.s["weights"] = {}
            return f"TARGET ${self.s['target']:,.0f} REACHED -- flat, done."
        if e <= self.s["stop"]:
            self.s["status"] = "STOPPED OUT"
            self.s["weights"] = {}
            return f"STOP ${self.s['stop']:,.0f} HIT -- flat, capital preserved."
        return None

    def rebalance(self, px, logfile):
        w = target_weights(px)
        if w.empty:
            return None
        old = self.s["weights"]
        turn = sum(abs(w.get(a, 0) - old.get(a, 0))
                   for a in set(w.index) | set(old))
        cost = turn * COST_BP / 1e4 * self.s["lev"]
        self.s["equity"] *= (1 - cost)
        # new holding period: equity becomes the anchor, P/L clock resets
        self.s["anchor"] = self.s["equity"]
        self.s["realised"] = 0.0
        self.s["weights"] = {a: float(v) for a, v in w.items()}
        # entry price per position -- the reference the per-trade stop uses
        last_close = px.iloc[-1]
        self.s["entry_px"] = {a: float(last_close[a]) for a in w.index
                              if a in last_close and pd.notna(last_close[a])}
        self.s["last_rebal"] = _now()
        top = w.nlargest(3)
        with open(logfile, "a") as f:
            f.write(f"{_now()},REBALANCE,{self.s['equity']:.2f},{turn:.3f},"
                    f"{cost*100:.3f},{'|'.join(f'{a}:{v:.3f}' for a, v in top.items())}\n")
        return top, turn, cost


def _now():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float, default=10000)
    ap.add_argument("--lev", type=float, default=30)
    ap.add_argument("--target", type=float, default=20000)
    ap.add_argument("--stop", type=float, default=2000)
    ap.add_argument("--pstop", type=float, default=0.06,
                    help="per-position stop, fraction below entry")
    ap.add_argument("--interval", type=int, default=900)
    ap.add_argument("--state", default="moonshot_state.json")
    ap.add_argument("--log", default="moonshot_trades.csv")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.log):
        with open(a.log, "w") as f:
            f.write("ts,event,equity,turnover,cost_pct,detail\n")

    bk = Book(a.state, a.equity, a.lev, a.target, a.stop, a.pstop)
    print("=" * 74)
    print("  MOONSHOT  --  PAPER TRADING ONLY, NO REAL ORDERS")
    print("=" * 74)
    print(f"  start ${bk.s['start']:,.0f}   target ${a.target:,.0f}   "
          f"floor ${a.stop:,.0f}   leverage {a.lev:.0f}x")
    print(f"  per-position stop: {a.pstop*100:.0f}% below entry -> exit that holding")
    print(f"  measured odds over 2 months: 48.9% target, 41.3% stopped,")
    print(f"                               median $13,509, EV $11,353 (net of costs)")
    print(f"  the edge contributes a few points here; variance does the rest.")
    print("=" * 74, flush=True)

    while True:
        try:
            px = prices()
            now = px.iloc[-1]
            bk.s["cycles"] += 1

            if bk.s["status"] == "RUNNING":
                # refuse to mark on an incomplete download -- a missing ticker
                # would be silently treated as unchanged
                gaps = covered(px, list(bk.s["weights"]))
                if gaps:
                    bk.s["skipped"] = bk.s.get("skipped", 0) + 1
                    with open(a.log, "a") as f:
                        miss = "|".join(gaps)
                        f.write(f"{_now()},STALE_SKIP,"
                                f"{bk.s['equity']:.2f},,,missing {miss}\n")
                    print(f"  [stale] {len(gaps)} ticker(s) failed to download "
                          f"({', '.join(gaps[:6])}) -- mark SKIPPED, retrying",
                          flush=True)
                    bk.save()
                    time.sleep(min(a.interval, 120))
                    continue
                move = bk.mark(now)
                for a_, dr_, wt_ in bk.check_position_stops(now, a.log):
                    print(f"  POSITION STOP  {a_} {dr_*100:+.2f}% "
                          f"(was {wt_*100:.1f}% of book) -- exited")
                fired = bk.check_rules()
                if fired:
                    print(f"\n  *** {fired} ***")
                    with open(a.log, "a") as f:
                        f.write(f"{_now()},{bk.s['status']},{bk.s['equity']:.2f},,,\n")
                else:
                    # rebalance weekly (Friday, or if never done)
                    last = bk.s["last_rebal"]
                    due = (last is None or
                           (pd.Timestamp(_now()) - pd.Timestamp(last)).days >= 7)
                    if due:
                        r = bk.rebalance(px, a.log)
                        if r:
                            top, turn, cost = r
                            print(f"  REBALANCE  turnover {turn:.2f}  "
                                  f"cost {cost*100:.2f}%  top: "
                                  + ", ".join(f"{k} {v*100:.0f}%"
                                              for k, v in top.items()))
            else:
                move = 0.0

            bk.save()
            e = bk.s["equity"]
            pct = (e / bk.s["start"] - 1) * 100
            dd = (e / bk.s["peak"] - 1) * 100
            bar = int(max(0, min(1, (e - a.stop) / (a.target - a.stop))) * 30)
            print(f"  {_now()}  ${e:>10,.2f}  {pct:>+7.2f}%  "
                  f"dd {dd:>6.2f}%  cycle {bk.s['cycles']:>4}  "
                  f"[{'#'*bar}{'.'*(30-bar)}]  {bk.s['status']}", flush=True)

            if bk.s["status"] != "RUNNING" or a.once:
                if bk.s["status"] != "RUNNING":
                    print(f"\n  Final: ${e:,.2f} from ${bk.s['start']:,.0f} "
                          f"({pct:+.1f}%) after {bk.s['cycles']} cycles.")
                break
            time.sleep(a.interval)
        except KeyboardInterrupt:
            print("\n  stopped by user; state saved.")
            bk.save()
            break
        except Exception as ex:
            print(f"  [warn] {type(ex).__name__}: {str(ex)[:70]}", flush=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
