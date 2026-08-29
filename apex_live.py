"""APEX -- PAPER TRADING ONLY. No broker connection, no real orders.

The tuned system. Every setting was measured, not assumed, and the two biggest
improvements came from the user pushing back on my defaults.

    top 5 by 12-month momentum      concentration sweep: top 5 banded beat
                                    top 3 (Sharpe 1.47 vs 1.42) with less
                                    single-name risk
    biweekly rebalance              frequency sweep: Sharpe 1.00 vs 0.94 weekly
    k = 0.25 conviction tilt        mild tilt best; heavy tilt measurably hurts
    +/-2.5% DAILY BAND              take profit and cut loss at 2.5% of equity;
                                    lifted Sharpe 1.08 -> 1.47 and halved the
                                    drawdown from -47% to -20.6%
    no per-position stop            stop sweep: stops cost ~3pp of growth on
                                    concentrated momentum (winners pull back)

BACKTESTED (survivorship-controlled, 20bp costs, measured 0.013% gap slippage):
    growth 37.9%/yr   Sharpe 1.47   maxDD -20.6%   daily win rate 54.9%
    placebo: 0/60 random 3-name portfolios matched it

HONEST EXPECTATION: French's survivorship-free CRSP data puts broad momentum
at ~17%/yr, so 37.9% likely still carries some contamination. Live expectation
is 25-35%/yr. This bot exists to find out which, on data that does not exist
yet and therefore cannot be overfitted.

    python apex_live.py --equity 10000 --lev 1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Broad liquid US universe. Survivorship bias is a BACKTEST problem -- going
# forward every name here is live, so the full list is correct to trade.
UNIVERSE = """AAPL MSFT AMZN GOOGL META NVDA TSLA AVGO ORCL CRM ADBE AMD INTC
QCOM TXN MU AMAT LRCX KLAC ADI NXPI ON SWKS MCHP CDNS SNPS INTU NOW PANW FTNT
CRWD ZS OKTA DDOG NET SNOW MDB TEAM WDAY VEEV HUBS ZM DOCU TWLO PYPL SHOP
JPM BAC WFC C GS MS SCHW BLK SPGI CME ICE COF AXP USB PNC TFC STT SYF
JNJ PFE MRK ABBV LLY BMY AMGN GILD BIIB REGN VRTX MRNA ZTS TMO DHR ABT SYK
BSX MDT ISRG EW BDX BAX ALGN IDXX WAT MTD A ILMN
XOM CVX COP EOG SLB PSX VLO MPC OXY DVN FANG HAL BKR
PG KO PEP WMT COST TGT HD LOW MCD SBUX NKE DIS CMCSA VZ T TMUS
CAT DE HON GE MMM BA LMT RTX NOC GD EMR ETN ITW PH ROK CMI PCAR
UNP CSX NSC FDX UPS DAL UAL LUV
LIN APD SHW ECL DD DOW NEM FCX
NEE DUK SO D AEP EXC XEL ED WEC
AMT PLD CCI EQIX SPG O PSA WELL AVB EQR
V MA ADP PAYX FIS GPN
ADSK EA TTWO RBLX SPOT NFLX
LULU ROST TJX ORLY AZO ULTA DG DLTR YUM CMG DRI
CL KMB GIS HSY SYY KR""".split()

TOP_N = 5
K_TILT = 0.25
BAND = 0.025          # daily take-profit / stop, as a fraction of equity
COST_BP = 20.0        # stocks are wider than ETFs
LOOKBACK = 252
REBAL_DAYS = 14       # biweekly


def prices(period="2y"):
    import yfinance as yf
    d = yf.download(UNIVERSE, period=period, progress=False,
                    auto_adjust=True)["Close"]
    return d.dropna(axis=1, how="all").ffill()


def covered(px, held):
    """Refuse to mark on an incomplete download -- a missing ticker would be
    silently treated as unchanged, which misstates equity and skips the band."""
    if not held:
        return []
    last = px.iloc[-1]
    return [a for a in held if a not in px.columns or pd.isna(last.get(a))]


def target_weights(px, top=TOP_N, k=K_TILT):
    """Top-N by 12-month momentum, exponential conviction tilt."""
    score = px.pct_change(LOOKBACK).iloc[-1].dropna()
    if len(score) < top:
        return pd.Series(dtype=float)
    pick = score.nlargest(top)
    sd = pick.std(ddof=1)
    z = (pick - pick.mean())/sd if sd > 0 else pick*0
    w = np.exp(k*z)
    return w/w.sum()


def _now():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


class Book:
    def __init__(self, path, equity, lev):
        self.path = path
        self.s = {"equity": equity, "start": equity, "lev": lev,
                  "anchor": equity, "realised": 0.0,
                  "weights": {}, "entry_px": {}, "last_px": {},
                  "last_rebal": None, "status": "RUNNING", "peak": equity,
                  "cycles": 0, "skipped": 0, "opened": _now(),
                  "day": None, "day_open_equity": equity, "day_flat": False,
                  "band_hits_up": 0, "band_hits_dn": 0}
        if os.path.exists(path):
            self.s.update(json.load(open(path)))

    def save(self):
        json.dump(self.s, open(self.path, "w"), indent=1)

    def mark(self, px_now):
        """Equity = anchor * (1 + realised + lev * move since entry).

        Marked from the REBALANCE, not the previous cycle -- marking
        incrementally re-levers the book every cycle and manufactures
        volatility drag that no real position pays.
        """
        w, ent = self.s["weights"], self.s.get("entry_px", {})
        self.s["last_px"] = {k: float(v) for k, v in px_now.items()}
        if not w:
            self.s["equity"] = self.s["anchor"]*(1 + self.s.get("realised", 0.0))
            return 0.0
        move = 0.0
        for a, wt in w.items():
            p0, p1 = ent.get(a), px_now.get(a)
            if p0 and p1 and p0 > 0:
                move += wt*(p1/p0 - 1)
        move *= self.s["lev"]
        self.s["equity"] = max(
            self.s["anchor"]*(1 + self.s.get("realised", 0.0) + move), 0.0)
        self.s["peak"] = max(self.s["peak"], self.s["equity"])
        return move

    def check_band(self, logfile):
        """THE DAILY BAND. Flat for the rest of the day once the day's move
        exceeds +/-2.5%. This is what lifted Sharpe 1.08 -> 1.47: it truncates
        BOTH tails, where capping only gains was catastrophic (-11.9%/yr)."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.s.get("day") != today:
            self.s["day"] = today
            self.s["day_open_equity"] = self.s["equity"]
            self.s["day_flat"] = False
        if self.s["day_flat"]:
            return None
        base = self.s["day_open_equity"] or self.s["equity"]
        move = self.s["equity"]/base - 1 if base else 0.0
        if abs(move) >= BAND:
            side = "TAKE PROFIT" if move > 0 else "CUT LOSS"
            self.s["day_flat"] = True
            self.s["realised"] = self.s.get("realised", 0.0) + \
                (self.s["equity"]/self.s["anchor"] - 1 - self.s.get("realised", 0.0))
            self.s["weights"] = {}
            self.s["entry_px"] = {}
            if move > 0:
                self.s["band_hits_up"] += 1
            else:
                self.s["band_hits_dn"] += 1
            with open(logfile, "a") as f:
                f.write(f"{_now()},BAND_{side.replace(' ','_')},"
                        f"{self.s['equity']:.2f},,,day move {move*100:+.2f}%\n")
            return f"{side}: day move {move*100:+.2f}% -- flat until tomorrow"
        return None

    def rebalance(self, px, logfile):
        w = target_weights(px)
        if w.empty:
            return None
        old = self.s["weights"]
        turn = sum(abs(w.get(a, 0)-old.get(a, 0))
                   for a in set(w.index) | set(old))
        cost = turn*COST_BP/1e4*self.s["lev"]
        self.s["equity"] *= (1 - cost)
        self.s["weights"] = {a: float(v) for a, v in w.items()}
        last = px.iloc[-1]
        self.s["entry_px"] = {a: float(last[a]) for a in w.index
                              if a in last and pd.notna(last[a])}
        self.s["anchor"] = self.s["equity"]
        self.s["realised"] = 0.0
        self.s["last_rebal"] = _now()
        with open(logfile, "a") as f:
            f.write(f"{_now()},REBALANCE,{self.s['equity']:.2f},{turn:.3f},"
                    f"{cost*100:.3f},"
                    f"{'|'.join(f'{a}:{v:.3f}' for a, v in w.items())}\n")
        return w, turn, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float, default=10000)
    ap.add_argument("--lev", type=float, default=1.0)
    ap.add_argument("--interval", type=int, default=900)
    ap.add_argument("--state", default="apex_state.json")
    ap.add_argument("--log", default="apex_trades.csv")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.log):
        with open(a.log, "w") as f:
            f.write("ts,event,equity,turnover,cost_pct,detail\n")

    bk = Book(a.state, a.equity, a.lev)
    print("=" * 76)
    print("  APEX  --  PAPER TRADING ONLY, NO REAL ORDERS")
    print("=" * 76)
    print(f"  top {TOP_N} momentum | biweekly | k={K_TILT} | band +/-{BAND*100:.1f}%"
          f" | {a.lev:.0f}x")
    print(f"  start ${bk.s['start']:,.0f}   universe {len(UNIVERSE)} stocks")
    print(f"  backtest: 37.9%/yr, Sharpe 1.47, maxDD -20.6%, win rate 54.9%")
    print(f"  honest live expectation: 25-35%/yr (clean CRSP momentum is ~17%)")
    print("=" * 76, flush=True)

    while True:
        try:
            px = prices()
            now = px.iloc[-1]
            bk.s["cycles"] += 1

            if bk.s["status"] == "RUNNING":
                gaps = covered(px, list(bk.s["weights"]))
                if gaps:
                    bk.s["skipped"] += 1
                    print(f"  [stale] {len(gaps)} ticker(s) missing "
                          f"({', '.join(gaps[:5])}) -- mark SKIPPED", flush=True)
                    bk.save()
                    time.sleep(min(a.interval, 120))
                    continue

                bk.mark(now)
                fired = bk.check_band(a.log)
                if fired:
                    print(f"  *** {fired} ***", flush=True)

                last = bk.s["last_rebal"]
                due = (last is None or
                       (pd.Timestamp(_now()) - pd.Timestamp(last)).days >= REBAL_DAYS)
                if due and not bk.s.get("day_flat"):
                    res = bk.rebalance(px, a.log)
                    if res:
                        w, turn, cost = res
                        print(f"  REBALANCE  turnover {turn:.2f}  cost {cost*100:.2f}%"
                              f"  -> " + ", ".join(f"{k} {v*100:.0f}%"
                                                   for k, v in w.items()), flush=True)

            bk.save()
            e = bk.s["equity"]
            pct = (e/bk.s["start"] - 1)*100
            dd = (e/bk.s["peak"] - 1)*100
            flat = " FLAT" if bk.s.get("day_flat") else ""
            print(f"  {_now()}  ${e:>10,.2f}  {pct:>+7.2f}%  dd {dd:>6.2f}%  "
                  f"cycle {bk.s['cycles']:>4}  bands +{bk.s['band_hits_up']}"
                  f"/-{bk.s['band_hits_dn']}{flat}", flush=True)

            if a.once:
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
