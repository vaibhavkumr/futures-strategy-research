"""LIVE TRADING BOT — same strategy, real broker.

    python live_trade.py --check          connectivity + account only
    python live_trade.py --dry            full loop, logs orders, sends none
    python live_trade.py                  DEMO account, real orders
    python live_trade.py --live           funded account (needs both switches)
    python live_trade.py --flatten        KILL SWITCH: cancel all, close all

Design rules, each one there because of a specific way bots lose money:

  BROKER IS THE TRUTH. Position and orders are always read from Tradovate,
  never from a local file. A bot that trusts its own state after a crash
  will happily open a second position on top of one it forgot about.

  BRACKET OR NOTHING. Entry, stop and target go in as one OSO request, so a
  filled entry can never sit unprotected.

  ONE POSITION AT A TIME, enforced against the broker's own numbers.

  HARD CAPS that no signal can override: max contracts, max daily loss,
  max consecutive losses, news blackout.

  FAIL CLOSED. Any unexpected error -> stop trading, keep the position
  protected by its resting bracket, alert. Never retry blindly into a market.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime

import pandas as pd

import econ_calendar as EC
import live_paper as LP
from tradovate import Tradovate, TradovateError

STATE = "live_trade_state.json"
LOG = "live_trade_orders.csv"

# ---- hard limits: signals cannot override these -------------------------
SYMBOL = "MNQ"            # micro. NEVER default to full-size NQ.
RISK_PCT = 0.01           # 1% of account per trade
MAX_CONTRACTS = 5         # absolute cap regardless of what sizing says
MAX_DAILY_LOSS_PCT = 0.05  # stop for the day at -5%
MAX_CONSEC_LOSSES = 3
TARGET_R = 0.5            # the r05 rule
POINT_VALUE = 2.0         # MNQ = $2/point


def load_state() -> dict:
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"day": None, "day_start_equity": None, "consec_losses": 0,
            "fired": [], "halted": False, "halt_reason": ""}


def save_state(s: dict):
    json.dump(s, open(STATE, "w"), indent=2)


def log(row: dict):
    hdr = not os.path.exists(LOG)
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=hdr, index=False)


def size_position(equity: float, risk_pts: float) -> int:
    """Contracts such that a stop-out costs ~RISK_PCT of equity."""
    risk_per_contract = risk_pts * POINT_VALUE
    if risk_per_contract <= 0:
        return 0
    n = int((equity * RISK_PCT) // risk_per_contract)
    return max(0, min(n, MAX_CONTRACTS))


def guard(state: dict, tv: Tradovate, equity: float) -> str | None:
    """Return a reason to refuse trading, or None to proceed."""
    if state.get("halted"):
        return f"HALTED: {state.get('halt_reason')}"
    today = datetime.now(LP.ET).strftime("%Y-%m-%d")
    if state.get("day") != today:
        state.update(day=today, day_start_equity=equity, consec_losses=0)
        save_state(state)
    start = state.get("day_start_equity") or equity
    if start > 0 and (equity - start) / start <= -MAX_DAILY_LOSS_PCT:
        return f"daily loss limit ({(equity/start-1)*100:.1f}%)"
    if state.get("consec_losses", 0) >= MAX_CONSEC_LOSSES:
        return f"{state['consec_losses']} consecutive losses"
    return None


def run_once(tv: Tradovate, dry: bool = False) -> None:
    state = load_state()
    tv.ensure()
    contract = tv.find_contract(SYMBOL)
    cid = contract["id"]

    # --- BROKER IS THE TRUTH -------------------------------------------
    net = tv.net_position(cid)
    working = [o for o in tv.working_orders() if o.get("contractId") == cid]
    equity = tv.cash_balance()
    now = datetime.now(LP.ET)
    print(f"[{now:%Y-%m-%d %H:%M ET}] {contract['name']}  equity ${equity:,.2f}  "
          f"pos {net:+d}  working {len(working)}")

    if net != 0 or working:
        print("   position or orders live -> managing, no new entries")
        return

    reason = guard(state, tv, equity)
    if reason:
        print(f"   not trading: {reason}")
        return

    blocked, why = EC.blackout(now)
    if blocked:
        print(f"   news blackout: {why}")
        return

    # --- signal (same logic as the paper bot) ---------------------------
    df = LP.fetch()
    if not LP.in_session(df.index[-1]):
        print("   outside killzone")
        return
    sig = LP.find_signal(df)
    if not sig:
        print("   no signal")
        return
    ts, side, entry, stop, risk_pts = sig
    if str(ts) in state["fired"]:
        print("   signal already traded")
        return
    if (df.index[-1] - ts).total_seconds() > 3600:
        print("   signal stale")
        return

    qty = size_position(equity, risk_pts)
    if qty < 1:
        print(f"   size < 1 contract (risk {risk_pts:.1f} pts on ${equity:,.0f})")
        return
    sgn = 1 if side == "long" else -1
    target = entry + sgn * TARGET_R * risk_pts
    action = "Buy" if side == "long" else "Sell"

    print(f"   SIGNAL {side.upper()} {qty}x @ {entry:.2f}  stop {stop:.2f}  "
          f"target {target:.2f}  (risk ${risk_pts*POINT_VALUE*qty:,.0f})")
    if dry:
        print("   [dry run - no order sent]")
        return

    res = tv.place_bracket(cid, action, qty, entry, stop, target,
                           tag=f"tjr {side} {ts:%H%M}")
    state["fired"] = (state["fired"] + [str(ts)])[-200:]
    save_state(state)
    log({"time": str(now), "contract": contract["name"], "side": side,
         "qty": qty, "entry": entry, "stop": stop, "target": target,
         "risk_pts": risk_pts, "equity": equity, "response": str(res)[:200]})
    print(f"   ORDER SENT: {res}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="funded account")
    ap.add_argument("--dry", action="store_true", help="no orders sent")
    ap.add_argument("--check", action="store_true", help="connectivity only")
    ap.add_argument("--flatten", action="store_true", help="KILL SWITCH")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    try:
        tv = Tradovate(live=a.live).connect()
    except TradovateError as e:
        print(f"connection failed:\n  {e}")
        raise SystemExit(1)

    env = "LIVE (real money)" if a.live else "DEMO"
    print(f"connected: {env}  account {tv.account_spec}  ${tv.cash_balance():,.2f}")

    if a.flatten:
        c = tv.find_contract(SYMBOL)
        print("FLATTENING:", tv.flatten(c["id"]))
        s = load_state(); s["halted"] = True; s["halt_reason"] = "manual flatten"
        save_state(s)
        return
    if a.check:
        c = tv.find_contract(SYMBOL)
        print(f"contract {c['name']} id {c['id']}  pos {tv.net_position(c['id'])}  "
              f"working {len(tv.working_orders())}")
        return
    if a.live:
        print("\n*** LIVE MONEY. Ctrl-C now if that is not intended. ***")
        time.sleep(5)

    while True:
        try:
            run_once(tv, dry=a.dry)
        except TradovateError as e:
            print(f"  broker error: {e}")
        except Exception:
            # Fail closed: stop trading, leave the resting bracket protecting
            # any open position, and require a human to look.
            traceback.print_exc()
            s = load_state()
            s["halted"] = True
            s["halt_reason"] = "unhandled exception -- inspect before resuming"
            save_state(s)
            print("  HALTED on unexpected error. Positions keep their brackets.")
            break
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
