"""Tradovate API client — auth, market data, orders, positions.

CREDENTIALS: this module never contains them and never asks you to paste
them anywhere. It reads environment variables you set yourself:

    setx TRADOVATE_USER      "your_username"
    setx TRADOVATE_PASS      "your_password"
    setx TRADOVATE_APPID     "your_app_name"
    setx TRADOVATE_CID       "your_cid"
    setx TRADOVATE_SECRET    "your_api_secret"

(open a NEW terminal after setx for them to take effect)

DEMO BY DEFAULT. Live trading requires passing live=True explicitly AND
setting TRADOVATE_ALLOW_LIVE=1. Two independent switches, deliberately.

Get API credentials from Tradovate: Application Settings -> API Access.
The demo environment is free and uses the same API surface as live.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEMO_URL = "https://demo.tradovateapi.com/v1"
LIVE_URL = "https://live.tradovateapi.com/v1"
MD_URL = "https://md.tradovateapi.com/v1"


class TradovateError(RuntimeError):
    pass


@dataclass
class Tradovate:
    live: bool = False
    token: str | None = None
    md_token: str | None = None
    account_id: int | None = None
    account_spec: str | None = None
    expires: float = 0.0
    _last_call: float = field(default=0.0, repr=False)

    # ---------------------------------------------------------------- base
    @property
    def base(self) -> str:
        return LIVE_URL if self.live else DEMO_URL

    def _throttle(self):
        """Tradovate rate-limits aggressively; keep a floor between calls."""
        gap = time.time() - self._last_call
        if gap < 0.25:
            time.sleep(0.25 - gap)
        self._last_call = time.time()

    def _req(self, path: str, payload: dict | None = None, method: str | None = None,
             base: str | None = None, auth: bool = True) -> Any:
        self._throttle()
        url = f"{base or self.base}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data,
                                     method=method or ("POST" if data else "GET"))
        req.add_header("Content-Type", "application/json")
        if auth:
            if not self.token:
                raise TradovateError("not authenticated -- call connect() first")
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read().decode()
            return json.loads(body) if body else None
        except urllib.error.HTTPError as e:
            raise TradovateError(f"{e.code} {path}: {e.read().decode()[:300]}") from None

    # ---------------------------------------------------------------- auth
    def connect(self) -> "Tradovate":
        need = ["TRADOVATE_USER", "TRADOVATE_PASS", "TRADOVATE_APPID",
                "TRADOVATE_CID", "TRADOVATE_SECRET"]
        missing = [k for k in need if not os.environ.get(k)]
        if missing:
            raise TradovateError(
                "missing environment variables: " + ", ".join(missing) +
                "\nSet them with setx (see module docstring). They are never "
                "stored in this repo.")
        if self.live and os.environ.get("TRADOVATE_ALLOW_LIVE") != "1":
            raise TradovateError(
                "live=True requires TRADOVATE_ALLOW_LIVE=1. Refusing to touch "
                "a funded account without that second switch.")
        body = {
            "name": os.environ["TRADOVATE_USER"],
            "password": os.environ["TRADOVATE_PASS"],
            "appId": os.environ["TRADOVATE_APPID"],
            "appVersion": "1.0",
            "cid": int(os.environ["TRADOVATE_CID"]),
            "sec": os.environ["TRADOVATE_SECRET"],
        }
        r = self._req("/auth/accesstokenrequest", body, auth=False)
        if not r or "accessToken" not in r:
            raise TradovateError(f"auth failed: {r}")
        self.token = r["accessToken"]
        self.md_token = r.get("mdAccessToken")
        self.expires = time.time() + 60 * 70          # tokens last ~80 min
        accts = self._req("/account/list")
        if not accts:
            raise TradovateError("no accounts on this login")
        self.account_id = accts[0]["id"]
        self.account_spec = accts[0]["name"]
        return self

    def ensure(self):
        if not self.token or time.time() > self.expires:
            self.connect()

    # ------------------------------------------------------------ contracts
    def find_contract(self, symbol: str = "MNQ") -> dict:
        """Resolve the FRONT-MONTH contract. Trading the wrong expiry is a
        classic and expensive mistake, so we always resolve, never hardcode."""
        self.ensure()
        r = self._req(f"/contract/suggest?t={symbol}&l=10")
        if not r:
            raise TradovateError(f"no contract found for {symbol}")
        # suggest returns front month first for continuous symbols
        return r[0]

    # -------------------------------------------------------------- account
    def positions(self) -> list[dict]:
        self.ensure()
        return self._req("/position/list") or []

    def net_position(self, contract_id: int) -> int:
        for p in self.positions():
            if p.get("contractId") == contract_id:
                return int(p.get("netPos", 0))
        return 0

    def orders(self) -> list[dict]:
        self.ensure()
        return self._req("/order/list") or []

    def working_orders(self) -> list[dict]:
        live_states = {"Working", "Pending", "Suspended"}
        return [o for o in self.orders() if o.get("ordStatus") in live_states]

    def cash_balance(self) -> float:
        self.ensure()
        r = self._req("/cashBalance/list") or []
        for c in r:
            if c.get("accountId") == self.account_id:
                return float(c.get("amount", 0))
        return 0.0

    # --------------------------------------------------------------- orders
    def place_bracket(self, contract_id: int, action: str, qty: int,
                      entry: float, stop: float, target: float,
                      tag: str = "") -> dict:
        """Entry LIMIT with an OCO stop/target attached (Tradovate OSO).

        The bracket is submitted as ONE request so the protective orders can
        never be missing while a position is open -- the failure mode that
        actually blows up automated accounts.
        """
        self.ensure()
        opp = "Sell" if action == "Buy" else "Buy"
        body = {
            "accountSpec": self.account_spec,
            "accountId": self.account_id,
            "action": action,
            "symbol": None,
            "contractId": contract_id,
            "orderQty": qty,
            "orderType": "Limit",
            "price": round(entry, 2),
            "timeInForce": "GTC",
            "isAutomated": True,
            "text": tag[:60],
            "bracket1": {"action": opp, "orderType": "Stop",
                         "stopPrice": round(stop, 2), "timeInForce": "GTC"},
            "bracket2": {"action": opp, "orderType": "Limit",
                         "price": round(target, 2), "timeInForce": "GTC"},
        }
        return self._req("/order/placeOSO", body)

    def cancel(self, order_id: int) -> dict:
        self.ensure()
        return self._req("/order/cancelorder", {"orderId": order_id})

    def cancel_all(self) -> int:
        n = 0
        for o in self.working_orders():
            try:
                self.cancel(o["id"])
                n += 1
            except TradovateError:
                pass
        return n

    def flatten(self, contract_id: int) -> dict | None:
        """Cancel everything, then market out of any remaining position.
        This is the kill switch. It must work even if bot state is corrupt,
        so it reads position from the BROKER, never from local state."""
        self.ensure()
        self.cancel_all()
        net = self.net_position(contract_id)
        if net == 0:
            return None
        return self._req("/order/placeorder", {
            "accountSpec": self.account_spec, "accountId": self.account_id,
            "action": "Sell" if net > 0 else "Buy",
            "contractId": contract_id, "orderQty": abs(net),
            "orderType": "Market", "isAutomated": True,
            "text": "KILL SWITCH"})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Tradovate connectivity check")
    ap.add_argument("--symbol", default="MNQ")
    ap.add_argument("--flatten", action="store_true", help="KILL SWITCH")
    a = ap.parse_args()
    try:
        tv = Tradovate(live=False).connect()
    except TradovateError as e:
        print(f"could not connect:\n  {e}")
        raise SystemExit(1)
    print(f"connected to DEMO  account={tv.account_spec} (id {tv.account_id})")
    print(f"cash balance: ${tv.cash_balance():,.2f}")
    c = tv.find_contract(a.symbol)
    print(f"front contract: {c['name']}  (id {c['id']})")
    print(f"net position:   {tv.net_position(c['id'])}")
    wo = tv.working_orders()
    print(f"working orders: {len(wo)}")
    if a.flatten:
        print("FLATTENING…", tv.flatten(c["id"]))
