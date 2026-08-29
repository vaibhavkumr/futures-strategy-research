"""AUDIT A MEME-COIN TRADING WALLET, AND TEST WHETHER IT IS COPYABLE.

Two questions, and the second matters more than the first:

  1. Does this wallet actually make money once LOSERS are counted? A PNL card
     shows one winning position. The chain shows every position.
  2. Would a COPIER have made money? You see their buy only after it lands
     on-chain, by which point their own buy has already moved the price. On a
     thin memecoin a large buy moves price 10-50%, so the copier is buying
     the trader's price impact. That gap is measurable.

Attribution note: wallet ownership here is NOT verified. These addresses come
from public listicles. The analysis is therefore about the STRUCTURE of
memecoin wallet returns, not about any named individual.

Method: pull signatures, fetch parsed transactions, and reconstruct SOL flow
per token from pre/post balances. SOL out on a token = buy, SOL in = sell.
Realised P&L per token = SOL in minus SOL out, for tokens fully exited.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

RPC = "https://api.mainnet-beta.solana.com"
WSOL = "So11111111111111111111111111111111111111112"


def rpc(method, params, retries=5):
    for a in range(retries):
        try:
            r = requests.post(RPC, json={"jsonrpc": "2.0", "id": 1,
                                         "method": method, "params": params},
                              timeout=60)
            if r.status_code == 429:
                time.sleep(2 + a)
                continue
            j = r.json()
            if "result" in j:
                return j["result"]
            time.sleep(1)
        except Exception:
            time.sleep(1 + a)
    return None


def signatures(addr, limit=1000):
    out, before = [], None
    while len(out) < limit:
        p = {"limit": min(1000, limit - len(out))}
        if before:
            p["before"] = before
        r = rpc("getSignaturesForAddress", [addr, p])
        if not r:
            break
        out.extend(r)
        before = r[-1]["signature"]
        if len(r) < p["limit"]:
            break
        time.sleep(0.2)
    return out


def parse_tx(sig, owner):
    """Return {mint: token_delta} and sol_delta for this wallet."""
    tx = rpc("getTransaction", [sig, {"encoding": "jsonParsed",
                                      "maxSupportedTransactionVersion": 0}])
    if not tx or not tx.get("meta"):
        return None
    meta = tx["meta"]
    if meta.get("err"):
        return None
    keys = [k["pubkey"] if isinstance(k, dict) else k
            for k in tx["transaction"]["message"]["accountKeys"]]
    try:
        i = keys.index(owner)
        sol_delta = (meta["postBalances"][i] - meta["preBalances"][i]) / 1e9
    except (ValueError, IndexError):
        sol_delta = 0.0

    pre = {(b["mint"]): float(b["uiTokenAmount"]["uiAmount"] or 0)
           for b in meta.get("preTokenBalances", []) if b.get("owner") == owner}
    post = {(b["mint"]): float(b["uiTokenAmount"]["uiAmount"] or 0)
            for b in meta.get("postTokenBalances", []) if b.get("owner") == owner}
    deltas = {}
    for m in set(pre) | set(post):
        d = post.get(m, 0.0) - pre.get(m, 0.0)
        if abs(d) > 1e-12 and m != WSOL:
            deltas[m] = d
    fee = meta.get("fee", 0) / 1e9
    return dict(sig=sig, blockTime=tx.get("blockTime"), sol=sol_delta,
                fee=fee, deltas=deltas)


def audit(addr, n_sigs=600, label=""):
    print(f"\n{'='*72}\nWALLET {label}: {addr[:12]}...\n{'='*72}", flush=True)
    sigs = signatures(addr, n_sigs)
    print(f"  pulled {len(sigs)} signatures", flush=True)
    if not sigs:
        return None
    rows = []
    for i, s in enumerate(sigs):
        r = parse_tx(s["signature"], addr)
        if r and r["deltas"]:
            rows.append(r)
        if (i + 1) % 100 == 0:
            print(f"    parsed {i+1}/{len(sigs)}  ({len(rows)} token txs)", flush=True)
        time.sleep(0.04)
    if not rows:
        return None
    # aggregate SOL in/out per token
    tok = defaultdict(lambda: dict(sol_in=0.0, sol_out=0.0, qty=0.0,
                                   buys=0, sells=0, first=None, last=None))
    for r in rows:
        for m, d in r["deltas"].items():
            t = tok[m]
            t["qty"] += d
            if d > 0:
                t["sol_out"] += max(-r["sol"], 0.0); t["buys"] += 1
            else:
                t["sol_in"] += max(r["sol"], 0.0); t["sells"] += 1
            bt = r["blockTime"]
            if bt:
                t["first"] = bt if t["first"] is None else min(t["first"], bt)
                t["last"] = bt if t["last"] is None else max(t["last"], bt)
    df = pd.DataFrame([dict(mint=m, **v) for m, v in tok.items()])
    df["pnl"] = df.sol_in - df.sol_out
    df["closed"] = df.qty.abs() < 1e-6
    return df, rows


if __name__ == "__main__":
    WALLETS = {
        "A": "suqh5sHtr8HyJ7q8scBimULPkPpA557prMG47xCHQfK",
        "B": "2fg5QD1eD7rzNNCsvnhmXFm5hqNgwTTG8p7kQ6f3rx6f",
    }
    for lab, a in WALLETS.items():
        out = audit(a, 600, lab)
        if out is None:
            print("  no parsable token activity")
            continue
        df, rows = out
        df.to_pickle(f"wallet_{lab}.pkl")
        c = df[df.closed]
        print(f"\n  tokens touched      : {len(df)}")
        print(f"  fully exited        : {len(c)}")
        if len(c) >= 5:
            print(f"  winners             : {(c.pnl>0).sum()}  "
                  f"({(c.pnl>0).mean()*100:.1f}%)")
            print(f"  total realised P&L  : {c.pnl.sum():+.2f} SOL")
            print(f"  median trade        : {c.pnl.median():+.3f} SOL")
            print(f"  best single trade   : {c.pnl.max():+.2f} SOL")
            if c.pnl.sum() > 0:
                print(f"  best trade as % of total profit: "
                      f"{c.pnl.max()/c.pnl.sum()*100:.0f}%")
