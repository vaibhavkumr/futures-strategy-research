"""ORDER FLOW — the one input class we could never test.

Everything until now came from OHLC(V) bars: what price did. Order flow adds
WHO DID IT -- whether trades hit the offer (aggressive buying) or the bid
(aggressive selling). That is the input professional futures traders
actually use, and it is the last untested hypothesis for where an edge could
live.

Binance publishes it free: every 5-min bar carries taker-buy volume and a
trade count, so we can reconstruct aggressor imbalance exactly.

The concept that matters most is ABSORPTION / DELTA DIVERGENCE:
  price makes a new low on heavy aggressive SELLING, but fails to continue
  -> someone large is absorbing that selling -> reversal.
That is invisible in OHLC bars. It is the single best candidate for the
information our fifty other tests were missing.

Features:
  delta          (taker_buy - taker_sell) / volume, i.e. aggressor imbalance
  delta_sweep    delta on the sweep bar itself
  cum_delta_n    cumulative delta over the n bars around the sweep
  divergence     price made a new extreme but cumulative delta did NOT
  avg_trade_sz   volume / trade count -- large prints suggest size players
  trade_intensity trade count vs its rolling average
"""
from __future__ import annotations
import numpy as np
import pandas as pd

STOP_SLIP_BP = 2.0          # crypto: cost in basis points of price


def add_flow(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    v = out["volume"].replace(0, np.nan)
    out["delta"] = (2 * out["taker_buy"] / v - 1).fillna(0.0)
    out["signed_vol"] = out["delta"] * out["volume"]
    out["cum_delta_12"] = out["signed_vol"].rolling(12).sum()
    out["cum_delta_36"] = out["signed_vol"].rolling(36).sum()
    out["avg_trade_sz"] = (out["volume"] / out["ntrades"].replace(0, np.nan)).fillna(0.0)
    out["ats_ratio"] = out["avg_trade_sz"] / out["avg_trade_sz"].rolling(200).mean()
    out["trade_intensity"] = out["ntrades"] / out["ntrades"].rolling(200).mean()
    tr = pd.concat([out["high"] - out["low"],
                    (out["high"] - out["close"].shift()).abs(),
                    (out["low"] - out["close"].shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    return out


def build(df: pd.DataFrame, swing_lb=2, fvg_window=12, tmult=1.0,
          horizon=60, div_look=36) -> pd.DataFrame:
    d = add_flow(df)
    h, l, c = (d[x].to_numpy(float) for x in ("high", "low", "close"))
    atr = d["atr"].to_numpy()
    delta = d["delta"].to_numpy()
    sv = d["signed_vol"].to_numpy()
    cd12 = d["cum_delta_12"].to_numpy()
    cd36 = d["cum_delta_36"].to_numpy()
    ats = d["ats_ratio"].to_numpy()
    ti = d["trade_intensity"].to_numpy()
    n = len(d)
    w = 2 * swing_lb + 1
    is_h = (d["high"].rolling(w, center=True).max() == d["high"]).to_numpy()
    is_l = (d["low"].rolling(w, center=True).min() == d["low"]).to_numpy()
    is_h[:swing_lb] = is_h[n - swing_lb:] = False
    is_l[:swing_lb] = is_l[n - swing_lb:] = False

    lsh = lsl = np.nan
    armed, sweep_i, expiry = 0, -1, -1
    rows, busy = [], -1
    for i in range(swing_lb + 2, n):
        conf = i - swing_lb
        if is_h[conf]:
            lsh = h[conf]
        if is_l[conf]:
            lsl = l[conf]
        a = atr[i]
        if not np.isfinite(a) or a <= 0 or i < div_look + 2:
            continue
        if not np.isnan(lsl) and l[i] < lsl:
            armed, sweep_i, expiry = 1, i, i + fvg_window
        elif not np.isnan(lsh) and h[i] > lsh:
            armed, sweep_i, expiry = -1, i, i + fvg_window
        if armed != 0 and i > expiry:
            armed = 0
        if armed == 0 or i <= busy:
            continue
        if armed == 1:
            if not (h[i - 2] < l[i]):
                continue
            entry = l[i]
        else:
            if not (l[i - 2] > h[i]):
                continue
            entry = h[i]
        sgn = armed
        stop = entry - sgn * a
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        tgt = entry + sgn * tmult * risk

        # ---- ORDER FLOW at/around the sweep ----
        sb = sweep_i
        win = slice(max(0, sb - div_look), sb + 1)
        if sgn == 1:
            price_new_low = l[sb] <= np.min(l[win])
            cd_new_low = cd36[sb] <= np.nanmin(cd36[win])
            divergence = int(price_new_low and not cd_new_low)   # absorption
        else:
            price_new_hi = h[sb] >= np.max(h[win])
            cd_new_hi = cd36[sb] >= np.nanmax(cd36[win])
            divergence = int(price_new_hi and not cd_new_hi)
        rows.append({
            "ts": d.index[i], "sgn": sgn, "entry": entry, "stop": stop,
            "risk": risk, "target": tgt, "bar": i,
            "delta_sweep": round(float(delta[sb]), 4),
            "delta_signed": round(float(delta[sb] * -sgn), 4),   # + = flow against us at sweep
            "cum_delta_12": float(cd12[sb]),
            "cd12_norm": round(float(cd12[sb] / (d['volume'].iloc[max(0,sb-12):sb+1].sum() + 1e-9)), 4),
            "divergence": divergence,
            "ats_ratio": round(float(ats[sb]), 3) if np.isfinite(ats[sb]) else np.nan,
            "trade_intensity": round(float(ti[sb]), 3) if np.isfinite(ti[sb]) else np.nan,
        })
        armed = 0
        busy = i
    sig = pd.DataFrame(rows)
    if sig.empty:
        return sig

    # ---- outcomes (limit fill, no same-bar exit) ----
    R, keep = [], []
    for _, r in sig.iterrows():
        i = int(r.bar)
        sgn, e, st, tg, rk = int(r.sgn), r.entry, r.stop, r.target, r.risk
        slip = e * STOP_SLIP_BP / 1e4
        filled, res = False, None
        for j in range(i + 1, min(i + 1 + horizon, n)):
            if not filled:
                if l[j] <= e <= h[j]:
                    filled = True
                continue
            if sgn == 1:
                if l[j] <= st:
                    res = (st - slip - e) / rk
                elif h[j] >= tg:
                    res = tmult
            else:
                if h[j] >= st:
                    res = (e - st - slip) / rk
                elif l[j] <= tg:
                    res = tmult
            if res is not None:
                break
        if res is not None:
            R.append(res)
            keep.append(True)
        else:
            keep.append(False)
    sig = sig[keep].copy()
    sig["R"] = R
    sig["win"] = (sig.R > 0).astype(int)
    return sig
