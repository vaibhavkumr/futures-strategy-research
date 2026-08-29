"""Two information sources we have never used: VOLUME and CROSS-MARKET.

Everything tested so far came from NQ's own OHLC bars -- the most heavily
processed data that exists, which is why the strategy lands exactly on
fair-coin odds. To beat fair odds you need an input that is not already
fully in that price series.

  vol_ratio    volume on the sweep bar / its 50-bar average. A stop run on
               heavy volume is a different event from a drift on thin volume.
  es_diverge   for a LONG (NQ swept a low): did ES make a new low too, or
               did it hold? NQ weak while ES holds = relative strength =
               classic bullish divergence. Mirror for shorts.
  es_lead      ES return over the prior 3 bars. ES is the deeper book and
               sometimes turns first.

Judged on the newest 30% of trades, which is not looked at while forming
the hypothesis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from live_paper import STOP_SLIP
import replay as RP


def load():
    import yfinance as yf
    d = yf.download(["NQ=F", "ES=F"], period="60d", interval="5m",
                    progress=False, auto_adjust=False)
    px = {}
    for f in ("Open", "High", "Low", "Close", "Volume"):
        px[f] = d[f]
    idx = pd.to_datetime(d.index, utc=True).tz_convert("America/New_York")
    nq = pd.DataFrame({k.lower(): px[k]["NQ=F"] for k in px}).set_index(idx).sort_index()
    es = pd.DataFrame({k.lower(): px[k]["ES=F"] for k in px}).set_index(idx).sort_index()
    keep = nq.dropna(subset=["high", "low", "close"]).index.intersection(
        es.dropna(subset=["high", "low", "close"]).index)
    return nq.loc[keep], es.loc[keep]


def build(nq: pd.DataFrame, es: pd.DataFrame, tmult=0.5, look=20):
    h, l, c, v = (nq[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    eh, el, ec = (es[x].to_numpy(float) for x in ("high", "low", "close"))
    n = len(nq)
    tr = pd.concat([nq["high"] - nq["low"],
                    (nq["high"] - nq["close"].shift()).abs(),
                    (nq["low"] - nq["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().to_numpy()
    volavg = pd.Series(v).rolling(50).mean().to_numpy()

    rows, busy = [], -1
    for (i, side, entry, sweep_stop, sweep_risk) in RP.gen_signals(nq):
        if i <= busy:
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0 or i < look + 2:
            continue
        sgn = 1 if side == "long" else -1
        stop = entry - sgn * a                  # ATR stop (best structure found)
        risk = abs(entry - stop)
        if risk < max(1e-6, 0.0005 * entry):
            continue
        tgt = entry + sgn * tmult * risk

        # ---- the new information ----
        vr = v[i] / volavg[i] if np.isfinite(volavg[i]) and volavg[i] > 0 else np.nan
        w = slice(i - look, i + 1)
        if side == "long":
            nq_new_low = l[i] <= np.min(l[w])
            es_new_low = el[i] <= np.min(el[w])
            diverge = int(nq_new_low and not es_new_low)   # NQ swept, ES held
        else:
            nq_new_hi = h[i] >= np.max(h[w])
            es_new_hi = eh[i] >= np.max(eh[w])
            diverge = int(nq_new_hi and not es_new_hi)
        es_lead = (ec[i] / ec[i - 3] - 1) * sgn * 1e4      # bp, signed with trade

        filled, R, last = False, None, i
        for j in range(i + 1, min(i + 61, n)):
            if not filled:
                if l[j] <= entry <= h[j]:
                    filled, last = True, j
                continue
            if side == "long":
                if l[j] <= stop:
                    R = (stop - STOP_SLIP - entry) / risk
                elif h[j] >= tgt:
                    R = tmult
            else:
                if h[j] >= stop:
                    R = (entry - stop - STOP_SLIP) / risk
                elif l[j] <= tgt:
                    R = tmult
            if R is not None:
                last = j
                break
        busy = last
        if R is None:
            continue
        rows.append({"ts": nq.index[i], "side": side, "R": R,
                     "vol_ratio": round(vr, 3) if np.isfinite(vr) else np.nan,
                     "es_diverge": diverge, "es_lead_bp": round(es_lead, 2)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    nq, es = load()
    print(f"aligned bars: {len(nq)}  {nq.index[0]:%Y-%m-%d} -> {nq.index[-1]:%Y-%m-%d}")
    for tm in (0.5, 1.0):
        t = build(nq, es, tmult=tm).dropna(subset=["R"])
        if len(t) < 40:
            continue
        t = t.sort_values("ts").reset_index(drop=True)
        cut = int(len(t) * 0.7)
        tr_, te = t[:cut], t[cut:]
        be = 100 / (1 + tm)
        print(f"\n=== target {tm}R  (fair-coin win rate {be:.1f}%) ===")
        print(f"  all trades: n={len(t)} win {(t.R>0).mean()*100:.1f}% expR {t.R.mean():+.3f}")

        def show(name, f):
            a, b = f(tr_), f(te)
            if len(a) < 15 or len(b) < 8:
                print(f"  {name:<26} (too few)")
                return
            print(f"  {name:<26} TRAIN win {(a.R>0).mean()*100:>5.1f}% expR {a.R.mean():+.3f} n={len(a):<4}"
                  f" TEST win {(b.R>0).mean()*100:>5.1f}% expR {b.R.mean():+.3f} n={len(b)}")

        show("baseline", lambda d: d)
        show("volume > 1.2x avg", lambda d: d[d.vol_ratio > 1.2])
        show("volume > 1.5x avg", lambda d: d[d.vol_ratio > 1.5])
        show("volume < 0.8x avg", lambda d: d[d.vol_ratio < 0.8])
        show("ES divergence", lambda d: d[d.es_diverge == 1])
        show("no ES divergence", lambda d: d[d.es_diverge == 0])
        show("ES leading our way", lambda d: d[d.es_lead_bp > 0])
        show("vol>1.2 AND ES diverge", lambda d: d[(d.vol_ratio > 1.2) & (d.es_diverge == 1)])
