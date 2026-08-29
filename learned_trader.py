"""Learn the SELECTION function — the codifiable part of trader judgment.

A skilled discretionary trader's edge is largely in what they DON'T take.
Rules fire on every setup; the human skips most of them. That skip decision
is the thing we can actually learn, because it leaves a label: the trade
either won or lost.

Setup:
  - every signal from the full-ICT pipeline, all 4 markets, 4.5 years
  - ~25 features: ICT structure, session context, volatility regime,
    distance to key levels, market state
  - target: did the 1R trade win
  - model: gradient boosting

VALIDATION (this is the whole game):
  1. TEMPORAL   train on 2022-2024, test on 2025-2026 (unseen time)
  2. CROSS-MARKET  train on 3 markets, test on the 4th (unseen instrument)
A model that only passes (1) has learned a regime. Passing (2) as well is
what would make it a genuine selection skill rather than memorisation.

Success bar: the model's chosen subset must beat FAIR ODDS (50% at a 1R
target) out-of-sample, by enough to cover ~1% of slippage cost.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

STOP_SLIP = 1.5
MARKETS = {"Nasdaq": "usatechidxusd", "S&P500": "usa500idxusd",
           "Dow30": "usa30idxusd", "DAX": "deuidxeur"}


def swings(df, lb=2):
    w = 2 * lb + 1
    hi = (df["high"].rolling(w, center=True).max() == df["high"]).to_numpy()
    lo = (df["low"].rolling(w, center=True).min() == df["low"]).to_numpy()
    n = len(df)
    hi[:lb] = hi[n - lb:] = False
    lo[:lb] = lo[n - lb:] = False
    return hi, lo


def build_dataset(df: pd.DataFrame, market: str, tmult=1.0,
                  swing_lb=2, window=24) -> pd.DataFrame:
    """Every sweep+FVG signal with rich context + realised outcome."""
    import ict_full as IF
    ctx = IF.htf_context(df)
    h, l, c, o = (ctx[x].to_numpy(float) for x in ("high", "low", "close", "open"))
    v = ctx["volume"].to_numpy(float) if "volume" in ctx else np.zeros(len(ctx))
    n = len(ctx)
    tr = pd.concat([ctx["high"] - ctx["low"],
                    (ctx["high"] - ctx["close"].shift()).abs(),
                    (ctx["low"] - ctx["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().to_numpy()
    atr_avg = pd.Series(atr).rolling(200).mean().to_numpy()
    volavg = pd.Series(v).rolling(50).mean().to_numpy()
    is_h, is_l = swings(ctx, swing_lb)
    mins = np.asarray(ctx.index.hour) * 60 + np.asarray(ctx.index.minute)
    day = ctx.index.normalize()
    day_open = ctx.groupby(day)["open"].transform("first").to_numpy()
    G = {k: ctx[k].to_numpy(float) for k in
         ("pdh", "pdl", "pd_mid", "d_trend", "pd_close_pos",
          "asia_hi", "asia_lo", "lon_hi", "lon_lo")}

    hi_px: list[float] = []
    lo_px: list[float] = []
    rows = []
    armed = 0
    armed_i = -1
    sweep_lvl = np.nan
    mss = 0
    busy = -1

    for i in range(swing_lb + 3, n):
        conf = i - swing_lb
        if is_h[conf]:
            hi_px.append(h[conf])
        if is_l[conf]:
            lo_px.append(l[conf])
        a = atr[i]
        if not np.isfinite(a) or a <= 0 or len(hi_px) < 3 or len(lo_px) < 3:
            continue

        highs = [x for x in (G["pdh"][i], G["asia_hi"][i], G["lon_hi"][i], hi_px[-1]) if np.isfinite(x)]
        lows = [x for x in (G["pdl"][i], G["asia_lo"][i], G["lon_lo"][i], lo_px[-1]) if np.isfinite(x)]
        sl = [x for x in lows if l[i] < x <= l[i] + 3 * a]
        sh = [x for x in highs if h[i] > x >= h[i] - 3 * a]
        if sl:
            armed, sweep_lvl, armed_i, mss = 1, max(sl), i, 0
        elif sh:
            armed, sweep_lvl, armed_i, mss = -1, min(sh), i, 0
        if armed != 0 and i - armed_i > window:
            armed = 0
        if armed == 0 or i <= busy:
            continue
        if not mss:
            if armed == 1 and c[i] > max(hi_px[-2:]):
                mss = 1
            elif armed == -1 and c[i] < min(lo_px[-2:]):
                mss = 1

        if armed == 1:
            if not (h[i - 2] < l[i]):
                continue
            entry = l[i]
            depth = (sweep_lvl - l[armed_i]) / a
            disp = (c[i] - l[armed_i]) / a
        else:
            if not (l[i - 2] > h[i]):
                continue
            entry = h[i]
            depth = (h[armed_i] - sweep_lvl) / a
            disp = (h[armed_i] - c[i]) / a
        sgn = 1 if armed == 1 else -1
        stop = entry - sgn * a
        risk = abs(entry - stop)
        if risk < max(1e-6, 0.0005 * entry):
            continue
        tgt = entry + sgn * tmult * risk

        rng_hi, rng_lo = max(hi_px[-10:]), min(lo_px[-10:])
        pdpos = (entry - rng_lo) / (rng_hi - rng_lo) if rng_hi > rng_lo else 0.5
        sb = armed_i
        bar_rng = max(h[sb] - l[sb], 1e-9)
        rejection = (max(sweep_lvl - l[sb], 0.0) if armed == 1
                     else max(h[sb] - sweep_lvl, 0.0)) / bar_rng

        filled, R, last = False, None, i
        for j in range(i + 1, min(i + 61, n)):
            if not filled:
                if l[j] <= entry <= h[j]:
                    filled, last = True, j
                continue
            if sgn == 1:
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
        armed = 0
        if R is None:
            continue

        rows.append({
            "market": market, "ts": ctx.index[i], "R": R, "win": int(R > 0),
            "side": sgn,
            "depth_atr": depth, "rejection": rejection, "disp_atr": disp,
            "mss": mss, "pd_pos": pdpos,
            "bars_since_sweep": i - armed_i,
            "tod_min": mins[i], "dow": ctx.index[i].dayofweek,
            "atr_regime": a / atr_avg[i] if np.isfinite(atr_avg[i]) and atr_avg[i] > 0 else np.nan,
            "risk_atr": risk / a,
            "d_trend_align": (G["d_trend"][i] * sgn) if np.isfinite(G["d_trend"][i]) else 0.0,
            "pd_close_pos": G["pd_close_pos"][i],
            "dist_pdh": (entry - G["pdh"][i]) / a if np.isfinite(G["pdh"][i]) else np.nan,
            "dist_pdl": (entry - G["pdl"][i]) / a if np.isfinite(G["pdl"][i]) else np.nan,
            "dist_dayopen": (entry - day_open[i]) / a,
            "vol_ratio": v[i] / volavg[i] if np.isfinite(volavg[i]) and volavg[i] > 0 else np.nan,
            "swept_pdh_pdl": int(np.isfinite(G["pdh"][i]) and
                                 (abs(sweep_lvl - G["pdh"][i]) < 0.1 * a or
                                  abs(sweep_lvl - G["pdl"][i]) < 0.1 * a)),
            "swept_asia": int(np.isfinite(G["asia_hi"][i]) and
                              (abs(sweep_lvl - G["asia_hi"][i]) < 0.1 * a or
                               abs(sweep_lvl - G["asia_lo"][i]) < 0.1 * a)),
        })
    return pd.DataFrame(rows)


FEATURES = ["side", "depth_atr", "rejection", "disp_atr", "mss", "pd_pos",
            "bars_since_sweep", "tod_min", "dow", "atr_regime", "risk_atr",
            "d_trend_align", "pd_close_pos", "dist_pdh", "dist_pdl",
            "dist_dayopen", "vol_ratio", "swept_pdh_pdl", "swept_asia"]


def evaluate(train: pd.DataFrame, test: pd.DataFrame, label: str,
             thresholds=(0.50, 0.55, 0.60)):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                       learning_rate=0.05,
                                       min_samples_leaf=60, random_state=0)
    m.fit(train[FEATURES].values, train["win"].values)
    p = m.predict_proba(test[FEATURES].values)[:, 1]
    base = test["win"].mean() * 100
    out = [f"  {label}: baseline {base:.1f}% win ({len(test)} trades), expR {test.R.mean():+.3f}"]
    for th in thresholds:
        sel = test[p >= th]
        if len(sel) < 25:
            out.append(f"     p>={th:.2f}: only {len(sel)} trades (too few)")
            continue
        se = sel.R.std(ddof=1) / np.sqrt(len(sel))
        out.append(f"     p>={th:.2f}: n={len(sel):>4}  win {sel.win.mean()*100:>5.1f}%  "
                   f"expR {sel.R.mean():+.3f}  t={sel.R.mean()/se:>+5.2f}")
    return "\n".join(out)
