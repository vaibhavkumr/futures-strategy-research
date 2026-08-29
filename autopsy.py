"""Autopsy: why did the losing trades lose?

Captures per-trade diagnostics, especially:
  mfe_R   max FAVOURABLE excursion before the trade resolved. If losers ran
          our way first and then reversed, the problem is exit placement.
          If they went straight against us, the problem is entry timing.
  mae_R   max ADVERSE excursion. How close did winners come to stopping out?
  bars_to_fill / filled -- are we chasing orders that never fill?

TRAIN/TEST: patterns are found on the OLDEST 70% of trades and judged on the
newest 30%, which is never inspected while forming a hypothesis. Without that
split this whole exercise just manufactures a curve fit.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from live_paper import TARGETS, SWING_LB, FVG_WINDOW, STOP_SLIP, in_session
import replay as RP


def load_nq(days=60):
    import yfinance as yf
    raw = yf.download("NQ=F", period="60d", interval="5m",
                      progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    idx = pd.to_datetime(raw.index, utc=True).tz_convert("America/New_York")
    df = raw.set_index(idx).sort_index()[["open", "high", "low", "close"]].astype(float)
    return df[df.index >= df.index[-1] - pd.Timedelta(days=days)]


def diagnose(df: pd.DataFrame, target_mult=0.5, max_bars=60):
    h, l, c, o = (df[x].to_numpy(float) for x in ("high", "low", "close", "open"))
    n = len(df)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().to_numpy()
    day = df.index.normalize()
    day_open = df.groupby(day)["open"].transform("first").to_numpy()

    rows = []
    busy = -1
    for (i, side, entry, stop, risk) in RP.gen_signals(df):
        if i <= busy:
            continue
        sgn = 1 if side == "long" else -1
        target = entry + sgn * target_mult * risk
        filled = False
        fill_bar = None
        R = np.nan
        mfe = mae = 0.0
        outcome = "no_fill"
        for j in range(i + 1, min(i + 1 + max_bars, n)):
            if not filled:
                if l[j] <= entry <= h[j]:
                    filled, fill_bar = True, j
                continue                      # no same-bar exit
            fav = (h[j] - entry) if side == "long" else (entry - l[j])
            adv = (entry - l[j]) if side == "long" else (h[j] - entry)
            mfe = max(mfe, fav / risk)
            mae = max(mae, adv / risk)
            if side == "long":
                if l[j] <= stop:
                    R, outcome = (stop - STOP_SLIP - entry) / risk, "stop"
                elif h[j] >= target:
                    R, outcome = target_mult, "target"
            else:
                if h[j] >= stop:
                    R, outcome = (entry - stop - STOP_SLIP) / risk, "stop"
                elif l[j] <= target:
                    R, outcome = target_mult, "target"
            if outcome in ("stop", "target"):
                break
        else:
            if filled:
                outcome = "timeout"
        ts = df.index[i]
        a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else np.nan
        rows.append({
            "ts": ts, "side": side, "outcome": outcome, "R": R,
            "mfe_R": round(mfe, 3), "mae_R": round(mae, 3),
            "filled": filled,
            "bars_to_fill": (fill_bar - i) if fill_bar else np.nan,
            "risk_pts": round(risk, 2),
            "risk_atr": round(risk / a, 3) if a else np.nan,
            "tod_min": ts.hour * 60 + ts.minute,
            "session": ("london" if ts.hour < 6 else
                        "ny_am" if ts.hour < 12 else "ny_pm"),
            "dow": ts.dayofweek,
            "above_open": int(c[i] > day_open[i]),
        })
        busy = fill_bar if fill_bar else i
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_nq(60)
    t = diagnose(df)
    t.to_csv("autopsy_trades.csv", index=False)
    res = t.dropna(subset=["R"])
    print(f"{len(t)} signals | filled {t.filled.sum()} | resolved {len(res)}")
    print(f"outcomes: {t.outcome.value_counts().to_dict()}")
    print(f"expectancy {res.R.mean():+.3f}R  win {(res.R>0).mean()*100:.0f}%")
