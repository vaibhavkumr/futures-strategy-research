"""Live signal for the spec-built TJR bot.

Needs 1m + 5m NQ and 5m ES (for SMT). Returns the newest actionable signal
so live_paper.py can drive it with existing risk controls and accounting.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import tjr_spec as S


def _yf(sym, period, interval):
    import yfinance as yf
    d = yf.download(sym, period=period, interval=interval,
                    progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.columns = [c.lower() for c in d.columns]
    idx = pd.to_datetime(d.index, utc=True).tz_convert("America/New_York")
    return d.set_index(idx).sort_index()[["open", "high", "low", "close"]].astype(float)


def find_signal_spec(df5: pd.DataFrame):
    """df5 comes from live_paper.fetch(); we pull 1m and ES ourselves."""
    try:
        df1 = _yf("NQ=F", "5d", "1m")
        es = _yf("ES=F", "10d", "5m")
    except Exception:
        return None
    if len(df1) < 200 or len(df5) < 100:
        return None
    lo = max(df1.index[0], df5.index[0])
    d5 = df5[df5.index >= lo]
    d1 = df1[df1.index >= lo]
    es = es.reindex(d5.index, method="ffill")
    t = S.generate(d5, d1, corr5=es)
    if t.empty:
        return None
    last = t.iloc[-1]
    ts = pd.Timestamp(last.ts)
    # rebuild entry/stop from the bar the signal fired on
    row = d1[d1.index <= ts]
    if row.empty:
        return None
    entry = float(row["close"].iloc[-1])
    risk = float(last.risk_atr) * float(
        (d5["high"] - d5["low"]).rolling(14).mean().iloc[-1])
    if risk <= 0:
        return None
    sgn = 1 if last.side == "long" else -1
    stop = entry - sgn * risk
    return (ts, last.side, entry, stop, risk)
