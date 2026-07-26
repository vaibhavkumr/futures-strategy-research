"""Data layer: load real OHLCV candles from CSV, or generate synthetic ones
so the whole pipeline runs with zero cost / zero data account.

Real data: export 1-minute or 5-minute NQ/ES candles to a CSV with columns
    timestamp,open,high,low,close,volume
(timestamp in UTC or with tz offset). Databento, your broker, or TradingView
export all work. Then: load_csv("nq_1m.csv").
"""
from __future__ import annotations
import io
import os
import time
import json
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd


def load_csv(path: str, tz: str = "America/New_York") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index(ts.dt.tz_convert(tz)).sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_yahoo(symbol: str = "NQ=F", interval: str = "5m", period: str = "60d",
                tz: str = "America/New_York") -> pd.DataFrame:
    """Pull real futures candles from Yahoo (free). Limits: 5m/15m <=60d,
    1h <=730d, 1d full history."""
    import yfinance as yf
    df = yf.download(symbol, interval=interval, period=period,
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    idx = pd.to_datetime(df.index, utc=True).tz_convert(tz)
    df = df.set_index(idx).sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _months(start: str, end: str) -> list[str]:
    """List 'YYYY-MM' strings from start to end inclusive."""
    s = pd.Period(start, "M")
    e = pd.Period(end, "M")
    return [str(p) for p in pd.period_range(s, e, freq="M")]


def fetch_alphavantage(symbol: str = "QQQ", interval: str = "5min",
                       start: str = "2023-01", end: str | None = None,
                       api_key: str | None = None,
                       cache_dir: str = "av_cache",
                       tz: str = "America/New_York",
                       pause: float = 1.0) -> pd.DataFrame:
    """Pull multi-year intraday bars from Alpha Vantage, one month per call,
    caching each month to CSV so re-runs are free and resumable.

    QQQ (Nasdaq-100 ETF) is a free proxy for NQ futures. Get a free key at
    alphavantage.co/support/#api-key. Free tier ~25 calls/day: if you hit the
    limit this stops cleanly and you resume tomorrow (cached months are kept).
    """
    api_key = api_key or os.environ.get("ALPHAVANTAGE_KEY")
    if not api_key:
        raise ValueError("No API key. Pass api_key= or set ALPHAVANTAGE_KEY.")
    end = end or pd.Timestamp.today().strftime("%Y-%m")
    os.makedirs(cache_dir, exist_ok=True)
    frames = []
    for mo in _months(start, end):
        fp = os.path.join(cache_dir, f"{symbol}_{interval}_{mo}.csv")
        if os.path.exists(fp):
            frames.append(pd.read_csv(fp))
            continue
        params = urllib.parse.urlencode({
            "function": "TIME_SERIES_INTRADAY", "symbol": symbol,
            "interval": interval, "month": mo, "outputsize": "full",
            "datatype": "csv", "apikey": api_key, "extended_hours": "false"})
        url = f"https://www.alphavantage.co/query?{params}"
        raw = urllib.request.urlopen(url, timeout=60).read().decode()
        if not raw.lstrip().startswith("timestamp"):
            # JSON note = rate limit / error. Stop cleanly, keep what we cached.
            try:
                msg = json.loads(raw)
            except Exception:
                msg = {"raw": raw[:200]}
            print(f"[stopped at {mo}] {msg}. Cached {len(frames)} months so far "
                  f"-- rerun later to resume.")
            break
        m = pd.read_csv(io.StringIO(raw))
        m.to_csv(fp, index=False)
        frames.append(m)
        print(f"fetched {symbol} {mo} ({len(m)} bars)")
        time.sleep(pause)
    if not frames:
        raise RuntimeError("No data fetched. Check key / rate limit.")
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    idx = pd.to_datetime(df["timestamp"]).dt.tz_localize(tz)
    df = df.set_index(idx).sort_index()
    df = df[~df.index.duplicated()]
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def synthetic(days: int = 20, freq_min: int = 5, seed: int = 42,
              tz: str = "America/New_York") -> pd.DataFrame:
    """Generate believable intraday candles (trending + noisy) for testing.
    NOT a market model — just enough structure/gaps to exercise the strategy.
    """
    rng = np.random.default_rng(seed)
    rows = []
    price = 20000.0  # roughly NQ-ish
    start = pd.Timestamp("2025-01-06 00:00", tz=tz)  # a Monday
    for d in range(days):
        day = start + pd.Timedelta(days=d)
        if day.weekday() >= 5:  # skip weekends
            continue
        # Regular US session 09:30-16:00 ET
        session_open = day + pd.Timedelta(hours=9, minutes=30)
        n = int((6.5 * 60) / freq_min)
        drift = rng.normal(0, 1.2)  # per-day trend bias
        for i in range(n):
            t = session_open + pd.Timedelta(minutes=i * freq_min)
            step = rng.normal(drift, 8.0)
            o = price
            c = o + step
            hi = max(o, c) + abs(rng.normal(0, 4))
            lo = min(o, c) - abs(rng.normal(0, 4))
            vol = abs(rng.normal(1000, 300))
            rows.append((t, o, hi, lo, c, vol))
            price = c
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low",
                                     "close", "volume"]).set_index("timestamp")
    return df


if __name__ == "__main__":
    df = synthetic()
    print(df.head())
    print(f"\n{len(df)} candles, {df.index[0]} -> {df.index[-1]}")
