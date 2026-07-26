"""THE BOT — validated momentum system, sized for a real account.

What survived the gauntlet (18 signals + a walk-forward ML model tested):
    12-month cross-sectional momentum, long-only, top 6 of 26 assets,
    rebalanced weekly.
    Holdout 2018-2026: Sharpe 0.87, CAGR 18.0%, max drawdown 24.6%
    (SPY over the same period: 0.66 / 14.4% / 33.7%)

Everything else -- reversal, volatility, volume, seasonality, macro regime,
and gradient boosting -- failed out-of-sample and is NOT in here. That is
deliberate. See signal_screen.csv.

Risk controls:
  - equal weight across N names (concentration is measured risk, not edge:
    top-3 raised drawdown to 34% for no extra return)
  - drawdown circuit breaker: de-risk if the account falls past a threshold
  - no leverage by default. Leverage multiplies drawdown 1:1 -- at 3x, the
    24.6% historical drawdown becomes ~74%.

Run it weekly. It prints the target book and the trades to get there.
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG",
            "GLD", "SLV", "DBC", "USO", "XLE", "XLF", "XLK", "XLV", "XLU",
            "XLP", "XLY", "XLI", "XLB", "EWJ", "FXI", "VNQ", "BTC-USD"]
LOOKBACK = 252
TOP_N = 6
STATE = "bot_state.json"
MAX_DD_DERISK = 0.20      # if account is >20% below its peak, halve exposure


def fetch(days: int = 500) -> pd.DataFrame:
    import yfinance as yf
    start = (pd.Timestamp.today() - pd.Timedelta(days=days * 1.6)).strftime("%Y-%m-%d")
    px = yf.download(UNIVERSE, start=start, interval="1d",
                     progress=False, auto_adjust=True)["Close"]
    return px.dropna(how="all").ffill()


def target_book(px: pd.DataFrame, equity: float, top_n: int = TOP_N):
    """Rank by trailing 12m return; hold the top N equally weighted."""
    mom = px.pct_change(LOOKBACK, fill_method=None).iloc[-1].dropna()
    ranked = mom.sort_values(ascending=False)
    picks = ranked.head(top_n)
    last = px.iloc[-1]
    per = equity / top_n
    rows = []
    for tkr, m in picks.items():
        p = float(last[tkr])
        rows.append({"ticker": tkr, "mom_12m": f"{m*100:+.1f}%",
                     "price": round(p, 2), "target_$": round(per, 2),
                     "shares": round(per / p, 4)})
    return pd.DataFrame(rows), ranked


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"peak_equity": None, "holdings": {}}


def save_state(s):
    with open(STATE, "w") as f:
        json.dump(s, f, indent=2)


def main(equity: float = 5000.0):
    px = fetch()
    st = load_state()
    peak = max(st.get("peak_equity") or equity, equity)
    dd = 1 - equity / peak
    scale = 0.5 if dd > MAX_DD_DERISK else 1.0

    print("=" * 66)
    print(f"  MOMENTUM BOT   equity ${equity:,.2f}   data through {px.index[-1].date()}")
    if scale < 1:
        print(f"  !! drawdown {dd*100:.1f}% > {MAX_DD_DERISK*100:.0f}% "
              f"-> DE-RISKING to {scale*100:.0f}% exposure")
    print("=" * 66)

    book, ranked = target_book(px, equity * scale)
    print("\nTARGET BOOK (equal weight, top %d by 12-month momentum):" % TOP_N)
    print(book.to_string(index=False))
    if scale < 1:
        print(f"\n  holding ${equity*(1-scale):,.0f} in cash (de-risked)")

    old = st.get("holdings", {})
    new = {r["ticker"]: r["shares"] for _, r in book.iterrows()}
    buys = [t for t in new if t not in old]
    sells = [t for t in old if t not in new]
    if old:
        print("\nCHANGES SINCE LAST RUN:")
        print(f"  BUY : {', '.join(buys) if buys else '(none)'}")
        print(f"  SELL: {', '.join(sells) if sells else '(none)'}")
        print(f"  HOLD: {', '.join(t for t in new if t in old) or '(none)'}")
    else:
        print("\n(first run — no prior book to compare)")

    print("\nFULL RANKING (what the bot is choosing between):")
    for i, (t, m) in enumerate(ranked.items(), 1):
        mark = " <-- HELD" if t in new else ""
        print(f"  {i:>2}. {t:<9} {m*100:>+7.1f}%{mark}")

    st["peak_equity"] = peak
    st["holdings"] = new
    st["last_run"] = str(px.index[-1].date())
    save_state(st)
    print(f"\nstate saved -> {STATE}")
    print("\nREMINDER: historical max drawdown was 24.6%. On $5,000 that is a")
    print("$1,230 loss at some point. That is normal, not a malfunction.")


if __name__ == "__main__":
    import sys
    eq = float(sys.argv[1]) if len(sys.argv) > 1 else 5000.0
    main(eq)
