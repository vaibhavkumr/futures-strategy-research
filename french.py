"""CLEAN DATA, FREE -- Kenneth French's Data Library.

I said a survivorship-free universe needed CRSP or Norgate, both paid. That
was wrong. Ken French publishes portfolio and factor returns BUILT on CRSP at
Dartmouth, free and without an API key, daily back to 1926. Every delisted,
bankrupt and acquired company is in there, because CRSP never drops a name.

This settles the question my 202-ticker list could not:

  1. IS MOMENTUM REAL on data with no survivorship bias? French's 10
     portfolios formed on prior 2-12 month returns give the answer directly.
     Decile 10 is the winners, decile 1 the losers.

  2. HOW BIG is the effect, honestly measured, over a century rather than
     the 16 years my free prices covered.

  3. DOES IT STILL WORK RECENTLY? Momentum is widely reported to have decayed
     since publication in 1993. Splitting pre/post-publication tests that
     directly, and it is the number that matters for trading it now.

  4. WHAT WOULD IT PAY after realistic costs? The portfolios are gross, so
     turnover costs have to be subtracted rather than assumed away.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import requests

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
HDR = {"User-Agent": "Mozilla/5.0 (research)"}


def fetch(name):
    """Download and parse one of French's CSV zips."""
    url = BASE + name
    r = requests.get(url, headers=HDR, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    fn = z.namelist()[0]
    raw = z.read(fn).decode("latin-1")
    return raw


def parse_block(raw, min_cols=3):
    """French files stack several tables with text headers. Take the first
    numeric block, which is the value-weighted average returns."""
    rows = []
    started = False
    for line in raw.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < min_cols:
            if started:
                break
            continue
        head = parts[0]
        if not head or not head.replace("-", "").isdigit():
            if started:
                break
            # this is the column-header line
            cols = parts[1:]
            continue
        started = True
        try:
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            break
        rows.append([head] + vals)
    if not rows:
        return None
    n = len(rows[0]) - 1
    df = pd.DataFrame(rows, columns=["date"] + list(cols[:n]))
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.replace([-99.99, -999], np.nan) / 100.0


def stats(x, ann=252):
    x = pd.Series(x).dropna()
    if len(x) < 200:
        return None
    eq = (1 + x).cumprod()
    yrs = len(x) / ann
    cagr = (eq.iloc[-1] ** (1 / yrs) - 1) * 100
    dd = (eq / eq.cummax() - 1).min() * 100
    return dict(cagr=cagr, dd=dd, vol=x.std(ddof=1) * np.sqrt(ann) * 100,
                sharpe=x.mean() / x.std(ddof=1) * np.sqrt(ann),
                t=x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


if __name__ == "__main__":
    print("fetching survivorship-free momentum portfolios from Dartmouth...",
          flush=True)
    raw = fetch("10_Portfolios_Prior_12_2_Daily_CSV.zip")
    P = parse_block(raw)
    if P is None:
        raise SystemExit("could not parse")
    P.columns = [f"D{i+1}" for i in range(P.shape[1])]
    print(f"got {P.shape[0]:,} daily rows, {P.shape[1]} deciles, "
          f"{P.index.min():%Y-%m} -> {P.index.max():%Y-%m}\n")

    print("=" * 80)
    print("1. MOMENTUM DECILES ON CLEAN DATA  (D1 = losers, D10 = winners)")
    print("=" * 80)
    print(f"  {'decile':<10}{'CAGR':>10}{'vol':>8}{'maxDD':>8}{'Sharpe':>9}{'t-stat':>9}")
    print("  " + "-" * 56)
    for c in P.columns:
        s = stats(P[c])
        if s:
            print(f"  {c:<10}{s['cagr']:>9.2f}%{s['vol']:>7.0f}%{s['dd']:>7.0f}%"
                  f"{s['sharpe']:>9.2f}{s['t']:>9.2f}")

    win, los = P.columns[-1], P.columns[0]
    spread = P[win] - P[los]
    s = stats(spread)
    print(f"\n  WINNERS minus LOSERS: {s['cagr']:.2f}%/yr  Sharpe {s['sharpe']:.2f}"
          f"  t={s['t']:.2f}")

    print("\n" + "=" * 80)
    print("2. HAS IT DECAYED SINCE PUBLICATION (Jegadeesh & Titman, 1993)?")
    print("=" * 80)
    print(f"  {'period':<24}{'D10 CAGR':>11}{'W-L spread':>13}{'Sharpe':>9}{'t':>8}")
    print("  " + "-" * 66)
    for lab, sl in (("1927-1993 (pre-pub)", slice(None, "1993-12-31")),
                    ("1994-2009", slice("1994-01-01", "2009-12-31")),
                    ("2010-2026 (my window)", slice("2010-01-01", None))):
        a, b = stats(P[win].loc[sl]), stats(spread.loc[sl])
        if a and b:
            print(f"  {lab:<24}{a['cagr']:>10.2f}%{b['cagr']:>12.2f}%"
                  f"{b['sharpe']:>9.2f}{b['t']:>8.2f}")

    print("\n" + "=" * 80)
    print("3. LONG-ONLY WINNERS vs THE MARKET  (what I would actually trade)")
    print("=" * 80)
    raw3 = fetch("F-F_Research_Data_Factors_daily_CSV.zip")
    FF = parse_block(raw3)
    mkt = (FF["Mkt-RF"] + FF["RF"]).dropna()
    print(f"  {'period':<24}{'D10':>10}{'market':>10}{'excess':>10}"
          f"{'D10 Sharpe':>12}")
    print("  " + "-" * 66)
    for lab, sl in (("full sample", slice(None)),
                    ("1994-2009", slice("1994-01-01", "2009-12-31")),
                    ("2010-2026", slice("2010-01-01", None))):
        a = stats(P[win].loc[sl])
        j = P[win].loc[sl].index.intersection(mkt.index)
        b = stats(mkt.loc[j])
        if a and b:
            print(f"  {lab:<24}{a['cagr']:>9.2f}%{b['cagr']:>9.2f}%"
                  f"{a['cagr']-b['cagr']:>+9.2f}%{a['sharpe']:>12.2f}")

    print("\n" + "=" * 80)
    print("4. AFTER COSTS -- decile portfolios rebalance monthly, ~50% turnover")
    print("=" * 80)
    print(f"  {'cost/turnover':<24}{'D10 net':>11}{'vs market':>12}")
    print("  " + "-" * 50)
    j = P[win].loc["2010":].index.intersection(mkt.index)
    mkt_c = stats(mkt.loc[j])["cagr"]
    for bp in (0, 10, 20, 50):
        drag = 0.50 * bp / 1e4 * 12          # 50% turnover, monthly
        net = P[win].loc["2010":] - drag / 252
        s2 = stats(net)
        print(f"  {bp:>3}bp round trip{'':<8}{s2['cagr']:>10.2f}%"
              f"{s2['cagr']-mkt_c:>+11.2f}%")
