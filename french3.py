"""THREE MORE ON CLEAN DATA -- industry momentum, crash timing, the short side.

Everything here uses French's survivorship-free CRSP portfolios, so none of it
can be contaminated by my ticker list.

  1. INDUSTRY MOMENTUM. Moskowitz & Grinblatt (1999) argued most of individual
     stock momentum is really INDUSTRY momentum -- you are buying whatever
     sector is running, not whatever stock is. French publishes 49 industry
     portfolios. If they are right, rotating industries should capture the
     effect with far fewer positions and far less turnover, which matters
     enormously for a small account.

  2. MOMENTUM CRASH TIMING. Daniel & Moskowitz (2016), "Momentum Crashes".
     Momentum's rare catastrophic losses are PREDICTABLE: they cluster after
     bear markets, when the market rebounds and beaten-down losers rip. The
     proposed fix is to cut momentum exposure when the market is down over the
     prior 2 years AND volatility is high. This is the single best-documented
     improvement to momentum and I have not tested it on clean data.

  3. THE SHORT SIDE. Everything tested here is long-only. French's decile data
     showed losers (D1) returning 1.56%/yr vs winners (D10) at 15.77%. Adding
     a short leg could roughly double the spread -- if it survives costs,
     borrow fees and the crash risk that shorting losers concentrates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from french import fetch, parse_block, stats


def ann(x):
    s = stats(x)
    return s if s else dict(cagr=np.nan, sharpe=np.nan, dd=np.nan, t=np.nan)


if __name__ == "__main__":
    print("fetching clean portfolios from Dartmouth...", flush=True)
    P = parse_block(fetch("10_Portfolios_Prior_12_2_Daily_CSV.zip"))
    P.columns = [f"D{i+1}" for i in range(P.shape[1])]
    FF = parse_block(fetch("F-F_Research_Data_Factors_daily_CSV.zip"))
    mkt = (FF["Mkt-RF"] + FF["RF"]).dropna()

    print("\n" + "=" * 78)
    print("1. INDUSTRY MOMENTUM  (49 clean industry portfolios)")
    print("=" * 78, flush=True)
    IND = parse_block(fetch("49_Industry_Portfolios_daily_CSV.zip"))
    IND = IND.replace(-0.9999, np.nan).dropna(axis=1, how="all")
    print(f"  {IND.shape[1]} industries, {IND.index.min():%Y-%m} -> "
          f"{IND.index.max():%Y-%m}\n")

    def ind_mom(top, look=252, hold="ME"):
        cum = (1 + IND.fillna(0)).cumprod()
        reb = cum.resample(hold).last().index
        sc = cum.pct_change(look)
        out = []
        for i in range(1, len(reb)):
            t0, t1 = reb[i-1], reb[i]
            s = sc.reindex([t0], method="ffill").iloc[0].dropna()
            if len(s) < top:
                continue
            pick = s.nlargest(top).index
            seg = IND.loc[(IND.index > t0) & (IND.index <= t1), list(pick)]
            if seg.empty:
                continue
            out.append(seg.mean(axis=1) - (10/1e4)/len(seg))
        return pd.concat(out).sort_index() if out else pd.Series(dtype=float)

    print(f"  {'strategy':<26}{'CAGR':>9}{'vol':>8}{'maxDD':>8}{'Sharpe':>9}")
    print("  " + "-" * 60)
    b = ann(IND.mean(axis=1))
    print(f"  {'equal-weight industries':<26}{b['cagr']:>8.2f}%{b['vol']:>7.0f}%"
          f"{b['dd']:>7.0f}%{b['sharpe']:>9.2f}")
    for top in (3, 5, 10):
        s = ann(ind_mom(top))
        print(f"  {f'top {top} industries':<26}{s['cagr']:>8.2f}%{s['vol']:>7.0f}%"
              f"{s['dd']:>7.0f}%{s['sharpe']:>9.2f}", flush=True)

    print("\n" + "=" * 78)
    print("2. MOMENTUM CRASH TIMING  (Daniel & Moskowitz)")
    print("=" * 78)
    win, los = "D10", "D1"
    spread = (P[win] - P[los]).dropna()
    m = mkt.reindex(spread.index).fillna(0)
    # bear = market down over the prior 2 years; stress = high recent vol
    bear = ((1 + m).rolling(504).apply(lambda x: x.prod(), raw=True) - 1) < 0
    vol = m.rolling(126).std()*np.sqrt(252)
    stress = vol > vol.rolling(504).median()
    danger = (bear & stress).shift(1).fillna(False)
    print(f"  danger regime active on {danger.mean()*100:.1f}% of days\n")
    print(f"  {'regime':<28}{'W-L CAGR':>11}{'Sharpe':>9}{'t':>8}")
    print("  " + "-" * 58)
    for lab, mask in (("all days", pd.Series(True, index=spread.index)),
                      ("normal", ~danger), ("DANGER (bear + high vol)", danger)):
        s = ann(spread[mask])
        print(f"  {lab:<28}{s['cagr']:>10.2f}%{s['sharpe']:>9.2f}{s['t']:>8.2f}")

    print(f"\n  {'strategy':<34}{'CAGR':>10}{'maxDD':>9}{'Sharpe':>9}")
    print("  " + "-" * 62)
    for lab, ser in (("winners long-only, always", P[win]),
                     ("winners, flat in danger", P[win].where(~danger, 0)),
                     ("W-L spread, always", spread),
                     ("W-L spread, flat in danger", spread.where(~danger, 0))):
        s = ann(ser)
        print(f"  {lab:<34}{s['cagr']:>9.2f}%{s['dd']:>8.0f}%{s['sharpe']:>9.2f}")

    print("\n" + "=" * 78)
    print("3. THE SHORT SIDE -- is it worth adding?")
    print("=" * 78)
    print(f"  {'construction':<30}{'CAGR':>10}{'vol':>8}{'maxDD':>9}{'Sharpe':>9}")
    print("  " + "-" * 66)
    for lab, ser in (("long winners only", P[win]),
                     ("long winners - short losers", spread),
                     ("long winners - short market", P[win] - m),
                     ("130/30 (1.3 long, 0.3 short)", 1.3*P[win] - 0.3*P[los])):
        s = ann(ser)
        print(f"  {lab:<30}{s['cagr']:>9.2f}%{s['vol']:>7.0f}%{s['dd']:>8.0f}%"
              f"{s['sharpe']:>9.2f}")
    print("\n  short leg costs 1-3%/yr in borrow on hard-to-borrow losers,")
    print("  which is NOT deducted above -- so the spread rows are optimistic.")

    print("\n" + "=" * 78)
    print("4. RECENT ERA ONLY  (2010+, what matters for trading now)")
    print("=" * 78)
    print(f"  {'strategy':<34}{'CAGR':>10}{'maxDD':>9}{'Sharpe':>9}")
    print("  " + "-" * 62)
    for lab, ser in (("market", mkt), ("winners long-only", P[win]),
                     ("winners, flat in danger", P[win].where(~danger, 0)),
                     ("top 5 industries", ind_mom(5)),
                     ("W-L spread", spread)):
        s = ann(ser.loc["2010":])
        print(f"  {lab:<34}{s['cagr']:>9.2f}%{s['dd']:>8.0f}%{s['sharpe']:>9.2f}")
