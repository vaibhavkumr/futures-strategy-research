"""SELLING OPTIONS -- the high-win-rate strategy I never tested.

Every directional method tested here lands at 46-55% win rate, and the user's
description of profitable daily traders keeps being "they win more than they
lose". Option SELLERS genuinely win 70-90% of trades. That profile is real and
it comes from a documented premium:

  VARIANCE RISK PREMIUM. Implied volatility systematically exceeds subsequent
  realised volatility, because buyers pay up for insurance. The seller earns
  the difference. Bollerslev/Tauchen/Zhou; one of the most robust anomalies
  in the literature and NOT arbitraged away, because bearing tail risk is a
  genuine service.

The catch, and the reason it is not free money: the payoff is inverted. Many
small wins, rare enormous losses. A strategy can win 90% of days and still
lose everything, so win rate is the wrong statistic and GROWTH is the right one.

Measured, using free data only:
  1. IS THE PREMIUM REAL? VIX (implied) vs subsequent 21-day realised vol.
  2. SHORT PUT on SPY, sized realistically, priced with Black-Scholes at
     the prevailing VIX, settled on actual outcomes.
  3. WIN RATE vs GROWTH -- the whole point.
  4. TAIL. What 2008, 2020, 2018-Feb actually did to the seller.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


def load():
    spy = yf.download("SPY", start="2005-01-01", progress=False,
                      auto_adjust=True)["Close"]
    vix = yf.download("^VIX", start="2005-01-01", progress=False,
                      auto_adjust=True)["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    df = pd.DataFrame({"px": spy, "vix": vix}).dropna()
    return df


def bs_put(S, K, T, sigma, r=0.03):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def stats(x, ann=252):
    x = pd.Series(x).dropna()
    if len(x) < 100:
        return None
    x = np.clip(x, -0.99, None)
    eq = (1+x).cumprod()
    yrs = len(x)/ann
    return dict(growth=(np.exp(np.log1p(x).mean()*ann)-1)*100,
                cagr=(eq.iloc[-1]**(1/yrs)-1)*100,
                dd=(eq/eq.cummax()-1).min()*100,
                vol=x.std(ddof=1)*np.sqrt(ann)*100,
                sharpe=x.mean()/x.std(ddof=1)*np.sqrt(ann),
                win=(x > 0).mean()*100)


if __name__ == "__main__":
    d = load()
    print(f"{len(d):,} days, {d.index.min():%Y-%m} -> {d.index.max():%Y-%m}\n")

    print("=" * 78)
    print("1. IS THE VARIANCE RISK PREMIUM REAL?")
    print("=" * 78)
    ret = d.px.pct_change()
    fwd_rv = ret.rolling(21).std().shift(-21)*np.sqrt(252)*100
    iv = d.vix
    cmp = pd.DataFrame({"iv": iv, "rv": fwd_rv}).dropna()
    prem = cmp.iv - cmp.rv
    print(f"  mean implied vol   {cmp.iv.mean():.2f}%")
    print(f"  mean realised vol  {cmp.rv.mean():.2f}%   (next 21 days)")
    print(f"  PREMIUM            {prem.mean():+.2f} vol points   "
          f"t = {prem.mean()/(prem.std(ddof=1)/np.sqrt(len(prem))):.1f}")
    print(f"  implied > realised on {(prem > 0).mean()*100:.1f}% of days")
    print("\n  -> the premium is real and large. Sellers are paid to bear tail risk.")

    print("\n" + "=" * 78)
    print("2. SHORT PUT ON SPY -- monthly, held to expiry")
    print("=" * 78)
    print("  sized so max loss at a -20% crash is survivable; priced at VIX\n")
    print(f"  {'moneyness':<14}{'win rate':>11}{'CAGR':>10}{'GROWTH':>10}"
          f"{'maxDD':>9}{'Sharpe':>9}")
    print("  " + "-" * 64)
    results = {}
    for otm, lab in ((0.00, "ATM"), (0.02, "2% OTM"), (0.05, "5% OTM"),
                     (0.10, "10% OTM")):
        rows = []
        T = 21/252
        idx = d.index
        i = 0
        while i < len(d) - 22:
            S0 = d.px.iloc[i]
            K = S0*(1-otm)
            sig = d.vix.iloc[i]/100
            prem_rec = bs_put(S0, K, T, sig)
            S1 = d.px.iloc[i+21]
            payoff = max(K - S1, 0.0)
            # return on capital = notional K (cash-secured)
            rows.append(dict(ts=idx[i+21], r=(prem_rec - payoff)/K))
            i += 21
        s = pd.DataFrame(rows).set_index("ts").r
        st = stats(s, ann=12)
        results[lab] = s
        if st:
            print(f"  {lab:<14}{(s>0).mean()*100:>10.1f}%{st['cagr']:>9.2f}%"
                  f"{st['growth']:>9.2f}%{st['dd']:>8.0f}%{st['sharpe']:>9.2f}")

    print("\n" + "=" * 78)
    print("3. WIN RATE vs GROWTH -- why win rate is the wrong statistic")
    print("=" * 78)
    s = results["5% OTM"]
    print(f"  5% OTM short put: wins {(s>0).mean()*100:.1f}% of months")
    print(f"    average win  {s[s>0].mean()*100:+.2f}%")
    print(f"    average loss {s[s<0].mean()*100:+.2f}%")
    print(f"    ratio        {abs(s[s<0].mean()/s[s>0].mean()):.1f}x bigger losses")
    print(f"    worst month  {s.min()*100:+.2f}%")
    st = stats(s, ann=12)
    print(f"\n  -> {(s>0).mean()*100:.0f}% win rate, and the growth rate is "
          f"{st['growth']:.2f}%/yr")

    print("\n" + "=" * 78)
    print("4. THE TAIL -- what the bad months did")
    print("=" * 78)
    worst = s.nsmallest(6)
    print(f"  {'date':<14}{'return':>10}")
    print("  " + "-" * 26)
    for t, v in worst.items():
        print(f"  {t:%Y-%m-%d}{v*100:>9.1f}%")

    print("\n" + "=" * 78)
    print("5. LEVERED -- what it pays on $10,000")
    print("=" * 78)
    print(f"  {'leverage':<12}{'CAGR':>10}{'maxDD':>9}{'$/day on 10k':>15}"
          f"{'P(ruin)':>10}")
    print("  " + "-" * 58)
    for L in (1, 2, 3, 5):
        x = s*L
        st = stats(x, ann=12)
        eq = (1+np.clip(x, -0.99, None)).cumprod()
        ruin = (eq.min() < 0.2)
        if st:
            print(f"  {L}x{'':<10}{st['cagr']:>9.2f}%{st['dd']:>8.0f}%"
                  f"{10000*st['cagr']/100/252:>14,.0f}{'YES' if ruin else 'no':>10}")
