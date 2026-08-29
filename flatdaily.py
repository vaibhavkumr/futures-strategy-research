"""GOING FLAT EVERY DAY -- what it actually costs.

The request removes overnight gap risk completely. It also has two costs that
have to be measured before building it, because together they decide whether
the thing can work at all:

  1. THE DRIFT IS OVERNIGHT. Measured today on this exact universe:
     overnight 3.3-4.9bp/day (t=2.65-4.58), intraday no measurable drift.
     Going flat at the close means holding ONLY the session with no return.

  2. DAILY ROUND TRIPS. Selling everything at the close and rebuying at the
     open is turnover of 2.0 per day (out and back), every day, and the cost
     scales with LEVERAGE. This is the part that is easy to underestimate.
"""
import numpy as np, pandas as pd, yfinance as yf
import factor_lab as F

px = F.universe()
d = yf.download(list(px.columns), start="2004-01-01", progress=False,
                auto_adjust=True)
O, C = d["Open"].ffill(), d["Close"].ffill()

intra = (C/O - 1).mean(axis=1).dropna()          # open -> close
overn = (O/C.shift(1) - 1).mean(axis=1).dropna()  # close -> open
full = C.pct_change().mean(axis=1).dropna()

print("="*74)
print("1. WHICH SESSION HAS THE RETURN?")
print("="*74)
for lab, s in (("overnight (close->open)", overn), ("intraday (open->close)", intra),
               ("full day (hold both)", full)):
    t = s.mean()/(s.std(ddof=1)/np.sqrt(len(s)))
    print(f"  {lab:<26}{s.mean()*1e4:>7.2f} bp/day   t={t:>5.2f}   "
          f"{((1+s.mean())**252-1)*100:>7.2f}%/yr")

print("\n"+"="*74)
print("2. THE DAILY ROUND-TRIP COST")
print("="*74)
print("  flat at the close + rebuy at the open = turnover 2.0/day")
print("  cost = 2.0 x 10bp x leverage\n")
print(f"  {'leverage':<12}{'cost/day':>12}{'cost/yr':>14}{'over 42 days':>15}")
print("  "+"-"*52)
for lev in (1, 5, 10, 20, 30):
    cd = 2.0*10/1e4*lev
    print(f"  {lev:>3}x{'':<8}{cd*100:>11.2f}%{cd*252*100:>13.0f}%"
          f"{(1-(1-cd)**42)*100:>14.1f}%")

print("\n"+"="*74)
print("3. NET RESULT OF THE DAILY-FLAT VERSION")
print("="*74)
print(f"  {'leverage':<12}{'gross/yr':>12}{'costs/yr':>12}{'NET/yr':>12}"
      f"{'$10k after 2mo':>17}")
print("  "+"-"*62)
gross_1x = ((1+intra.mean())**252-1)
for lev in (1, 5, 10, 20, 30):
    cd = 2.0*10/1e4*lev
    daily_net = intra.mean()*lev - cd
    net_yr = (1+daily_net)**252-1
    after = 10000*(1+daily_net)**42
    print(f"  {lev:>3}x{'':<8}{gross_1x*lev*100:>11.1f}%{cd*252*100:>11.0f}%"
          f"{net_yr*100:>11.1f}%{after:>17,.0f}")

print("\n"+"="*74)
print("4. FOR COMPARISON -- holding overnight, which is what runs now")
print("="*74)
for lev in (10, 20):
    dn = full.mean()*lev
    print(f"  {lev:>3}x hold overnight   net {((1+dn)**252-1)*100:>8.1f}%/yr"
          f"   $10k after 2mo: {10000*(1+dn)**42:>10,.0f}")
