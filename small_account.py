"""CAN THIS RUN ON $200 OF REAL MONEY?

Three independent constraints, checked separately, because each one alone can
be fatal:

  1. LEVERAGE. The bot runs 20x. US retail equity/ETF accounts are governed by
     Reg T: 50% initial margin = 2:1 max overnight. Day-trading buying power of
     4:1 exists but requires a $25,000 minimum equity (PDT rule) AND being flat
     by the close -- which, per the overnight/intraday work, removes the entire
     return. Portfolio margin (6:1) needs $100,000+.
     There is no retail path to 20x on an ETF book at any account size.

  2. WHOLE SHARES. $200 cannot buy one share of most of these. Fractional
     shares fix this at Fidelity/Schwab/Robinhood/IBKR, so this one is solvable.

  3. COSTS AS A FRACTION OF CAPITAL. Commissions are mostly zero now, but the
     bid-ask spread is not, and it does not shrink with account size.

This computes what $200 actually earns at LEGAL leverage.
"""
import numpy as np
import factor_lab as F

px = F.universe()
r = F.run(px, F.f_mom12)          # the verified weekly momentum book
ann = (1+r).prod()**(52/ (len(r)/5)) if False else None
mu_d = r.mean(); sd_d = r.std(ddof=1)
ann_ret = (1+mu_d)**252 - 1
ann_vol = sd_d*np.sqrt(252)
print("="*74)
print("THE VERIFIED SYSTEM, UNLEVERED")
print("="*74)
print(f"  annual return {ann_ret*100:6.2f}%   vol {ann_vol*100:5.2f}%   "
      f"Sharpe {mu_d/sd_d*np.sqrt(252):.2f}")

FIN = 0.055     # margin interest, typical retail
print("\n"+"="*74)
print("WHAT $200 EARNS AT EACH LEVERAGE")
print("="*74)
print(f"  {'leverage':<26}{'net/yr':>9}{'$200 ->':>11}{'profit/yr':>12}{'legal?':>22}")
print("  "+"-"*78)
rows = [
    (1.0,  "1x cash account",          "yes"),
    (2.0,  "2x Reg T margin",          "yes"),
    (4.0,  "4x day-trade buying power","needs $25k + flat daily"),
    (6.0,  "6x portfolio margin",      "needs $100k+"),
    (20.0, "20x (what the bot runs)",  "NOT AVAILABLE to retail"),
]
for lev, lab, legal in rows:
    net = lev*ann_ret - (lev-1)*FIN
    end = 200*(1+net)
    print(f"  {lab:<26}{net*100:>8.1f}%{end:>11,.0f}{end-200:>+12,.0f}{legal:>22}")

print("\n"+"="*74)
print("SPREAD COST -- does it scale away at $200?")
print("="*74)
print("  Spread is a PERCENTAGE, so it hits $200 and $200,000 identically.")
print("  Weekly rebalance turnover ~12-100%, at ~2bp round trip on liquid ETFs:\n")
for turn, lab in ((0.12, "typical rebalance (12% turnover)"),
                  (1.00, "full reshuffle (100%)")):
    for lev in (1, 2):
        c = turn*2/1e4*lev*52
        print(f"    {lab:<34} {lev}x -> {c*100:5.2f}%/yr  "
              f"= ${200*c:5.2f} on $200")

print("\n"+"="*74)
print("MINIMUM ACCOUNT FOR THE BOOK AS WRITTEN (whole shares, 26 names)")
print("="*74)
import json
s = json.load(open("moonshot_state.json"))
lp, w = s["last_px"], s["weights"]
need = max(lp[a]/wt for a, wt in w.items() if a in lp and wt > 0)
print(f"  smallest weight that must still buy 1 share -> ${need:,.0f} of notional")
for lev in (1, 2):
    print(f"    at {lev}x that is ${need/lev:,.0f} of your own capital")
print("  (fractional shares remove this constraint entirely)")
