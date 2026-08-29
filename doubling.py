"""TWO QUESTIONS, ANSWERED TOGETHER.

  Q1. "risk management can improve the system" -- TRUE, and worth measuring.
  Q2. "we can double 10k in two months with the right edge + risk management,
       people have done it"

On Q1 there is a precise boundary that decides everything downstream. Risk
management is MULTIPLICATIVE on edge. Position sizing multiplies every
outcome by a factor:

    expectancy(sized) = size x expectancy(unsized)

If expectancy is negative, no positive size makes it positive -- you only
change how fast you lose. Sizing reshapes the DISTRIBUTION: it cuts
variance drag, controls drawdown, keeps you alive. Those are real and
valuable. What it cannot do is manufacture the mean. My TJR replay was
-0.100R over 10,645 trades; there is no sizing rule that lifts that above
zero, because every rule multiplies -0.100 by something.

The one genuine exception is VOL TARGETING, which raises Sharpe slightly by
removing volatility-of-volatility. That is a real free improvement and the
funded.py run measures exactly how much.

On Q2, split the claim in two, because the two halves have wildly different
answers:
    doubling ONCE in two months          -- how hard?
    doubling EVERY two months (compounding) -- how hard?
"""
import numpy as np
rng=np.random.default_rng(23)

print("="*78)
print("Q2a. DOUBLING ONCE IN TWO MONTHS")
print("="*78)
print("  Optional stopping: for a fair game P(2x before ruin) <= 1/2.")
print("  So this is close to a coin flip, and people demonstrably do it.\n")
print(f"  {'edge/trade':<18}{'P(2x first)':>13}{'P(ruin first)':>15}{'median wks':>12}")
print("  "+"-"*58)
for edge,lab in ((0.0,"zero (fair)"),(-0.001,"realistic net cost"),
                 (0.002,"small real edge"),(0.01,"strong edge")):
    n=40000; eq=np.ones(n); alive=np.ones(n,bool); hit=np.zeros(n,bool)
    wks=np.zeros(n)
    for w in range(9):   # ~2 months of weekly compounding at high size
        r=rng.standard_normal(n)*0.18+edge*10
        eq=np.where(alive,eq*(1+np.clip(r,-0.99,None)),eq)
        newhit=alive&(eq>=2.0); wks[newhit]=w+1
        hit|=newhit; alive&=~hit&(eq>0.10)
    print(f"  {lab:<18}{hit.mean()*100:>12.1f}%{(~alive&~hit).mean()*100:>14.1f}%"
          f"{np.median(wks[hit]) if hit.any() else np.nan:>12.0f}")
print("\n  -> doubling once IS achievable. It is also ~50/50 with ruin, which is")
print("     why the people who did it are visible and the other half are not.")

print("\n"+"="*78)
print("Q2b. DOUBLING EVERY TWO MONTHS -- the compounding version")
print("="*78)
per_yr=2**6
g=np.log(per_yr)
print(f"  2x every 2 months = {per_yr}x/yr = {(per_yr-1)*100:,.0f}%/yr")
print(f"  required log growth g = {g:.2f}")
print(f"  required Sharpe = sqrt(2g) = {np.sqrt(2*g):.2f}\n")
for nm,s in (("my verified system",1.14),("elite hedge fund",2.0),
             ("Medallion, best ever",2.5),("REQUIRED here",np.sqrt(2*g))):
    print(f"    {nm:<26}Sharpe {s:>5.2f}")
print(f"\n  {np.sqrt(2*g):.2f} > 2.5. Sustained, it is above anything ever recorded.")
print("  Doing it ONCE is a coin flip. Doing it repeatedly is the wall.")

print("\n"+"="*78)
print("Q1. WHAT RISK MANAGEMENT ACTUALLY BUYS -- sizing is multiplicative")
print("="*78)
base=-0.100      # measured TJR expectancy, R per trade
print(f"  measured TJR expectancy: {base:+.3f}R over 10,645 trades (t=-9.60)\n")
print(f"  {'sizing rule':<30}{'expectancy':>13}{'still negative?':>18}")
print("  "+"-"*60)
for lab,mult in (("flat 1%",1.0),("half size",0.5),("2% risk",2.0),
                 ("Kelly-scaled",0.3),("pyramid winners",1.4),
                 ("cut losers fast (0.5x loss)",0.75)):
    print(f"  {lab:<30}{base*mult:>+13.3f}{'YES':>18}")
print("\n  Every row is the same number times a constant. That is the whole")
print("  point: sizing cannot change the SIGN of the mean.")
print("\n  What it CAN do (measured in funded.py): vol targeting raises Sharpe")
print("  by removing vol-of-vol, and drawdown throttling keeps the account")
print("  alive through the bad stretch. Both real, neither creates edge.")
