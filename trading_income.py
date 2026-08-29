"""$10k/MONTH THROUGH TRADING -- what it actually takes.

The wall is not "trading cannot pay $120k/yr". Plenty of people earn that
trading. The wall is earning it on $10,000 of capital, which needs a ~4.08
Sharpe that has never existed.

Separate the two variables. Monthly income = CAPITAL x RETURN. I have spent
the whole session attacking RETURN and hit a hard ceiling around Sharpe 1.14
(26%/yr levered). So the remaining variable is CAPITAL -- and there is an
industry built specifically to hand traders capital they do not own.

  PROP FIRMS / FUNDED ACCOUNTS. Pass an evaluation, trade the firm's money,
  keep 80-90% of profits. Topstep, Apex, FTMO, Take Profit Trader et al.
  You risk an evaluation fee (~$50-600), not your $10,000.

This computes: capital required at my verified return, what the funded route
needs to clear, and -- honestly -- the part that does not fit.
"""
import numpy as np

TARGET_M=10_000; TARGET_Y=TARGET_M*12
print("="*76)
print("1. CAPITAL REQUIRED FOR $10,000/MONTH")
print("="*76)
print(f"  {'annual return':<20}{'capital needed':>18}{'realistic?':>14}")
print("  "+"-"*52)
for r,note in ((0.10,"index buy&hold"),(0.146,"my verified system 1x"),
               (0.261,"verified + tilt + 3x"),(0.50,"exceptional"),
               (1.00,"top 0.1% of traders"),(12.0,"needed from $10k")):
    cap=TARGET_Y/r
    flag="yes" if cap>=50_000 else ("borderline" if cap>=20_000 else "no")
    print(f"  {r*100:>5.0f}%  {note:<13}{cap:>18,.0f}{flag:>14}")
print(f"\n  At my measured 26.1%/yr you need ${TARGET_Y/0.261:,.0f} working.")
print(f"  That is the whole problem in one number.")

print("\n"+"="*76)
print("2. THE FUNDED-ACCOUNT ROUTE")
print("="*76)
print("  Trade the firm's capital, keep a split. Typical structures:\n")
print(f"  {'account':<12}{'split':>7}{'your $/mo at 26%/yr':>22}{'%/mo needed for $10k':>22}")
print("  "+"-"*62)
for cap,split in ((50_000,0.90),(100_000,0.90),(150_000,0.90),
                  (300_000,0.90),(500_000,0.90),(1_000_000,0.90)):
    monthly=cap*0.261/12*split
    need=TARGET_M/(cap*split)*100
    print(f"  ${cap:>10,}{split*100:>6.0f}%{monthly:>22,.0f}{need:>21.2f}%")
print("\n  Scaling plans at the big firms top out around $1-2M of buying power,")
print("  reached by hitting profit targets consistently over many months.")

print("\n"+"="*76)
print("3. WHAT MY VERIFIED EDGE DELIVERS ON FUNDED CAPITAL")
print("="*76)
print("  26.1%/yr = 1.95%/month gross, before the profit split.\n")
for cap in (50_000,150_000,300_000,500_000):
    m=cap*0.261/12*0.90
    print(f"  ${cap:>8,} funded  ->  ${m:>7,.0f}/month   "
          f"{'REACHES $10k' if m>=10000 else f'{m/TARGET_M*100:.0f}% of target'}")
print(f"\n  -> ${TARGET_Y/0.261/0.9:,.0f} of funded capital clears $10k/month at my numbers.")

print("\n"+"="*76)
print("4. THE PART THAT DOES NOT FIT -- read this before paying any fee")
print("="*76)
print("""  My verified edge is a WEEKLY-REBALANCED ETF PORTFOLIO. Nearly every
  futures prop firm requires:
     - intraday futures, flat or near-flat by the close
     - a trailing drawdown limit, often 3-5% of account
     - consistency rules capping any single day's share of profit

  My 26%/yr system holds positions for a week and takes -38% drawdowns.
  It would violate the drawdown rule in month one and cannot be run there.

  And the intraday futures trading these firms DO permit is exactly where I
  tested ~50 candidates and found no edge that survives costs -- including
  the full TJR methodology at 48.4% win rate, t=-9.60 over 10,645 trades.

  Published pass rates run ~5-10% for the evaluation, and far fewer stay
  funded. That is the firms' business model: evaluation fees from the ~90%.

  So the honest statement: funded capital solves the CAPITAL problem, and I
  have no verified edge that fits the CONSTRAINTS those accounts impose.""")

print("\n"+"="*76)
print("5. WHAT WOULD ACTUALLY HAVE TO HAPPEN")
print("="*76)
print("""  Either
    A. find an intraday futures edge that survives costs AND a 4% trailing
       drawdown. ~50 candidates tested, 0 found. Not proven impossible,
       but I would not spend your money on the odds.
    B. use a swing-friendly funded program (some equities/forex firms allow
       overnight and weekly holds) and run the verified system there. This
       is the only route where an edge I have ACTUALLY VERIFIED meets
       capital I can access.
    C. grow your own capital toward ~$460k. Slow, certain, needs income.""")
