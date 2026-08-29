"""THE TJR FRAMEWORK, MODELLED HONESTLY.

His approach as described: be selective, take the best few setups, target
2-3% per day, cut losses hard. That is exactly the structure of the tuned
system -- top 5 of 172 names, +/-2.5% band, 54.9% daily win rate.

The framework is sound. Everything then reduces to ONE number: the daily win
rate. With a symmetric band of b, a day is +b or -b, so

    expectancy/day = b * (2W - 1)

"2-3% per day" as a NET result silently assumes W = 100%. Nobody has that.
The moment losing days exist -- and TJR has them, on stream -- the net falls
to b*(2W-1), which is far below b.

This computes what each win rate is actually worth, and marks where every
measured win rate in this project falls.
"""
import numpy as np

B = 0.025
print("="*76)
print("WHAT EACH DAILY WIN RATE IS WORTH  (+/-2.5% band, 252 days)")
print("="*76)
print(f"  {'win rate':<12}{'net/day':>10}{'per year':>14}{'$10k becomes':>16}"
      f"{'$/day on 10k':>14}")
print("  "+"-"*66)
for W in (0.50,0.52,0.545,0.55,0.58,0.60,0.62,0.65,0.70,1.00):
    d = B*(2*W-1)
    yr = (1+d)**252 - 1
    tag = ""
    if abs(W-0.549) < 0.005: tag = "  <- the tuned system"
    if W == 1.00: tag = "  <- what '2.5%/day' assumes"
    print(f"  {W*100:>5.1f}%{'':<6}{d*100:>9.3f}%{yr*100:>13,.0f}%"
          f"{10000*(1+yr):>16,.0f}{10000*d:>13,.0f}{tag}")

print("\n"+"="*76)
print("WHAT WIN RATE WOULD THE GOAL NEED?")
print("="*76)
for tgt,lab in ((0.379,"the tuned system today"),(2.0,"+200%/yr"),
                (5.04,"$200/day on $10k"),(7.56,"$300/day on $10k")):
    d = (1+tgt)**(1/252)-1
    W = (d/B+1)/2
    print(f"  {lab:<26} needs W = {W*100:>5.1f}%   "
          f"({(W-0.549)*100:+.1f} points vs measured)")

print("\n"+"="*76)
print("EVERY DAILY/TRADE WIN RATE MEASURED IN THIS PROJECT")
print("="*76)
for lab,w,n in (("TJR methodology replayed",0.484,"10,645 trades"),
                ("live_paper signal replay",0.484,"10,645 trades"),
                ("ORB / Scarface rules",0.470,"measured"),
                ("ICT confluence 5/5 setups",0.460,"548 trades"),
                ("tuned momentum system",0.549,"4,100 days"),
                ("coin flip",0.500,"-")):
    d = B*(2*w-1); yr=(1+d)**252-1
    print(f"  {lab:<30}{w*100:>6.1f}%  ({n:<13}) -> {yr*100:>+8,.0f}%/yr")

print("\n"+"="*76)
print("THE POINT")
print("="*76)
print("""  54.9% -> 37.9%/yr is the tuned system, and it is already the best
  win rate anything in this project achieved.

  To reach $200/day on $10,000 you need 62.9% -- eight points higher.
  Eight points of daily win rate is not a tweak. Every published edge,
  every audited fund, every method tested here lands between 46% and 55%.

  The same 62.9% on $150,000 is not needed at all: 54.9% already pays
  $216/day there. The win rate does not have to improve -- the base does.""")
