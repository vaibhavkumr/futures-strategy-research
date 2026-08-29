"""DO THE BIG WINNERS PROVE AN EDGE EXISTS?

The claim: some people DID turn small accounts into huge ones by sizing up on
trades they were confident about. True -- they demonstrably exist.

The question is what their existence is evidence OF. There is an exact result
here, not a hand-wave. For a FAIR game (zero edge), the optional stopping
theorem gives a hard bound:

    P(multiply your capital by k before losing it)  <=  1/k

Exactly 1/k for a fair game, strictly less once costs are included. No
strategy, sizing rule, or conviction changes this bound -- it follows from the
game being fair, and it is the same bound for a coin-flipper and for someone
with a detailed thesis.

So: if a million people try to turn 10k into 1M with zero edge, roughly
1,000,000/100 = 10,000 of them SUCCEED. Ten thousand people with a real
account screenshot, a real story, and a genuine memory of the trade they were
sure about. Their existence is exactly what a zero-edge world predicts.

This file checks that bound by simulation, adds realistic costs, and then asks
the only question that separates the two worlds: does the number of winners
EXCEED what chance produces?
"""
import numpy as np
rng=np.random.default_rng(3)

print("="*74)
print("1. THE BOUND: P(100x before ruin) for a FAIR game")
print("="*74)
n=400_000
for edge_bp,lab in ((0.0,"zero edge (fair)"),(-1.0,"realistic: -1bp/trade cost"),
                    (+2.0,"a REAL small edge: +2bp/trade")):
    # aggressive sizing: 25% of equity per trade, 400 trades/yr, 2 yrs
    eq=np.ones(n); alive=np.ones(n,bool); hit=np.zeros(n,bool)
    for t in range(800):
        r=rng.standard_normal(n)*0.25 + edge_bp/1e4*25
        eq=np.where(alive,eq*(1+np.clip(r,-0.99,None)),eq)
        hit|=alive&(eq>=100)
        alive&=~hit&(eq>0.02)
    print(f"  {lab:<30} P(100x) = {hit.mean()*100:6.3f}%   "
          f"theory 1/100 = 1.000%   ruined {(~alive&~hit).mean()*100:.1f}%")

print("\n"+"="*74)
print("2. HOW MANY WINNERS DOES CHANCE ALONE CREATE?")
print("="*74)
print("  FINRA/broker data: roughly 10 million retail accounts trade actively;")
print("  take a conservative 1 million pushing hard for outsized growth.\n")
for pop in (100_000,1_000_000,10_000_000):
    print(f"  {pop:>10,} people attempting 10k -> 1M  ->  "
          f"{pop//100:>8,} succeed BY CHANCE ALONE")
print("\n  Every one of them has a real screenshot and a real story about the")
print("  trade they were confident on. None of it is fabricated. It is simply")
print("  that we only ever hear from this group -- the ~99% who sized up with")
print("  equal conviction and went to zero do not post, sell courses, or stream.")

print("\n"+"="*74)
print("3. THE TEST THAT ACTUALLY SEPARATES SKILL FROM CHANCE")
print("="*74)
print("  If big winners come from SKILL, their count exceeds the chance rate,")
print("  and -- critically -- they REPEAT. Chance winners cannot repeat.\n")
p1=0.01
for k,lab in ((2,"do it twice"),(3,"three times")):
    print(f"  P(a chance winner {lab} in a row) = {p1**k*100:.4f}% "
          f"-> of 10,000 chance winners, {10000*p1**k:.1f} repeat")
print("\n  So the diagnostic is not 'did anyone do it' -- someone always does.")
print("  It is 'does the SAME person do it repeatedly, audited, net of fees.'")
print("  That set is tiny, and its members run Sharpe ~2, not Sharpe 4.")

print("\n"+"="*74)
print("4. WHAT CONVICTION IS WORTH -- measured, not argued")
print("="*74)
print("  From this project's own data:")
print("   - 10,645 TJR-methodology trades replayed: 48.4% win, -0.100R, t=-9.60")
print("   - Riley 72-stream audit: his highest-conviction, most-explained")
print("     setups performed no differently from the rest")
print("   - Every high-conviction filter I tested (confluence, MTF alignment,")
print("     killzone + sweep + FVG stacked) landed at fair odds")
print("  Conviction is real as a feeling and measurable as zero as a predictor.")
