"""Traders HAVE compounded $10k into a $10k/month income. How?

Not by being wrong about that -- it happens. The question is the mechanism.
Run a large population of traders with identical, realistic assumptions and
count how many get there, how many are wiped out, and what the median does.

If the winners are the tail of a distribution rather than the result of a
repeatable process, then copying their POSITION SIZE is what produced them,
and the same sizing produces the wipeouts nobody posts about.
"""
import numpy as np

START, GOAL = 10_000.0, 120_000.0     # $10k -> $10k/month of income
DAYS, TRADES = 252, 3
RUIN = 2_000.0                        # below this you cannot trade futures


def population(edge_R, risk_frac, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    bal = np.full(n, START)
    alive = np.ones(n, bool)
    peak_ruin = np.zeros(n, bool)
    p = 0.5 + edge_R / 2.0
    for _ in range(DAYS * TRADES):
        stake = bal * risk_frac
        wins = rng.random(n) < p
        bal = np.where(alive, bal + np.where(wins, stake, -stake), bal)
        newly = alive & (bal < RUIN)
        peak_ruin |= newly
        alive &= ~newly
    return bal, peak_ruin


print("=" * 84)
print("200,000 traders. $10,000 each. One year. Identical strategy and sizing.")
print("=" * 84)
print(f"{'edge':<12}{'risk/trade':<12}{'reached $120k':>15}{'wiped out':>12}"
      f"{'median end':>14}{'mean end':>14}")
print("-" * 84)
for edge, elabel in ((0.00, "coin flip"), (0.05, "+0.05R"), (0.10, "+0.10R")):
    for rf in (0.01, 0.05, 0.15, 0.30):
        bal, ruin = population(edge, rf)
        hit = (bal >= GOAL).mean() * 100
        print(f"{elabel:<12}{rf*100:>4.0f}%       {hit:>14.2f}%"
              f"{ruin.mean()*100:>11.1f}%{np.median(bal):>14,.0f}"
              f"{bal.mean():>14,.0f}")
    print()

print("=" * 84)
print("THE POINT")
print("=" * 84)
bal, ruin = population(0.00, 0.30)
hit = bal >= GOAL
print(f"  At 30% risk with NO EDGE AT ALL: {hit.mean()*100:.2f}% reach $120k.")
print(f"  That is {int(hit.sum()):,} traders out of 200,000 who genuinely did it,")
print(f"  with a strategy that is provably worthless.")
print(f"  In the same population {ruin.mean()*100:.1f}% were wiped out.")
print()
print("  Every one of those winners has a real screenshot, a real account,")
print("  and a real story about discipline and their strategy. None of them")
print("  is lying. The strategy just wasn't what produced the outcome.")
print()
print("  Note the mean vs the median: the mean is dragged up by a handful of")
print("  enormous winners while the typical trader is at or near zero. That")
print("  gap IS survivorship bias, in one number.")
