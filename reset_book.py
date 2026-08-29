"""Reset the moonshot paper book to a clean $10,000.

Fair to reset: the first run's equity was partly an artifact of the 15-minute
re-levering bug in mark(), now fixed. Starting clean on corrected code gives
an honest forward test.

Not fixed by the reset: overnight gap risk. Measured on this universe, a -6%
gap in a held name happens roughly once every 92 sessions, so a 2-month run
carries a 37% chance of one and a 94% chance of a -3% one. Lower leverage
reduces the damage per gap; nothing removes the gaps.
"""
import json
import os
import sys

LEV = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0

for f in ("moonshot_state.json", "moonshot_trades.csv"):
    if os.path.exists(f):
        os.remove(f)
        print(f"  removed {f}")

print(f"\n  book reset: $10,000 start, {LEV:.0f}x leverage")
print(f"  target $20,000   floor $2,000   per-position stop 6%")
print(f"  first rebalance happens on the next cycle")
