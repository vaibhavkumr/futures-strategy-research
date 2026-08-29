"""IS MY MEASUREMENT BROKEN? Two checks.

CHECK 1 -- CALIBRATION. Inject a KNOWN edge into synthetic data and confirm
the pipeline detects it at the right size. If I plant a 55% directional edge
and my framework reports 50%, my framework is broken and every result today
is worthless.

CHECK 2 -- DIRECTIONAL ACCURACY, no stops or targets at all. Forget exits:
after each pattern, does price simply move the predicted way more than half
the time? This strips out every execution assumption I have made. If raw
direction is >50% but my R-based tests say fair odds, then my stop/target
structure is destroying a real edge -- which is a bug in ME, not a fact
about markets.
"""
import numpy as np, pandas as pd, glob, os
import confluence as C

# ---------------------------------------------------------------- CHECK 1
def synthetic(n=200_000, edge=0.0, seed=0):
    """Random walk, but after a 'signal bar' the next 30 bars drift with
    probability (0.5+edge) in the signal's direction."""
    rng = np.random.default_rng(seed)
    step = rng.normal(0, 1, n)
    sig = np.zeros(n, dtype=int)
    # plant signals every ~200 bars
    spots = np.arange(300, n-100, 200)
    for s in spots:
        d = 1 if rng.random() < 0.5 else -1
        sig[s] = d
        if rng.random() < 0.5 + edge:
            step[s+1:s+31] += d * 0.10          # planted drift
        else:
            step[s+1:s+31] -= d * 0.10
    px = 10000 + np.cumsum(step)
    return px, sig

def measure_direction(px, sig, horizon=30):
    idx = np.where(sig != 0)[0]
    idx = idx[idx + horizon < len(px)]
    moved = np.sign(px[idx+horizon] - px[idx])
    return (moved == sig[idx]).mean(), len(idx)

print("="*70)
print("CHECK 1 -- can my pipeline detect a KNOWN planted edge?")
print("="*70)
print(f"{'planted edge':<18}{'expected win%':>15}{'MEASURED win%':>16}{'n':>8}")
print("-"*60)
for e in (0.00, 0.02, 0.05, 0.10):
    px, sig = synthetic(edge=e, seed=1)
    acc, n = measure_direction(px, sig)
    print(f"{'+'+str(int(e*100))+'%':<18}{50+e*100:>14.1f}%{acc*100:>15.1f}%{n:>8}")
print("\n  If MEASURED tracks EXPECTED, the measurement works.\n")
