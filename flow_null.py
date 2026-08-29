"""Compare IC against its OWN null distribution, not against zero.

With a 12-bar horizon on 5-min data, consecutive observations overlap and are
strongly autocorrelated. The classic IC t-stat assumes independence, so it is
inflated -- which is why a SHUFFLED control returned t=2.75 when it must be
zero. Every IC t-statistic quoted earlier in this project is overstated for
the same reason.

The honest test: run the shuffled control many times to build the null
distribution of IC under "no signal", then ask where the real IC falls in it.
Also evaluate on NON-OVERLAPPING samples so the observations are closer to
independent.
"""
import numpy as np, pandas as pd
import flow_edge as F

HOR = 12

def nonoverlap(df):
    """Keep every HOR-th row so targets do not overlap."""
    return df.iloc[::HOR]

if __name__ == "__main__":
    D = {s: F.build(s) for s in F.SYMS}
    cols = [c for c in D[F.SYMS[0]].columns if c not in ("y","close","sym")]
    PC = [c for c in cols if c.startswith("p_")]
    FC = [c for c in cols if c.startswith("f_")]

    print("Building null distribution: 8 shuffled runs per symbol...")
    print("(non-overlapping evaluation, every 12th bar)\n")
    print(f"{'symbol':<10}{'real IC':>10}{'null mean':>11}{'null sd':>9}"
          f"{'z vs null':>11}{'verdict':>14}")
    print("-"*68)
    for s in F.SYMS:
        real = F.ic(nonoverlap(F.walk(D[s], cols)))[0]
        nulls = []
        for seed in range(8):
            nulls.append(F.ic(nonoverlap(F.walk(D[s], cols, shuffle=True,
                                                seed=100+seed)))[0])
        nulls = np.array(nulls)
        z = (real - nulls.mean())/nulls.std(ddof=1) if nulls.std(ddof=1)>0 else np.nan
        tag = "DEV" if s in F.DEV_SYMS else "HOLD"
        verdict = "SIGNAL" if z > 2.5 else ("marginal" if z > 1.5 else "noise")
        print(f"{s:<10}{real:>+10.4f}{nulls.mean():>+11.4f}{nulls.std(ddof=1):>9.4f}"
              f"{z:>+11.2f}{verdict:>14}  [{tag}]")

    print("\nFLOW-ONLY vs PRICE-ONLY, non-overlapping, same null treatment:")
    print(f"{'symbol':<10}{'price IC':>11}{'flow IC':>10}{'null sd':>10}")
    print("-"*45)
    for s in F.SYMS:
        rp = F.ic(nonoverlap(F.walk(D[s], PC)))[0]
        rf = F.ic(nonoverlap(F.walk(D[s], FC)))[0]
        nulls = [F.ic(nonoverlap(F.walk(D[s], FC, shuffle=True, seed=200+k)))[0]
                 for k in range(5)]
        print(f"{s:<10}{rp:>+11.4f}{rf:>+10.4f}{np.std(nulls,ddof=1):>10.4f}")
