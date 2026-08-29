"""Does the flow signal survive REAL crypto costs?

IC is not money. Binance spot taker is 0.10% PER SIDE -- 20bp round trip,
ten times the friction we were fighting in futures. An IC of 0.02 has to move
price more than that, after slippage, or it is a statistical curiosity.
"""
import numpy as np, pandas as pd
import flow_edge as F

def simulate(X, pred, thresh, fee_bp, hold=12, slip_bp=2.0):
    X = X.loc[pred.index]
    c = X["close"].values
    p = pred.values
    n = len(X)
    rows = []
    i = 0
    while i < n - hold - 1:
        if abs(p[i]) < thresh:
            i += 1
            continue
        sgn = 1 if p[i] > 0 else -1
        r = (c[i + hold] / c[i] - 1) * sgn * 1e4        # bp
        rows.append(r - fee_bp - slip_bp)
        i += hold
    return np.array(rows)

if __name__ == "__main__":
    D = {s: F.build(s) for s in F.SYMS}
    cols = [c for c in D[F.SYMS[0]].columns if c not in ("y","close","sym")]
    P = {s: F.walk(D[s], cols) for s in F.SYMS}

    print("=" * 76)
    print("TRADE SIMULATION -- signal converted to money, per fee level")
    print("=" * 76)
    for fee, label in ((0.0,"0bp  (free, impossible)"),
                       (4.0,"4bp  (VIP maker rebate tier)"),
                       (10.0,"10bp (Binance maker, both sides)"),
                       (20.0,"20bp (Binance TAKER, realistic)")):
        allr=[]
        for s in F.SYMS:
            q = np.quantile(np.abs(P[s]["pred"]), 0.80)
            allr.append(simulate(D[s], P[s]["pred"], q, fee))
        R=np.concatenate(allr)
        m,se=R.mean(),R.std(ddof=1)/np.sqrt(len(R))
        print(f"  {label:<34} n={len(R):<6} {m:+7.2f}bp  t={m/se:+6.2f}  "
              f"win {(R>0).mean()*100:5.1f}%")

    print("\n" + "=" * 76)
    print("PER SYMBOL at realistic 20bp taker cost (top 20% confidence)")
    print("=" * 76)
    for s in F.SYMS:
        q = np.quantile(np.abs(P[s]["pred"]), 0.80)
        R = simulate(D[s], P[s]["pred"], q, 20.0)
        m,se=R.mean(),R.std(ddof=1)/np.sqrt(len(R))
        tag = "DEV" if s in F.DEV_SYMS else "HOLD"
        print(f"  {s:<9}[{tag:<4}] n={len(R):<6} {m:+7.2f}bp  t={m/se:+6.2f}  "
              f"win {(R>0).mean()*100:5.1f}%")

    print("\n" + "=" * 76)
    print("HOW BIG IS THE MOVE THE SIGNAL PREDICTS? (gross, no costs)")
    print("=" * 76)
    for s in F.SYMS:
        q = np.quantile(np.abs(P[s]["pred"]), 0.80)
        R = simulate(D[s], P[s]["pred"], q, 0.0, slip_bp=0.0)
        print(f"  {s:<9} top-20% confidence trades move {R.mean():+6.2f}bp "
              f"in the predicted direction")
    print("\n  Round-trip cost to beat: 20bp taker / 10bp maker.")
