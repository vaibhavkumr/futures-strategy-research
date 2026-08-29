"""Temporal split + lookahead audit on the spec build."""
import numpy as np, pandas as pd, tjr_spec as S

t = pd.read_pickle("tjr_spec_nq.pkl")
t["ts"] = pd.to_datetime(t["ts"])
t = t.sort_values("ts").reset_index(drop=True)

def stat(x, name):
    x = np.asarray(x, float)
    n = len(x)
    if n < 5:
        print(f"{name:<22} n={n:<5} (too few)"); return
    m, sd = x.mean(), x.std(ddof=1)
    se = sd / np.sqrt(n)
    tt = m / se if se else 0.0
    wr = (x > 0).mean() * 100
    print(f"{name:<22} n={n:<5} win {wr:5.1f}%  expR {m:+.3f}  t={tt:+5.2f}  "
          f"CI[{m-1.96*se:+.3f},{m+1.96*se:+.3f}]")

print("=" * 74)
print("TEMPORAL SPLIT  (dev = first 60% of calendar time, holdout = last 40%)")
print("=" * 74)
cut = t.ts.min() + (t.ts.max() - t.ts.min()) * 0.60
dev, hold = t[t.ts <= cut], t[t.ts > cut]
print(f"cut at {cut:%Y-%m-%d}")
stat(t.R, "ALL")
stat(dev.R, "  dev (in-sample)")
stat(hold.R, "  HOLDOUT")

print("\nper calendar half-year:")
for p, g in t.groupby(t.ts.dt.to_period("Q")):
    stat(g.R, f"  {p}")

print("\nby working timeframe x split:")
for tf in ("5m", "15m"):
    for nm, g in (("dev", dev), ("hold", hold)):
        stat(g[g.tf == tf].R, f"  {tf} {nm}")

print("\nby side:")
for sd in ("long", "short"):
    stat(t[t.side == sd].R, f"  {sd}")

print("\nby confirmation type:")
if "confirm" in t.columns:
    for c, g in t.groupby("confirm"):
        stat(g.R, f"  {c}")

print("\nby third confluence:")
if "conf3" in t.columns:
    for c, g in t.groupby("conf3"):
        stat(g.R, f"  {c}")
