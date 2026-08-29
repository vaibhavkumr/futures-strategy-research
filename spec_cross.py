"""CROSS-MARKET: the spec build on four indices it was never developed on."""
import numpy as np, pandas as pd, tjr_spec as S

from duka import load

MK = {"NASDAQ": "usatechidxusd", "S&P 500": "usa500idxusd",
      "DOW": "usa30idxusd", "DAX": "deuidxeur"}
# SMT partner: each index paired with a different one, as TJR pairs NQ with ES
PAIR = {"NASDAQ": "usa500idxusd", "S&P 500": "usatechidxusd",
        "DOW": "usa500idxusd", "DAX": "usa500idxusd"}

print(f"{'market':<10}{'n':>6}{'win%':>8}{'expR':>9}{'t':>8}   95% CI")
print("-" * 60)
res = {}
for name, slug in MK.items():
    try:
        d5, d1 = load(slug, "m5"), load(slug, "m1")
    except Exception as e:
        print(f"{name:<10}  load failed: {e}"); continue
    lo = max(d5.index[0], d1.index[0])
    d5, d1 = d5[d5.index >= lo], d1[d1.index >= lo]
    cor = load(PAIR[name], "m5").reindex(d5.index, method="ffill")
    t = S.generate(d5, d1, corr5=cor)
    if len(t) < 10:
        print(f"{name:<10}{len(t):>6}  too few"); continue
    R = t.R.values
    m, se = R.mean(), R.std(ddof=1) / np.sqrt(len(R))
    res[name] = t
    print(f"{name:<10}{len(R):>6}{(R>0).mean()*100:>8.1f}{m:>9.3f}{m/se:>8.2f}"
          f"   [{m-1.96*se:+.3f}, {m+1.96*se:+.3f}]")

if res:
    allR = np.concatenate([t.R.values for t in res.values()])
    m, se = allR.mean(), allR.std(ddof=1) / np.sqrt(len(allR))
    print("-" * 60)
    print(f"{'POOLED':<10}{len(allR):>6}{(allR>0).mean()*100:>8.1f}{m:>9.3f}{m/se:>8.2f}"
          f"   [{m-1.96*se:+.3f}, {m+1.96*se:+.3f}]")
    print("\n15m branch only (the one that looked positive on Nasdaq):")
    b = np.concatenate([t[t.tf == "15m"].R.values for t in res.values()])
    if len(b) > 5:
        m, se = b.mean(), b.std(ddof=1)/np.sqrt(len(b))
        print(f"{'  15m':<10}{len(b):>6}{(b>0).mean()*100:>8.1f}{m:>9.3f}{m/se:>8.2f}"
              f"   [{m-1.96*se:+.3f}, {m+1.96*se:+.3f}]")
    b5 = np.concatenate([t[t.tf == "5m"].R.values for t in res.values()])
    if len(b5) > 5:
        m, se = b5.mean(), b5.std(ddof=1)/np.sqrt(len(b5))
        print(f"{'  5m':<10}{len(b5):>6}{(b5>0).mean()*100:>8.1f}{m:>9.3f}{m/se:>8.2f}"
              f"   [{m-1.96*se:+.3f}, {m+1.96*se:+.3f}]")

print("\n15m branch, per market (are the 4 markets independent evidence?):")
for name, t in res.items():
    g = t[t.tf == "15m"]
    if len(g) < 5: continue
    m, se = g.R.mean(), g.R.std(ddof=1)/np.sqrt(len(g))
    print(f"  {name:<9}{len(g):>5}{(g.R>0).mean()*100:>8.1f}{m:>9.3f}{m/se:>8.2f}")

# how much do these markets overlap? same-day same-direction signals are
# not independent observations -- they are one bet counted four times.
print("\nsame-day directional overlap between markets:")
days = {k: set(zip(pd.to_datetime(v.ts).dt.date, v.side)) for k, v in res.items()}
ks = list(days)
for a in range(len(ks)):
    for b in range(a+1, len(ks)):
        inter = len(days[ks[a]] & days[ks[b]])
        union = len(days[ks[a]] | days[ks[b]])
        print(f"  {ks[a]:<9} vs {ks[b]:<9} {inter:>4} shared / {union} = {inter/union*100:.0f}%")
