"""LOOKAHEAD AUDIT.

Replay the strategy as if live. For each signal the full-history run produced,
re-run generate() on data TRUNCATED at that signal's bar. If the signal is
real, it must still be there. If it vanishes or changes, the full run was
reading bars that had not happened yet.
"""
import numpy as np, pandas as pd, tjr_spec as S

df5 = pd.read_pickle("nq_5m.pkl"); df1 = pd.read_pickle("nq_1m.pkl")
lo = max(df5.index[0], df1.index[0])
df5, df1 = df5[df5.index >= lo], df1[df1.index >= lo]
es = pd.read_csv("download/usa500idxusd-m5-bid-2022-01-01-2026-07-24.csv")
es.columns = [c.lower() for c in es.columns]
es.index = pd.to_datetime(es["timestamp"], utc=True).dt.tz_convert("America/New_York")
es = es[["open","high","low","close"]].astype(float).reindex(df5.index, method="ffill")

full = S.generate(df5, df1, corr5=es)
full["ts"] = pd.to_datetime(full["ts"])
print(f"full-history run: {len(full)} signals")

pick = full.sample(min(50, len(full)), random_state=7).sort_values("ts")
ok = miss = diff = 0
for _, row in pick.iterrows():
    ts = row.ts
    c5, c1, ce = df5[df5.index <= ts], df1[df1.index <= ts], es[es.index <= ts]
    if len(c1) < 300 or len(c5) < 200:
        continue
    t = S.generate(c5, c1, corr5=ce)
    if t.empty:
        miss += 1; continue
    t["ts"] = pd.to_datetime(t["ts"])
    m = t[t.ts == ts]
    if m.empty:
        miss += 1
    elif m.iloc[0].side != row.side:
        diff += 1
    else:
        ok += 1

n = ok + miss + diff
print(f"\nreplayed {n} signals with ALL future bars removed")
print(f"  reproduced identically : {ok}")
print(f"  VANISHED               : {miss}")
print(f"  changed side           : {diff}")
print("\nVERDICT:", "CLEAN -- no lookahead in signal generation" if miss == 0 and diff == 0
      else f"LOOKAHEAD -- {miss+diff}/{n} signals depend on future bars")
