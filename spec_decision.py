"""DECISION-ONLY lookahead test.

max_hold=1 makes every trade resolve one bar after entry, so the exit
simulation needs almost no future data. Anything that still vanishes under
truncation is the ENTRY DECISION reading bars that had not happened yet.
"""
import numpy as np, pandas as pd, tjr_spec as S

df5 = pd.read_pickle("nq_5m.pkl"); df1 = pd.read_pickle("nq_1m.pkl")
lo = max(df5.index[0], df1.index[0])
df5, df1 = df5[df5.index >= lo], df1[df1.index >= lo]
es = pd.read_csv("download/usa500idxusd-m5-bid-2022-01-01-2026-07-24.csv")
es.columns = [c.lower() for c in es.columns]
es.index = pd.to_datetime(es["timestamp"], utc=True).dt.tz_convert("America/New_York")
es = es[["open","high","low","close"]].astype(float).reindex(df5.index, method="ffill")

full = S.generate(df5, df1, corr5=es, max_hold=1)
full["ts"] = pd.to_datetime(full["ts"])
print(f"decision-only full run: {len(full)} signals")
pick = full.sample(40, random_state=11).sort_values("ts")

ok = miss = mism = 0
bad = []
for _, row in pick.iterrows():
    ts = row.ts
    e5 = df5.index[min(df5.index.searchsorted(ts) + 3, len(df5) - 1)]
    e1 = df1.index[min(df1.index.searchsorted(ts) + 3, len(df1) - 1)]
    end = max(e5, e1)
    t = S.generate(df5[df5.index <= end], df1[df1.index <= end],
                   corr5=es[es.index <= end], max_hold=1)
    if t.empty:
        miss += 1; bad.append((ts, "no signals")); continue
    t["ts"] = pd.to_datetime(t["ts"])
    m = t[t.ts == ts]
    if m.empty:
        miss += 1; bad.append((ts, "ts absent"))
    elif m.iloc[0].side != row.side or abs(float(m.iloc[0].risk_atr) - float(row.risk_atr)) > 1e-9:
        mism += 1; bad.append((ts, f"{row.side}/{row.risk_atr:.4f} -> {m.iloc[0].side}/{m.iloc[0].risk_atr:.4f}"))
    else:
        ok += 1

n = ok + miss + mism
print(f"\ntruncated 3 bars past entry, {n} signals replayed")
print(f"  reproduced : {ok}\n  vanished   : {miss}\n  mismatched : {mism}")
for ts, why in bad[:8]:
    print(f"    {ts}  {why}")
print("\nVERDICT:", "ENTRY LOGIC CLEAN -- no lookahead" if miss == 0 and mism == 0
      else f"LOOKAHEAD IN ENTRY LOGIC -- {miss+mism}/{n}")
