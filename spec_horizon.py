"""How far into the FUTURE must data extend before a signal appears?

Truncate at (entry bar + k). The exit simulation legitimately needs bars
after entry, so small k failing is a test artifact. But the ENTRY DECISION
(ts, side, stop) must not need bars beyond that -- if it only reproduces at
large k, the entry logic is reading the future.
"""
import numpy as np, pandas as pd, tjr_spec as S

df5 = pd.read_pickle("nq_5m.pkl"); df1 = pd.read_pickle("nq_1m.pkl")
lo = max(df5.index[0], df1.index[0])
df5, df1 = df5[df5.index >= lo], df1[df1.index >= lo]
es = pd.read_csv("download/usa500idxusd-m5-bid-2022-01-01-2026-07-24.csv")
es.columns = [c.lower() for c in es.columns]
es.index = pd.to_datetime(es["timestamp"], utc=True).dt.tz_convert("America/New_York")
es = es[["open","high","low","close"]].astype(float).reindex(df5.index, method="ffill")

full = S.generate(df5, df1, corr5=es); full["ts"] = pd.to_datetime(full["ts"])
pick = full.sample(25, random_state=7).sort_values("ts")

for k in (0, 6, 24, 60, 300):
    ok = miss = mism = 0
    for _, row in pick.iterrows():
        ts = row.ts
        end5 = df5.index[min(df5.index.searchsorted(ts) + k, len(df5) - 1)]
        end1 = df1.index[min(df1.index.searchsorted(ts) + k * 5, len(df1) - 1)]
        end = max(end5, end1)
        c5, c1, ce = df5[df5.index <= end], df1[df1.index <= end], es[es.index <= end]
        t = S.generate(c5, c1, corr5=ce)
        if t.empty:
            miss += 1; continue
        t["ts"] = pd.to_datetime(t["ts"])
        m = t[t.ts == ts]
        if m.empty:
            miss += 1
        elif m.iloc[0].side != row.side or abs(float(m.iloc[0].risk_atr) - float(row.risk_atr)) > 1e-9:
            mism += 1
        else:
            ok += 1
    print(f"k={k:<4} bars of future data:  reproduced {ok:2d}   vanished {miss:2d}   mismatched {mism:2d}")
