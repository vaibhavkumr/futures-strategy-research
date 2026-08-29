"""MIRROR TEST -- is the long/short gap a bug or the market?

Flip every price series upside down (h <-> -l). A correct, symmetric
implementation must then produce the mirror image of its own output:
what was a long becomes a short with an identical R.

  code symmetric  -> flipped longs match real shorts
  code asymmetric -> they don't, and the bug is mine
"""
import numpy as np, pandas as pd, tjr_spec as S

def flip(d):
    f = pd.DataFrame(index=d.index)
    f["open"], f["close"] = -d["open"], -d["close"]
    f["high"], f["low"] = -d["low"], -d["high"]
    return f

df5 = pd.read_pickle("nq_5m.pkl"); df1 = pd.read_pickle("nq_1m.pkl")
lo = max(df5.index[0], df1.index[0])
df5, df1 = df5[df5.index >= lo], df1[df1.index >= lo]
es = pd.read_csv("download/usa500idxusd-m5-bid-2022-01-01-2026-07-24.csv")
es.columns = [c.lower() for c in es.columns]
es.index = pd.to_datetime(es["timestamp"], utc=True).dt.tz_convert("America/New_York")
es = es[["open","high","low","close"]].astype(float).reindex(df5.index, method="ffill")

real = S.generate(df5, df1, corr5=es)
mirr = S.generate(flip(df5), flip(df1), corr5=flip(es))

def show(t, name):
    print(f"\n{name}: {len(t)} trades")
    for sd in ("long", "short"):
        g = t[t.side == sd]
        if len(g) < 3:
            print(f"   {sd:<6} n={len(g)}"); continue
        m = g.R.mean(); se = g.R.std(ddof=1)/np.sqrt(len(g))
        print(f"   {sd:<6} n={len(g):<4} win {(g.R>0).mean()*100:5.1f}%  "
              f"expR {m:+.3f}  t={m/se:+.2f}")

show(real, "REAL chart")
show(mirr, "FLIPPED chart")

print("\n" + "="*66)
rl, rs = real[real.side=="long"], real[real.side=="short"]
ml, ms = mirr[mirr.side=="long"], mirr[mirr.side=="short"]
print(f"real LONGS  n={len(rl):<4} expR {rl.R.mean():+.3f}   <-> "
      f"flipped SHORTS n={len(ms):<4} expR {ms.R.mean():+.3f}")
print(f"real SHORTS n={len(rs):<4} expR {rs.R.mean():+.3f}   <-> "
      f"flipped LONGS  n={len(ml):<4} expR {ml.R.mean():+.3f}")
cnt_ok = abs(len(rl)-len(ms)) <= max(3, 0.10*len(rl)) and abs(len(rs)-len(ml)) <= max(3, 0.10*len(rs))
print("\nsignal COUNTS mirror:", "yes" if cnt_ok else "NO -- code is asymmetric")
gap_real = rl.R.mean() - rs.R.mean()
gap_mirr = ml.R.mean() - ms.R.mean()
print(f"long-minus-short gap   real {gap_real:+.3f}   flipped {gap_mirr:+.3f}")
print("\nVERDICT:", "gap FLIPS with the chart -> it is the MARKET (bull trend), not a bug"
      if gap_real*gap_mirr < 0 else
      "gap does NOT flip -> the asymmetry is in MY CODE")
