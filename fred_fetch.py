"""Free macro data from FRED -- no API key, direct CSV download.

These are DOCUMENTED equity-return predictors, and they are mechanistically
unrelated to price-based signals, which is exactly what a third and fourth
edge needs to be:

  credit spreads   Fama & French; Gilchrist & Zakrajsek -- widening spreads
                   predict lower future equity returns
  term structure   yield-curve slope predicts returns and recessions
  VIX / variance   variance risk premium (Bollerslev/Tauchen/Zhou)
  financial conds  Chicago Fed NFCI
"""
import io, sys, pandas as pd, requests
SER={"VIXCLS":"VIX","BAMLH0A0HYM2":"HY credit spread","T10Y2Y":"10Y-2Y curve",
     "BAA10Y":"Baa-10Y spread","NFCI":"financial conditions","T10Y3M":"10Y-3M curve",
     "BAMLC0A0CM":"IG credit spread","DTWEXBGS":"dollar index"}
ok={}
for s,lab in SER.items():
    try:
        r=requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}",
                       timeout=25,headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code!=200: print(f"  {s:<13}{lab:<22}HTTP {r.status_code}",flush=True); continue
        d=pd.read_csv(io.StringIO(r.text)); d.columns=["date","val"]
        d["date"]=pd.to_datetime(d["date"]); d["val"]=pd.to_numeric(d["val"],errors="coerce")
        d=d.dropna().set_index("date")["val"]
        ok[s]=d
        print(f"  {s:<13}{lab:<22}{len(d):>6} obs  {d.index.min():%Y-%m}->{d.index.max():%Y-%m}",flush=True)
    except Exception as e:
        print(f"  {s:<13}{lab:<22}ERR {str(e)[:36]}",flush=True)
if ok:
    pd.to_pickle(ok,"fred.pkl"); print(f"\nsaved {len(ok)} series (free, no key)")
