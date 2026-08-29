"""Find the top DAY TRADERS on eToro -- not buy-and-hold investors.

The previous scan found people holding 35 semiconductor stocks. That is not
day trading, and their returns turned out to be sector beta (they lost to
SOXX). To isolate actual day traders, filter on BEHAVIOUR rather than return:

    trades per active week -- a buy-and-hold investor makes a few trades a
    month. A day trader makes many per week.

Then ask the only question that matters: what do the best of them actually
return over a real track record, and what do their copiers get?
"""
import time, requests, pandas as pd, numpy as np
H={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
URL="https://www.etoro.com/sapi/rankings/rankings"

def page(pg,sort,period="OneYearAgo",size=100):
    for _ in range(4):
        try:
            r=requests.get(URL,params={"client_request_id":"a","period":period,
                "page":pg,"pageSize":size,"sort":sort},headers=H,timeout=60)
            if r.status_code==200: return r.json().get("Items",[])
        except Exception: pass
        time.sleep(2)
    return []

if __name__=="__main__":
    rows=[]
    for sort in ("-copiers","-gain","-trades","-winratio"):
        for p in range(1,16):
            it=page(p,sort)
            if not it: break
            rows.extend(it); time.sleep(0.35)
        print(f"  {sort}: {len(rows)} cumulative", flush=True)
    d=pd.DataFrame(rows).drop_duplicates("CustomerId")
    d.to_pickle("etoro_all.pkl")
    print(f"\nunique traders: {len(d):,}")

    d["tpw"]=d.Trades/d.ActiveWeeks.replace(0,np.nan)
    d["mo"]=(1+d.Gain/100)**(1/12)-1
    d["yrs"]=d.WeeksSinceRegistration/52
    print(f"\ntrades/week distribution: median {d.tpw.median():.1f}  "
          f"90th {d.tpw.quantile(.9):.0f}  max {d.tpw.max():.0f}")

    # DAY TRADERS: >=20 trades/week, real history, actually copyable
    dt=d[(d.tpw>=20)&(d.yrs>=2)&(d.Trades>=500)]
    print(f"\n{'='*88}")
    print(f"ACTUAL DAY TRADERS (>=20 trades/wk, >=2yr history, >=500 trades): "
          f"{len(dt)} of {len(d):,}")
    print("="*88)
    if len(dt):
        print(f"  median 1yr gain      : {dt.Gain.median():+.1f}%")
        print(f"  median monthly       : {dt.mo.median()*100:+.2f}%")
        print(f"  median max drawdown  : {dt.PeakToValley.median():.1f}%")
        print(f"  median profitable mo%: {dt.ProfitableMonthsPct.median():.1f}%")
        print(f"  PROFITABLE at all    : {(dt.Gain>0).mean()*100:.0f}%")
        print(f"  beat 5%/month        : {(dt.mo>0.05).sum()} of {len(dt)}")
        print()
        best=dt.sort_values("Gain",ascending=False).head(12)
        print(f"{'trader':<18}{'%/mo':>7}{'1yr':>9}{'tr/wk':>7}{'maxDD':>8}"
              f"{'win%':>7}{'copiers':>8}{'copierGain':>12}")
        print("-"*78)
        for _,r in best.iterrows():
            print(f"{r.UserName[:17]:<18}{r.mo*100:>6.1f}%{r.Gain:>8.1f}%"
                  f"{r.tpw:>7.0f}{r.PeakToValley:>7.1f}%{r.WinRatio:>6.1f}%"
                  f"{int(r.Copiers):>8}{r.CopiersGain:>11.1f}%")
