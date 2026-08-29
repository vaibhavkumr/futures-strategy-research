"""Who on eToro actually sustains >5%/month -- and what do their COPIERS get?

eToro ranks 3.6 MILLION traders. That is the survivorship engine in plain
sight: the top of a 3.6m leaderboard is, by construction, mostly luck.

The dataset has one field that settles the copy-trading question directly:
    Gain         what the TRADER made
    CopiersGain  what people COPYING them actually made
The gap between those two is the haircut, measured rather than argued.
"""
import time, requests, pandas as pd, numpy as np
H={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
URL="https://www.etoro.com/sapi/rankings/rankings"

def page(pg, period="OneYearAgo", size=100, sort="-gain"):
    for _ in range(4):
        try:
            r=requests.get(URL,params={"client_request_id":"a","period":period,
                "page":pg,"pageSize":size,"sort":sort},headers=H,timeout=60)
            if r.status_code==200: return r.json().get("Items",[])
        except Exception: pass
        time.sleep(2)
    return []

def scan(period, pages=25):
    rows=[]
    for p in range(1,pages+1):
        it=page(p,period)
        if not it: break
        rows.extend(it); time.sleep(0.4)
    return pd.DataFrame(rows)

if __name__=="__main__":
    for period,label in (("OneYearAgo","1 YEAR"),):
        df=scan(period)
        df.to_pickle("etoro_1y.pkl")
        print(f"pulled {len(df):,} top-ranked traders ({label})\n")
        d=df.copy()
        print("=== the raw leaderboard is nonsense ===")
        print(f"  top gain reported     : {d.Gain.max():,.0f}%")
        print(f"  median of top {len(d)}    : {d.Gain.median():,.1f}%")
        print(f"  traders with 0 copiers: {(d.Copiers==0).sum()} of {len(d)}")
        print()
        # credible = has copiers, real track record, sane risk
        cred=d[(d.Copiers>=10)&(d.WeeksSinceRegistration>=104)]
        print(f"=== CREDIBLE subset: >=10 copiers AND >=2yr track record ===")
        print(f"  survivors: {len(cred)} of {len(d)}")
        if len(cred):
            print(f"  median annual gain    : {cred.Gain.median():.1f}%")
            print(f"  median monthly        : {((1+cred.Gain.median()/100)**(1/12)-1)*100:.2f}%")
            print(f"  median max drawdown   : {cred.PeakToValley.median():.1f}%")
            print(f"  median profitable mo% : {cred.ProfitableMonthsPct.median():.1f}%")
        print()
        print("=== THE COPIER HAIRCUT (Gain vs CopiersGain) ===")
        h=d[(d.Copiers>=10)&(d.CopiersGain!=0)]
        if len(h):
            print(f"  n={len(h)}")
            print(f"  median trader gain    : {h.Gain.median():+.1f}%")
            print(f"  median COPIER gain    : {h.CopiersGain.median():+.1f}%")
            print(f"  median haircut        : {(h.CopiersGain-h.Gain).median():+.1f} pts")
            print(f"  copiers who did WORSE : {(h.CopiersGain<h.Gain).mean()*100:.0f}%")
