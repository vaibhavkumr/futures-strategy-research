"""DO THE PATTERNS PREDICT DIRECTION? No stops. No targets. No exits.

Every result today came through a stop/target simulation. If my stops are too
tight, a real edge would be destroyed by execution and show up as fair odds.
This removes execution entirely:

    after the pattern fires, does price move the predicted way, at horizons
    from 5 minutes to 4 hours?

>50% = the patterns carry directional information and my exits were hiding it.
=50% = there is nothing to hide.

Also reports MEAN MOVE in ATR, because direction and magnitude can differ.
"""
import numpy as np, pandas as pd
import confluence as C

HOR=[1,3,6,12,24,48]     # 5m bars -> 5min .. 4h

if __name__=="__main__":
    res={}
    for nm,slug in list(C.MK.items())[:4]:
        d=C.load(slug)
        o,h,l,c,a=C.prep(d)
        rows=C.signals(d)
        n=len(d)
        for name,fn in C.PATTERNS.items():
            pass
        # per-pattern directional accuracy
        for name,fn in C.PATTERNS.items():
            idx=[];sg=[]
            for i in range(60,n-60):
                s=fn(o,h,l,c,a,i)
                if s: idx.append(i); sg.append(s)
            if len(idx)<200: continue
            idx=np.array(idx); sg=np.array(sg)
            key=(name,nm)
            res[key]={}
            for hz in HOR:
                ok=idx+hz<n
                fwd=(c[idx[ok]+hz]-c[idx[ok]])
                acc=(np.sign(fwd)==sg[ok]).mean()*100
                mv=(fwd*sg[ok]/a[idx[ok]]).mean()
                res[key][hz]=(acc,mv,ok.sum())
    # aggregate across markets
    print("DIRECTIONAL ACCURACY (%), pooled across 4 markets, no exits\n")
    print(f"{'pattern':<20}" + "".join(f"{str(h*5)+'m':>9}" for h in HOR))
    print("-"*(20+9*len(HOR)))
    pats=sorted({k[0] for k in res})
    for p in pats:
        row=f"{p:<20}"
        for hz in HOR:
            accs=[res[k][hz][0] for k in res if k[0]==p]
            ns=[res[k][hz][2] for k in res if k[0]==p]
            w=np.average(accs,weights=ns) if accs else np.nan
            row+=f"{w:>9.2f}"
        print(row)
    print("\nMEAN MOVE in predicted direction (ATR units)\n")
    print(f"{'pattern':<20}" + "".join(f"{str(h*5)+'m':>9}" for h in HOR))
    print("-"*(20+9*len(HOR)))
    for p in pats:
        row=f"{p:<20}"
        for hz in HOR:
            mvs=[res[k][hz][1] for k in res if k[0]==p]
            ns=[res[k][hz][2] for k in res if k[0]==p]
            w=np.average(mvs,weights=ns) if mvs else np.nan
            row+=f"{w:>+9.3f}"
        print(row)
    print("\n  50.00% = coin flip.  0.000 ATR = no movement edge.")
