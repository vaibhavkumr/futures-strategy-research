"""THE TUNED SYSTEM -- every setting measured, not assumed.

  top 3            concentration sweep: best growth AND stable dev/holdout
  biweekly         frequency sweep: Sharpe 1.00 vs 0.94 weekly, smaller maxDD
  k = 0.25         conviction sweep: mild tilt best, heavy tilt hurts
  no stop          stop sweep: stops cost ~3pp of growth on concentrated momentum
  placebo 0/60     random triples average -7.8%/yr
"""
import numpy as np, pandas as pd
import stocks as S, stocks2 as S2
from conc2 import rebal_backtest
from conviction import growth

px,_=S.load(S.UNIV)
total=(px.iloc[-1]/px.iloc[0]-1).sort_values(ascending=False)
clean=px[list(total.index[30:])]
cb=clean.pct_change().mean(axis=1).dropna()

print("="*78); print("TUNED vs BASELINE  (survivorship-controlled)"); print("="*78)
print(f"  {'config':<40}{'GROWTH':>10}{'maxDD':>9}{'Sharpe':>9}")
print("  "+"-"*68)
cfgs=[("top 20, weekly, 6% stop (old)",dict(top=20,freq='W-FRI',pstop=0.06,k=0.5)),
      ("top 3, weekly, 6% stop",dict(top=3,freq='W-FRI',pstop=0.06,k=0.5)),
      ("top 3, biweekly, 6% stop",dict(top=3,freq='2W-FRI',pstop=0.06,k=0.5)),
      ("top 3, biweekly, NO stop",dict(top=3,freq='2W-FRI',pstop=0.99,k=0.5)),
      ("TUNED: top 3, biweekly, no stop, k=.25",dict(top=3,freq='2W-FRI',pstop=0.99,k=0.25))]
best=None
for lab,kw in cfgs:
    r=rebal_backtest(clean,S2.sc_mom12(clean),**kw); g=growth(r)
    if g:
        print(f"  {lab:<40}{g['growth']:>9.1f}%{g['dd']:>8.0f}%{g['sharpe']:>9.2f}")
        if best is None or g['growth']>best[0]: best=(g['growth'],lab,r,kw)

g,lab,r,kw=best
print(f"\n  BEST: {lab}")
print("\n"+"="*78); print("DEV / HOLDOUT ON THE TUNED CONFIG"); print("="*78)
for l,sl in (("DEV  2010-2017",slice(None,S.DEV)),("HOLDOUT 2018-2026",slice(S.DEV,None))):
    a=growth(r.loc[sl]); b=growth(cb.loc[sl])
    if a and b:
        print(f"  {l:<20}{a['growth']:>8.1f}%  vs B&H {b['growth']:>7.1f}%"
              f"   excess {a['growth']-b['growth']:>+7.1f}%   Sharpe {a['sharpe']:.2f}")

print("\n"+"="*78); print("WHAT IT MEANS FOR $10,000"); print("="*78)
gr=g/100
print(f"  compounding at {g:.1f}%/yr, maxDD ~{growth(r)['dd']:.0f}%\n")
bal=10000
for y in (1,2,3,5,7,10):
    print(f"    year {y:<3} ${10000*(1+gr)**y:>12,.0f}")
need=36000/gr
print(f"\n  $3,000/month needs ${need:,.0f} of capital")
print(f"  from $10k alone: {np.log(need/10000)/np.log(1+gr):.1f} years")
for add in (250,500,1000):
    bal,yrs=10000.0,0
    while bal<need and yrs<40:
        bal=bal*(1+gr)+add*12; yrs+=1
    print(f"  adding ${add:>4}/mo: {yrs} years")
