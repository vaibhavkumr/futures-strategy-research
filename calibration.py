"""IS CONFIDENCE CALIBRATED? -- the one question that would change everything.

If high-confidence trades genuinely win more often, then variable sizing is not
just sensible, it MULTIPLIES the Sharpe ratio. This is the real mechanism
professionals use, and it is the one honest route to a higher Sharpe that I
have not yet tested directly.

The theory. If every trade carries identical edge mu, Sharpe is fixed. But if
edge VARIES across trades and you can see which is which beforehand, sizing in
proportion to edge gives

    Sharpe_sized^2  =  Sharpe_flat^2 * (1 + CV^2)

where CV is the coefficient of variation of the edge across trades. Dispersion
in edge that you can PREDICT is worth real Sharpe. Dispersion you cannot
predict is worth exactly nothing.

So this is entirely an empirical question, and it decomposes cleanly:

  1. MONOTONICITY  bin trades by stated confidence. Does expectancy rise
                   across bins? A calibrated signal is monotone.
  2. OUT-OF-SAMPLE does the top bin still lead on data the model never saw?
                   In-sample confidence is trivially calibrated; that is what
                   the model was fit to do.
  3. SPREAD        top-bin minus bottom-bin expectancy, with a t-stat. This is
                   the number that becomes Sharpe.

Two independent datasets:
  conf_trades.pkl   41,175 trades carrying an explicit model confidence
  live_autopsy.pkl  10,645 TJR trades -> build the CONFLUENCE score the
                    methodology itself claims predicts quality (trend
                    alignment + sweep depth + session + entry speed)
"""
import numpy as np, pandas as pd

def tstat(x):
    x=np.asarray(x,float)
    return x.mean()/(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>2 else np.nan

print("="*76)
print("DATASET 1 -- explicit model confidence, 41,175 trades")
print("="*76)
d=pd.read_pickle("conf_trades.pkl").dropna(subset=["conf","R"])
d["ts"]=pd.to_datetime(d.ts)
d=d.sort_values("ts")
cut=d.ts.quantile(0.7)
print(f"  dev: {(d.ts<cut).sum():,}   holdout: {(d.ts>=cut).sum():,}   "
      f"split at {cut:%Y-%m}\n")
d["bin"]=pd.qcut(d.conf,5,labels=False,duplicates="drop")
print(f"  {'conf bin':<12}{'n':>8}{'conf':>8}{'ALL expR':>10}{'t':>7}"
      f"{'DEV':>9}{'HOLDOUT':>10}{'t_hold':>8}")
print("  "+"-"*64)
for b in sorted(d.bin.dropna().unique()):
    s=d[d.bin==b]
    dv=s[s.ts<cut].R; hd=s[s.ts>=cut].R
    print(f"  {'Q'+str(int(b)+1):<12}{len(s):>8}{s.conf.mean():>8.3f}"
          f"{s.R.mean():>10.4f}{tstat(s.R):>7.2f}{dv.mean():>9.4f}"
          f"{hd.mean():>10.4f}{tstat(hd):>8.2f}")
top=d[d.bin==d.bin.max()]; bot=d[d.bin==d.bin.min()]
sp=top.R.mean()-bot.R.mean()
se=np.sqrt(top.R.var(ddof=1)/len(top)+bot.R.var(ddof=1)/len(bot))
print(f"\n  TOP minus BOTTOM = {sp:+.4f}R   t = {sp/se:+.2f}")
hd_t,hd_b=top[top.ts>=cut].R,bot[bot.ts>=cut].R
sp2=hd_t.mean()-hd_b.mean()
se2=np.sqrt(hd_t.var(ddof=1)/len(hd_t)+hd_b.var(ddof=1)/len(hd_b))
print(f"  HOLDOUT top-bottom = {sp2:+.4f}R   t = {sp2/se2:+.2f}")
r=np.corrcoef(d.conf,d.R)[0,1]
print(f"  correlation(confidence, R) = {r:+.4f}")

print("\n"+"="*76)
print("DATASET 2 -- TJR CONFLUENCE score, 10,645 trades")
print("="*76)
a=pd.read_pickle("live_autopsy.pkl").dropna(subset=["R"])
a["ts"]=pd.to_datetime(a.ts)
# the confluence score the methodology itself says marks an A+ setup
score=pd.Series(0,index=a.index)
if "trend_align" in a:      score+=(a.trend_align>0).astype(int)
if "sweep_depth" in a:      score+=(a.sweep_depth>a.sweep_depth.median()).astype(int)
if "bars_since_sweep" in a: score+=(a.bars_since_sweep<=a.bars_since_sweep.median()).astype(int)
if "sess" in a:             score+=a.sess.astype(str).str.contains("NY|LOND",case=False).astype(int)
if "fill_delay" in a:       score+=(a.fill_delay<=a.fill_delay.median()).astype(int)
a["score"]=score
cut2=a.ts.quantile(0.7)
print(f"  score = how many A+ conditions align (0-5)\n")
print(f"  {'score':<10}{'n':>8}{'win%':>8}{'expR':>10}{'t':>8}{'HOLDOUT expR':>14}{'t':>8}")
print("  "+"-"*58)
for s in sorted(a.score.unique()):
    g=a[a.score==s]
    if len(g)<40: continue
    hd=g[g.ts>=cut2].R
    print(f"  {int(s):<10}{len(g):>8}{(g.R>0).mean()*100:>7.1f}%{g.R.mean():>10.4f}"
          f"{tstat(g.R):>8.2f}{hd.mean() if len(hd)>10 else np.nan:>14.4f}"
          f"{tstat(hd) if len(hd)>10 else np.nan:>8.2f}")
hi=a[a.score>=4].R; lo=a[a.score<=1].R
if len(hi)>30 and len(lo)>30:
    sp3=hi.mean()-lo.mean()
    se3=np.sqrt(hi.var(ddof=1)/len(hi)+lo.var(ddof=1)/len(lo))
    print(f"\n  A+ setups (4-5) minus weak (0-1) = {sp3:+.4f}R   t = {sp3/se3:+.2f}")

print("\n"+"="*76)
print("WHAT WOULD CALIBRATION BE WORTH?")
print("="*76)
print("  Sharpe_sized = Sharpe_flat * sqrt(1 + CV^2), CV = spread of PREDICTABLE edge\n")
print(f"  {'CV of predictable edge':<28}{'Sharpe 1.14 becomes':>22}{'max growth/yr':>16}")
print("  "+"-"*64)
for cv in (0,0.5,1,2,3,5):
    s=1.14*np.sqrt(1+cv**2)
    print(f"  {cv:<28.1f}{s:>22.2f}{(np.exp(s**2/2)-1)*100:>15,.0f}%")
print(f"\n  to reach Sharpe 4.08 you need CV = {np.sqrt((4.08/1.14)**2-1):.2f}")
print("  i.e. the predictable part of your edge must vary by 3.4x its own mean,")
print("  trade to trade, and you must SEE it in advance.")
