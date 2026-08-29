"""SIGNAL LAB II -- expanding the hunt for signals 3 and 4.

Round one tested 10 candidates and found 2 usable (calendar, vol_mr), which
together halved max drawdown from -10.0% to -5.1% at the same Sharpe. That
proved the mechanism. Now: more mechanisms, same discipline.

New families here, deliberately NOT variations on price shape:
  DRAWDOWN RECOVERY   buy after an N-day decline (documented rebound)
  CONSECUTIVE DAYS    streak effects
  NEW HIGH / LOW      proximity to 52-week extremes
  CROSS-SECTIONAL     long the strongest index vs the weakest
  DISPERSION          indices diverge, then reconverge
  VOL TERM STRUCTURE  short-horizon vol vs long-horizon vol
  MONTH SEASONALITY   sell-in-May and month-of-year effects
  WEEK OF MONTH       payroll and options cycles
  LAST HOUR           closing auction / MOC imbalance proxy
  BIG MOVE FOLLOW     after an outsized day, drift or reverse
  VOLUME SURGE        high-volume days as an information signal

Same bar as round one: positive on DEV and HOLDOUT, 3+ of 4 markets, costs
included, and then checked for correlation against the signals already held.
With ~20 candidates I expect 1-2 to pass by luck, so survivors are checked
against a shuffled-label null before being taken seriously.
"""
import numpy as np, pandas as pd
import signal_lab as L

def s_dd_recovery(D, look=5, thr=-150):
    """Buy after a sharp N-day decline."""
    dec=(D.c/D.c.shift(look)-1)*1e4
    return pd.Series(np.where(dec.shift(1)<thr,1.0,0.0),index=D.index)

def s_streak(D, n=3, follow=False):
    """After n consecutive down days: fade (buy) or follow."""
    up=(D.c>D.c.shift(1)).astype(int)
    run_dn=(up.rolling(n).sum()==0).shift(1).fillna(False)
    run_up=(up.rolling(n).sum()==n).shift(1).fillna(False)
    s=np.zeros(len(D))
    s[run_dn.values]= 1.0 if not follow else -1.0
    s[run_up.values]=-1.0 if not follow else  1.0
    return pd.Series(s,index=D.index)

def s_near_high(D, look=252, thr=0.98):
    """Near 52-week highs -> momentum continuation."""
    hh=D.c.rolling(look).max()
    return pd.Series(np.where((D.c/hh).shift(1)>thr,1.0,0.0),index=D.index)

def s_near_low(D, look=252, thr=1.03):
    lo=D.c.rolling(look).min()
    return pd.Series(np.where((D.c/lo).shift(1)<thr,1.0,0.0),index=D.index)

def s_vol_term(D, s_=5, l_=60):
    """Short vol below long vol = calm regime -> long."""
    sv=D.cc.rolling(s_).std(); lv=D.cc.rolling(l_).std()
    return pd.Series(np.where((sv/lv).shift(1)<0.8,1.0,0.0),index=D.index)

def s_vol_term_hi(D, s_=5, l_=60):
    sv=D.cc.rolling(s_).std(); lv=D.cc.rolling(l_).std()
    return pd.Series(np.where((sv/lv).shift(1)>1.3,1.0,0.0),index=D.index)

def s_month_seas(D):
    """Nov-Apr long (the documented seasonal half)."""
    m=D.index.month
    return pd.Series(np.where(np.isin(m,[11,12,1,2,3,4]),1.0,0.0),index=D.index)

def s_week_of_month(D, wk=1):
    """Payroll week / first week of month."""
    w=((D.index.day-1)//7)+1
    return pd.Series(np.where(w==wk,1.0,0.0),index=D.index)

def s_bigmove_fade(D, thr=200):
    prev=D.r.shift(1)
    return pd.Series(np.where(np.abs(prev)>thr,-np.sign(prev),0.0),index=D.index).fillna(0)

def s_bigmove_go(D, thr=200):
    prev=D.r.shift(1)
    return pd.Series(np.where(np.abs(prev)>thr,np.sign(prev),0.0),index=D.index).fillna(0)

def s_volsurge(D, look=20, mult=1.5):
    vm=D.v.rolling(look).mean()
    return pd.Series(np.where((D.v/vm).shift(1)>mult,1.0,0.0),index=D.index).fillna(0)

def s_volquiet(D, look=20, mult=0.7):
    vm=D.v.rolling(look).mean()
    return pd.Series(np.where((D.v/vm).shift(1)<mult,1.0,0.0),index=D.index).fillna(0)

NEW={
 "dd_recovery":  s_dd_recovery,
 "streak_fade":  lambda D: s_streak(D,3,False),
 "streak_go":    lambda D: s_streak(D,3,True),
 "near_52w_high":s_near_high,
 "near_52w_low": s_near_low,
 "vol_term_calm":s_vol_term,
 "vol_term_hi":  s_vol_term_hi,
 "month_seas":   s_month_seas,
 "week1":        lambda D: s_week_of_month(D,1),
 "week3":        lambda D: s_week_of_month(D,3),
 "bigmove_fade": s_bigmove_fade,
 "bigmove_go":   s_bigmove_go,
 "vol_surge":    s_volsurge,
 "vol_quiet":    s_volquiet,
}

if __name__=="__main__":
    D={nm:L.daily(sl) for nm,sl in L.MK.items()}
    print(f"{len(NEW)} new candidates\n")
    print(f"{'signal':<16}{'DEV bp':>9}{'t':>7}{'HOLD bp':>9}{'t':>7}{'mkts+':>7}{'':>10}")
    print("-"*66)
    keep={}
    for name,fn in NEW.items():
        P=L.evaluate(name,fn,D)
        r=L.report(P,name)
        if not r:
            print(f"{name:<16}  (too few)"); continue
        ok=(r["dev_m"]>0 and r["hold_m"]>0 and r["mkts_pos"]>=3)
        if ok: keep[name]=r["series"]
        print(f"{name:<16}{r['dev_m']:>9.2f}{r['dev_t']:>7.2f}"
              f"{r['hold_m']:>9.2f}{r['hold_t']:>7.2f}{r['mkts_pos']:>5}/4"
              f"{'  SURVIVES' if ok else '':>10}")
    print(f"\nSURVIVORS: {list(keep) if keep else 'none'}")
    if keep:
        base={k:L.evaluate(k,L.SIGNALS[k],D).mean(axis=1).dropna()
              for k in ("calendar","vol_mr")}
        allsig={**base,**keep}
        C=pd.DataFrame(allsig).corr()
        print("\nCORRELATION vs signals already held:")
        print(C.round(3).to_string())
