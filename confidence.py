"""Does model CONFIDENCE predict trade quality?

If it does, sizing by confidence is real risk management. If expectancy is
flat across confidence, sizing is just a louder version of the same bet --
and if expectancy is negative everywhere, sizing accelerates the loss.

The decisive question is not "does confidence help?" but "is the TOP
confidence bucket positive AFTER costs?" If no bucket clears zero, there is
nothing for sizing to amplify.
"""
import numpy as np, pandas as pd
import edge500 as E

if __name__ == "__main__":
    data, preds = {}, {}
    for name in E.MK:
        data[name] = E.build(name)
        preds[name] = E.walk_forward(data[name])
        print(f"  {name:<9} {len(preds[name]):>7,} predictions")
    trades = {}
    for name in E.MK:
        tr = E.simulate(data[name], preds[name]["pred"], thresh=0.0)
        tr["mkt"] = name
        tr["conf"] = tr.pred.abs()
        trades[name] = tr
    T = pd.concat(trades.values(), ignore_index=True)
    T.to_pickle("conf_trades.pkl")

    dev = T[T.mkt.isin(E.DEV_MK)]
    hold = T[T.mkt.isin(E.HOLD_MK)]
    print(f"\nDEV {len(dev):,} trades   HOLDOUT {len(hold):,} trades")

    for lab, D in (("DEV", dev), ("HOLDOUT", hold)):
        print("\n" + "=" * 70)
        print(f"{lab}: expectancy by CONFIDENCE decile (costs already included)")
        print("=" * 70)
        D = D.copy()
        D["dec"] = pd.qcut(D.conf, 10, labels=False, duplicates="drop")
        print(f"{'decile':<9}{'n':>7}{'conf range':>18}{'win%':>8}{'expR':>9}{'t':>8}")
        print("-" * 62)
        for d, g in D.groupby("dec"):
            m = g.R.mean(); se = g.R.std(ddof=1)/np.sqrt(len(g))
            print(f"{int(d)+1:<9}{len(g):>7}{g.conf.min():>8.2f}-{g.conf.max():<9.2f}"
                  f"{(g.R>0).mean()*100:>8.1f}{m:>9.3f}{m/se:>8.2f}")
        # is confidence informative at all?
        r = np.corrcoef(D.conf, D.R)[0, 1]
        t = r*np.sqrt(len(D)-2)/np.sqrt(max(1-r**2, 1e-12))
        print(f"\n  corr(confidence, R) = {r:+.4f}  t={t:+.2f}"
              f"   -> {'informative' if abs(t)>2 else 'NOT informative'}")
        top = D[D.dec == D.dec.max()]
        m, se = top.R.mean(), top.R.std(ddof=1)/np.sqrt(len(top))
        print(f"  TOP decile: expR {m:+.3f}  t={m/se:+.2f}  "
              f"CI[{m-1.96*se:+.3f},{m+1.96*se:+.3f}]")

    print("\n" + "=" * 70)
    print("SIZING TEST -- risk proportional to confidence vs flat risk")
    print("=" * 70)
    h = hold.copy()
    flat = h.R.values
    w = (h.conf / h.conf.mean()).clip(0, 3).values      # confidence weighting
    sized = flat * w
    for lab, x in (("flat 1 unit per trade", flat),
                   ("risk scaled by confidence", sized)):
        m, se = x.mean(), x.std(ddof=1)/np.sqrt(len(x))
        print(f"  {lab:<28} mean {m:+.4f}R  t={m/se:+6.2f}  "
              f"total {x.sum():+8.1f}R")
    print("\n  Scaling a negative expectancy by confidence makes the loss")
    print("  BIGGER, not smaller, unless the top bucket is genuinely positive.")
