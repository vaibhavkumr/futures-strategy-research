"""HOW OFTEN DO GAPS LIKE USO'S HAPPEN?

The premise "we won't have gap issues anymore" needs testing, not assuming.
Nothing in the config change removes overnight gaps -- the book still holds 25
positions through every close, and a gap is simply what those positions do
when news lands while the market is shut. Lower leverage changes the DAMAGE
per gap, not the frequency.

Measured on the actual 26-ticker universe, 2004-2026: how often does a held
name gap by more than the per-position stop, and what does that cost at 20x?
"""
import numpy as np, pandas as pd, yfinance as yf
import factor_lab as F

px = F.universe()
d = yf.download(list(px.columns), start="2004-01-01", progress=False,
                auto_adjust=True)
O, C = d["Open"].ffill(), d["Close"].ffill()
gap = (O / C.shift(1) - 1).dropna(how="all")

print("="*76)
print("OVERNIGHT GAP FREQUENCY -- 26 ETFs, 2004-2026")
print("="*76)
n_days = len(gap)
print(f"  {n_days:,} sessions, {gap.shape[1]} tickers\n")
print(f"  {'gap size':<16}{'occurrences':>13}{'per year':>11}"
      f"{'days between':>14}")
print("  "+"-"*54)
yrs = n_days/252
for thr in (0.03, 0.04, 0.06, 0.08, 0.10):
    cnt = int((gap <= -thr).sum().sum())
    # sessions where AT LEAST ONE held name gapped that much
    sess = int((gap <= -thr).any(axis=1).sum())
    print(f"  worse than -{thr*100:.0f}%{cnt:>13,}{cnt/yrs:>11.0f}"
          f"{n_days/max(sess,1):>14.0f}")

print(f"\n  A -6% gap (USO's size) somewhere in the book happens on "
      f"{(gap<=-0.06).any(axis=1).mean()*100:.1f}% of sessions")
print(f"  = roughly once every {1/max((gap<=-0.06).any(axis=1).mean(),1e-9):.0f} "
      f"trading days.")

print("\n"+"="*76)
print("WHAT ONE COSTS AT 20x  (typical held weight ~4%, top weight ~12%)")
print("="*76)
print(f"  {'gap':<10}{'at 4% weight':>16}{'at 12% weight':>16}")
print("  "+"-"*42)
for g in (0.03, 0.06, 0.10):
    print(f"  -{g*100:>3.0f}%{-g*0.04*20*100:>15.1f}%{-g*0.12*20*100:>16.1f}%")
print("\n  The per-position stop cannot prevent any of these: it fires AFTER")
print("  the open, and the gap is already in the opening price.")

print("\n"+"="*76)
print("EXPECTED GAPS DURING A 2-MONTH RUN (42 sessions)")
print("="*76)
for thr in (0.03, 0.06):
    p = (gap <= -thr).any(axis=1).mean()
    exp = p*42
    print(f"  gaps worse than -{thr*100:.0f}%: expect {exp:.1f} over the run "
          f"(P(at least one) = {1-(1-p)**42:.0%})")
