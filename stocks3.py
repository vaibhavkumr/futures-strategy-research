"""WIDER AND SMALLER -- the remaining untested lever on the stock edge.

Established so far: plain 12-month momentum on 202 large-cap US stocks gives
~21.5%/yr at Sharpe 0.90 after stripping survivorship bias, roughly double the
26-ETF book. Four documented refinements (skip-month, residual, low-vol stack,
vol-scaling) all failed to beat it once the bias control was applied.

What is still untested is the universe itself. Two reasons to expect more:

  1. SIZE. Momentum is documented as stronger in smaller, less-covered names
     -- fewer analysts, slower information diffusion, more underreaction.
     Every ticker tested so far is a mega-cap.

  2. BREADTH. Momentum's return scales with cross-sectional dispersion. Going
     26 -> 202 names roughly doubled it. 202 -> 400+ including mid and small
     caps should extend that, if dispersion really is the binding constraint.

Also tests SELECTIVITY (top 10/20/40 out of a bigger pool) and REBALANCE
FREQUENCY, neither of which was swept on the stock universe.

The drop-top-winners control is applied to everything, because in a wider
universe the survivorship problem gets WORSE, not better -- small caps that
failed are exactly the ones missing from a present-day ticker list.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import stocks as S
import stocks2 as S2

# mid and small caps to extend the large-cap list in stocks.py
SMALLER = """ETSY W CHWY CVNA RH WSM DKS FIVE BURL DECK CROX SKX ONON BIRK
YETI PLNT WING SHAK CAVA TXRH EAT BLMN CAKE PENN DKNG CZR MGM LVS WYNN NCLH
CCL RCL ABNB EXPE BKNG MAR HLT H WH CHH
ANET PSTG NTAP WDC STX SMCI DELL HPQ HPE JNPR CIEN LITE COHR VIAV AAOI
FSLR ENPH SEDG RUN NOVA PLUG BE BLDP FCEL AMRC
RIVN LCID NKLA CHPT BLNK QS MP LAC ALB SQM
UPST AFRM SOFI LC OPRT NU HOOD IBKR VIRT MKTX TW
ROKU FUBO PARA WBD LYV MSGS TKO EDR
TDOC DOCS HIMS OSCR CLOV ALHC PGNY
CRSP NTLA BEAM EDIT VERV SANA RXRX ABSI SDGR
AXON PLTR SNPS FICO MSCI VRSK EFX TRU
BLD MAS OC AZEK TREX POOL SITE WSO FAST GWW
NVT ATKR EME PWR MTZ DY IESC ROAD STRL
CELH MNST KDP STZ TAP SAM BF-B DEO
LNTH EXEL HALO INCY JAZZ NBIX SRPT UTHR
CMC NUE STLD X CLF AA CENX ATI HAYN
FANG PR MTDR SM CHRD CIVI CRC NOG"""


def load_wide():
    tickers = sorted(set(S.UNIV) | set(SMALLER.split()))
    print(f"attempting {len(tickers)} tickers...", flush=True)
    return S.load(tickers)


def line(lab, s, bench):
    st = S.stats(s)
    if not st:
        print(f"  {lab:<30} (insufficient)")
        return None
    b = S.stats(bench.reindex(pd.Series(s).dropna().index).fillna(0))
    ex = st["cagr"]-b["cagr"] if b else np.nan
    print(f"  {lab:<30}{st['cagr']:>8.2f}%{st['vol']:>7.0f}%{st['dd']:>7.0f}%"
          f"{st['sharpe']:>8.2f}{ex:>+9.2f}")
    return st


if __name__ == "__main__":
    px, vol = load_wide()
    print(f"usable: {px.shape[1]} stocks, {px.index.min():%Y-%m} -> "
          f"{px.index.max():%Y-%m}\n", flush=True)
    bench = px.pct_change().mean(axis=1).dropna()

    # median dollar volume as a size proxy, computed on the FIRST half of the
    # sample so it is not a look-ahead into which names grew
    half = len(px)//2
    dv = (px.iloc[:half]*vol.iloc[:half]).median().dropna().sort_values()
    small = list(dv.index[:len(dv)//3])
    large = list(dv.index[-len(dv)//3:])

    print("="*84)
    print("A. WIDER UNIVERSE  (top 20, weekly, 20bp, 6% stops)")
    print("="*84)
    print(f"  {'universe':<30}{'CAGR':>9}{'vol':>7}{'maxDD':>7}{'Sharpe':>8}{'vs B&H':>9}")
    print("  "+"-"*70)
    line(f"all {px.shape[1]} names", S2.backtest(px, S2.sc_mom12(px)), bench)
    for lab, sub in (("smallest third", px[small]), ("largest third", px[large])):
        b2 = sub.pct_change().mean(axis=1).dropna()
        line(f"{lab} ({len(sub.columns)})", S2.backtest(sub, S2.sc_mom12(sub)), b2)

    print("\n"+"="*84)
    print("B. SELECTIVITY -- how many names to hold from the wider pool")
    print("="*84)
    print(f"  {'top N':<30}{'CAGR':>9}{'vol':>7}{'maxDD':>7}{'Sharpe':>8}{'vs B&H':>9}")
    print("  "+"-"*70)
    for top in (10, 20, 30, 50):
        line(f"top {top}", S2.backtest(px, S2.sc_mom12(px), top=top), bench)

    print("\n"+"="*84)
    print("C. SURVIVORSHIP CONTROL -- drop the biggest full-period winners")
    print("="*84)
    total = (px.iloc[-1]/px.iloc[0] - 1).sort_values(ascending=False)
    print(f"  {'config':<30}{'CAGR':>9}{'vol':>7}{'maxDD':>7}{'Sharpe':>8}{'vs B&H':>9}")
    print("  "+"-"*70)
    for drop in (0, 30, 60):
        sub = px[list(total.index[drop:])]
        b2 = sub.pct_change().mean(axis=1).dropna()
        lab = "full" if drop == 0 else f"drop top {drop}"
        line(f"{lab}, top 20", S2.backtest(sub, S2.sc_mom12(sub), top=20), b2)

    print("\n"+"="*84)
    print("D. DEV / HOLDOUT, bias-controlled")
    print("="*84)
    sub = px[list(total.index[30:])]
    b2 = sub.pct_change().mean(axis=1).dropna()
    r = S2.backtest(sub, S2.sc_mom12(sub), top=20)
    for lab, sl in (("DEV  2010-2017", slice(None, S.DEV)),
                    ("HOLDOUT 2018-2026", slice(S.DEV, None))):
        a, b = S.stats(r.loc[sl]), S.stats(b2.loc[sl])
        if a and b:
            print(f"  {lab:<20}{a['cagr']:>8.2f}%  vs B&H {b['cagr']:>7.2f}%"
                  f"   excess {a['cagr']-b['cagr']:>+7.2f}%   Sharpe {a['sharpe']:.2f}")
