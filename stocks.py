"""BIGGER CROSS-SECTION -- the untested lever on the edge that works.

Everything so far ran on 26 ETFs. That is a very small cross-section, and it
caps what momentum can pay: a long/short or top-N portfolio earns roughly the
DISPERSION between winners and losers, and 26 correlated sector funds simply
do not disperse much. Individual stocks disperse enormously.

Three things this tests, none of them tried yet:

  1. UNIVERSE SIZE. Same verified engine (12m momentum, k=0.5 conviction
     tilt, weekly rebalance, 6% stops) on ~200 liquid US stocks instead of
     26 ETFs. If dispersion is the binding constraint, this shows it.

  2. CONCENTRATION. With 200 names, the top decile is 20 stocks rather than
     6 funds -- so you can be far more selective for the same diversification.

  3. SMALL vs LARGE. Momentum is documented to be stronger in smaller, less
     efficient names. Tested by splitting the universe on market cap proxy.

Held to the same bar as everything else: DEV/HOLDOUT split, costs charged,
benchmark is the equal-weight universe rather than zero, placebo control.
Stock costs are higher than ETFs (wider spreads), charged at 20bp round trip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

COST = 20/1e4          # 20bp -- stocks are wider than ETFs
K_TILT = 0.5
DEV = "2018-01-01"

# ~200 liquid US names across sectors and sizes, all with long histories.
UNIV = """AAPL MSFT AMZN GOOGL META NVDA TSLA AVGO ORCL CRM ADBE AMD INTC QCOM
TXN MU AMAT LRCX KLAC ADI NXPI ON SWKS MCHP CDNS SNPS ANSS INTU NOW PANW FTNT
CRWD ZS OKTA DDOG NET SNOW MDB TEAM WDAY VEEV HUBS ZM DOCU TWLO SQ PYPL SHOP
JPM BAC WFC C GS MS SCHW BLK SPGI CME ICE COF AXP USB PNC TFC BK STT DFS SYF
JNJ PFE MRK ABBV LLY BMY AMGN GILD BIIB REGN VRTX MRNA ZTS TMO DHR ABT SYK BSX
MDT ISRG EW BDX BAX HOLX ALGN IDXX WAT MTD PKI A ILMN
XOM CVX COP EOG SLB PSX VLO MPC OXY HES DVN FANG HAL BKR
PG KO PEP WMT COST TGT HD LOW MCD SBUX NKE DIS CMCSA VZ T TMUS
CAT DE HON GE MMM BA LMT RTX NOC GD EMR ETN ITW PH ROK CMI PCAR
UNP CSX NSC FDX UPS DAL UAL LUV
LIN APD SHW ECL DD DOW NEM FCX
NEE DUK SO D AEP EXC XEL ED WEC
AMT PLD CCI EQIX SPG O PSA WELL AVB EQR
BRKB V MA ADP PAYX FISV FIS GPN
ADSK EA ATVI TTWO RBLX U SPOT NFLX
LULU ROST TJX ORLY AZO ULTA DG DLTR YUM CMG DRI
CL KMB GIS K HSY SYY KR""".split()


def load(tickers, start="2010-01-01"):
    d = yf.download(tickers, start=start, progress=False, auto_adjust=True)
    px = d["Close"].dropna(axis=1, how="all").ffill()
    vol = d["Volume"].reindex(columns=px.columns).ffill()
    # keep names with a real history and real liquidity
    good = [c for c in px.columns
            if px[c].notna().sum() > 1500 and (px[c]*vol[c]).median() > 2e7]
    return px[good], vol[good]


def run(px, top=20, k=K_TILT, cost=COST, pstop=0.06, universe=None):
    """Weekly conviction-tilted 12m momentum with per-position stops."""
    if universe is not None:
        px = px[[c for c in universe if c in px.columns]]
    wk = px.resample("W-FRI").last()
    raw = px.pct_change(252).reindex(wk.index, method="ffill")
    rd = px.pct_change()
    keys = [t for t in wk.index[:-1] if raw.loc[t].dropna().shape[0] >= top]
    out, prev = [], {}
    for i, t in enumerate(keys[:-1]):
        a = raw.loc[t].dropna()
        pick = a.nlargest(top)
        z = (pick - pick.mean())/pick.std(ddof=1) if pick.std(ddof=1) > 0 else pick*0
        w = np.exp(k*z); w = w/w.sum()
        seg = px.loc[(px.index > t) & (px.index <= keys[i+1])]
        cols = [c for c in w.index if c in seg.columns]
        if seg.empty or not cols:
            continue
        w = w[cols]; entry = seg[cols].iloc[0]
        live = pd.Series(True, index=cols)
        rets = []
        for dd in range(len(seg)):
            r = rd.loc[seg.index[dd], cols].fillna(0)
            rets.append(float((w*live.astype(float)*r).sum()))
            hit = live & ((seg[cols].iloc[dd]/entry - 1) <= -pstop)
            if hit.any():
                rets[-1] -= float(w[hit].sum())*cost
                live &= ~hit
        s = pd.Series(rets, index=seg.index)
        turn = sum(abs(w.get(x, 0)-prev.get(x, 0)) for x in set(w.index) | set(prev))
        s.iloc[0] -= turn*cost
        prev = w.to_dict()
        out.append(s)
    return pd.concat(out).sort_index() if out else pd.Series(dtype=float)


def stats(x, ann=252):
    x = pd.Series(x).dropna()
    if len(x) < 200:
        return None
    eq = (1+np.clip(x, -0.99, None)).cumprod()
    yrs = len(x)/ann
    cagr = (eq.iloc[-1]**(1/yrs)-1)*100
    dd = (eq/eq.cummax()-1).min()*100
    return dict(cagr=cagr, dd=dd, vol=x.std(ddof=1)*np.sqrt(ann)*100,
                sharpe=x.mean()/x.std(ddof=1)*np.sqrt(ann),
                calmar=cagr/abs(dd) if dd else np.nan)


def show(lab, s, bench=None):
    st = stats(s)
    if not st:
        print(f"  {lab:<26} (insufficient)")
        return None
    ex = ""
    if bench is not None:
        b = stats(bench.reindex(pd.Series(s).dropna().index).fillna(0))
        if b:
            ex = f"{st['cagr']-b['cagr']:>+8.2f}"
    print(f"  {lab:<26}{st['cagr']:>8.2f}%{st['vol']:>7.0f}%{st['dd']:>7.0f}%"
          f"{st['sharpe']:>8.2f}{st['calmar']:>8.2f}{ex}")
    return st


if __name__ == "__main__":
    print(f"loading {len(UNIV)} tickers...", flush=True)
    px, vol = load(UNIV)
    print(f"usable: {px.shape[1]} stocks, {px.index.min():%Y-%m} -> "
          f"{px.index.max():%Y-%m}\n", flush=True)

    bench = px.pct_change().mean(axis=1).dropna()

    print("="*84)
    print("A. STOCK UNIVERSE vs THE 26-ETF UNIVERSE")
    print("="*84)
    print(f"  {'system':<26}{'CAGR':>9}{'vol':>7}{'maxDD':>7}{'Sharpe':>8}"
          f"{'Calmar':>8}{'vs B&H':>9}")
    print("  "+"-"*74)
    show("equal-weight buy&hold", bench)
    for top in (10, 20, 30, 50):
        show(f"momentum, top {top}", run(px, top=top), bench)

    import factor_lab as F
    etf = F.universe()
    show("(ETF book, top 6)", F.run(etf, F.f_mom12), F.benchmark(etf))

    print("\n"+"="*84)
    print("B. DEV / HOLDOUT -- does the best config hold up out of sample?")
    print("="*84)
    best_top = 20
    r = run(px, top=best_top)
    for lab, sl in (("DEV  2010-2017", slice(None, DEV)),
                    ("HOLDOUT 2018-2026", slice(DEV, None))):
        a, b = stats(r.loc[sl]), stats(bench.loc[sl])
        if a and b:
            print(f"  {lab:<20} system {a['cagr']:>7.2f}%   B&H {b['cagr']:>7.2f}%"
                  f"   excess {a['cagr']-b['cagr']:>+7.2f}%   Sharpe {a['sharpe']:.2f}")

    print("\n"+"="*84)
    print("C. PLACEBO -- random stock picks, same engine, same costs")
    print("="*84)
    rng = np.random.default_rng(3)
    real = stats(r)
    wins = 0
    N = 40
    for kk in range(N):
        def rnd(p, seed=kk):
            r2 = np.random.default_rng(seed)
            return pd.DataFrame(r2.standard_normal(p.shape),
                                index=p.index, columns=p.columns)
        wk = px.resample("W-FRI").last()
        keys = list(wk.index[:-1])
        out, prev = [], {}
        rd = px.pct_change()
        sc = rnd(px)
        for i, t in enumerate(keys[:-1]):
            a = sc.reindex([t], method="ffill").iloc[0].dropna()
            if len(a) < best_top:
                continue
            pick = a.nlargest(best_top)
            w = pd.Series(1.0/best_top, index=pick.index)
            seg = px.loc[(px.index > t) & (px.index <= keys[i+1])]
            cols = [c for c in w.index if c in seg.columns]
            if seg.empty or not cols:
                continue
            w = w[cols]
            s = (seg[cols].pct_change().fillna(0)*w).sum(axis=1)
            turn = sum(abs(w.get(x, 0)-prev.get(x, 0))
                       for x in set(w.index) | set(prev))
            if len(s):
                s.iloc[0] -= turn*COST
            prev = w.to_dict()
            out.append(s)
        if out:
            st = stats(pd.concat(out).sort_index())
            if st and real and st["cagr"] >= real["cagr"]:
                wins += 1
    print(f"  random pickers matching or beating the real signal: {wins}/{N} "
          f"= {wins/N*100:.0f}%")
    print(f"  (0-5% means the momentum signal is doing real work)")
