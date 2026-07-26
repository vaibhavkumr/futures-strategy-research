# TJR-style NQ/ES backtesting bot

Codifies the **mechanical** parts of TJR's ICT playbook (liquidity sweep →
reversal fair-value-gap entry, inside NY killzones) and backtests them so you
can find out whether the strategy actually makes money **before risking a
cent.**

## Reality check (read this)
- A backtester tells you if an edge *existed in the past*. It is not a promise
  about the future, and "$200/day" is a goal, not a setting.
- The discretionary judgment TJR applies live is NOT in here — only the
  rules that can be written as code. The gap between the two is why manual
  traders often beat their own bots.
- If you plan to run this on a **prop firm** account (Topstep, Apex, etc.),
  check their rules first — many **ban full automation**.

## Setup
```bash
pip install pandas numpy
```

## Run it
Synthetic data (zero cost, proves the pipeline):
```bash
python backtest.py --days 60
```
Real data — export NQ 1m/5m candles to a CSV with columns
`timestamp,open,high,low,close,volume`, then:
```bash
python backtest.py --csv nq_5m.csv --rr 2.0
```

## Where to get real NQ data
- **Databento** — cheap pay-as-you-go CME data, best option.
- **Your broker** (Tradovate / Interactive Brokers) — export or API.
- **TradingView** — manual chart export for quick tests.

## Files
- `data.py` — loaders: CSV, Yahoo (free NQ, 60d 5m / 2yr 1h), Alpha Vantage
  (free QQQ proxy, 2+ yrs 5m), synthetic generator.
- `strategy.py` — killzones, swing structure (no-lookahead), sweep detection,
  FVG, signal generation.
- `backtest.py` — bracket-order sim with slippage + conservative intrabar
  fills; P&L / drawdown / profit-factor stats.
- `walkforward.py` — out-of-sample validation (optimize in-sample, test on
  unseen data). The anti-curve-fit test.
- `papertest.py` — forward paper-trade logger; run daily to build a real
  track record for free.

## Getting lots of data for free
- **QQQ via Alpha Vantage** (best): free key at alphavantage.co/support/#api-key,
  ~2 yrs of 5-min. QQQ tracks the same Nasdaq-100 index as NQ (a proxy, not the
  exact contract). `fetch_alphavantage("QQQ","5min",start="2023-01",api_key=...)`.
  Free tier ~25 calls/day; the loader caches each month and resumes.
- **Kaggle**: multi-year NQ/ES 1-min CSV dumps — download once, load with
  `load_csv`.
- **Databento**: ~$125 free signup credit = real NQ minute data (no card charged).
- Workflow: develop/validate on the proxy (big sample), then CONFIRM on real
  NQ (Yahoo's 60 days or Databento) before risking money.

## Tuning knobs
- `--rr` risk:reward target multiple (try 1.5–3).
- In `strategy.py`: `KILLZONES`, `swing_lb` (swing sensitivity),
  `fvg_window` (how long after a sweep an FVG still counts).
- In `backtest.py`: `POINT_VALUE` (NQ=20, MNQ=2, ES=50, MES=5), `CONTRACTS`,
  `COMMISSION`.

## Roadmap (in order — don't skip)
1. ✅ Backtester + strategy engine (this).
2. Feed real NQ data; measure honest expectancy across 6–12 months.
3. If (and only if) the edge holds: paper trade live via broker API.
4. If paper trading matches the backtest: risk tiny real size (1 MNQ = $2/pt).
5. Scale only after a long, boring track record.
