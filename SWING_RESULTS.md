# Swing / position trading — honest evaluation

Harness: `swing.py`. Free daily data (yfinance), 18-ETF diversified universe,
2004-2026, weekly rebalance, 10bp round-trip costs, signals shifted 1 day.
DEV = 2004-2017 (contains 2008). HOLDOUT = 2018-2026, opened once.

Strategies are **documented effects**, not invented: time-series momentum
(Moskowitz/Ooi/Pedersen), cross-sectional momentum (Jegadeesh/Titman),
Donchian breakout, short-term reversal.

## Dev (2004-2017) — literature-consistent

Long lookbacks (126-252d) worked; short (21d) did not — exactly what the
momentum literature predicts. Trend strategies matched SPY's return with
**19.7% max drawdown vs SPY's 55.2%** (2008 is the reason).

## Holdout (2018-2026) — the one look

| Strategy | CAGR | Sharpe | maxDD | +months |
|---|---|---|---|---|
| xsmom(252) *(secondary pick)* | +16.6% | **1.02** | 20.7% | 64% |
| Buy & hold QQQ | +19.4% | 0.86 | 35.1% | 63% |
| Buy & hold SPY | +14.3% | 0.80 | 33.7% | 66% |
| tsmom(126) *(secondary)* | +9.3% | 0.74 | 23.6% | 66% |
| **donchian(252) — PRE-REGISTERED** | **+9.4%** | **0.71** | 21.3% | 58% |

Random-signal control: Sharpe mean 0.40, max 0.67.

**The pre-registered pick underperformed buy & hold** on both return and
Sharpe, and barely cleared the random control. That is the disciplined
answer and it goes in the record.

`xsmom(252)` beat SPY risk-adjusted (1.02 vs 0.80, much lower drawdown) and
clearly beat the random control — but it was one of three inspected, so
selection is partly in play. Regime caveat: dev contained 2008 (trend's best
environment), holdout was mostly a bull market (buy & hold's best
environment). The comparison is not regime-neutral in either direction.

## What monthly measurement actually feels like (xsmom, 103 months)

- 64% positive months → **roughly 1 month in 3 loses money**
- median +1.5%, best +12.1%, **worst -9.1%**
- 9 months worse than -5%; longest losing streak **3 months**

## Timeline to $1,000,000 at 16.6%/yr (if the edge holds)

| Start | Monthly added | Years |
|---|---|---|
| $2,000 | $0 | **40** |
| $2,000 | $500 | 21 |
| $2,000 | $1,500 | 15 |
| $10,000 | $1,500 | 14 |

## Conclusion

Strategy quality is **not** the binding constraint — contribution rate is.
$2,000 compounding alone at an excellent 16.6% still takes 40 years. The
same strategy with $1,500/month takes 15. No realistic edge closes that gap;
income does.

Momentum/trend is real and survives honest testing, but its benefit is
mostly **lower drawdown**, not higher returns — and it lagged plain buy &
hold in the 2018-2026 bull market.
