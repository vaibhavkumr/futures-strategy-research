# Findings — TJR/ICT mechanical strategy, honest evaluation

Data: Dukascopy 5-min index CFDs, 2022-01 → 2026-07 (~91k RTH bars/market,
1,174 sessions, includes the 2022 bear market). Free, no account.
Verified the CFD feed's microstructure matches real NQ futures closely
(FVG rate 22.2% vs 23.7%, median gap/ATR 0.259 vs 0.274) — it's a fair proxy.

## Bugs found and fixed (these mattered more than any tuning)

| Bug | Effect when fixed |
|---|---|
| Swing points confirmed using future bars | removed lookahead in structure |
| **Entry filled on the signal bar's own low** | **win rate 58.9% → 38.0%, PF 2.72 → 1.16** |
| Degenerate ~0-risk stops (1e-9) | removed infinite-R artifacts |

The second bug alone was manufacturing ~86% of the apparent profit. Any
backtest that fills within the signal bar is lying to you.

## Baseline (TJR-style 2:1 targets)

576 trades, 38.0% win, **+0.10R**, PF 1.16, t=1.68,
95% CI **[-0.02R, +0.23R] — includes zero.** Not statistically significant.

## Search (230 configs, dev = 2022-2024 only)

Clear monotonic finding: **lower R:R is dramatically better.** Not a single
lucky config — a broad plateau across every `fvg_window` value.

Best/robust config: `rr=1.0, swing_lb=2, fvg_window=12, zone=ny_am`.

| Set | n | win | expR | PF | ddR | t |
|---|---|---|---|---|---|---|
| Dev 2022-24 (searched) | 416 | 64.4% | +0.261 | 1.69 | 8.6 | 5.41 |
| **Holdout 2025-26 (unseen)** | **206** | **60.2%** | **+0.184** | **1.44** | **12.6** | **2.64** |

Random-side control on holdout: mean -0.048R; our signal beat **10/10**
random controls → the direction logic adds real value.
Slippage tolerance: edge survives to ~5pts of stop slippage (realistic NQ 1-3).

**Interpretation:** the mechanical edge is a *short-horizon bounce* after the
sweep. Price reliably travels ~1R; the 2:1/3:1 targets give it back. The
discretion is plausibly what lets a human hold for the bigger targets.

## Cross-market validation (locked config, zero retuning) — THE CAVEAT

| Market | n | win | expR | PF | t |
|---|---|---|---|---|---|
| Nasdaq-100 (developed on) | 622 | 63.0% | +0.235 | 1.60 | 5.94 |
| Dow 30 (new) | 592 | 55.9% | +0.102 | 1.22 | 2.46 |
| **S&P 500 (new)** | 581 | 54.4% | **-0.038** | 0.93 | -0.84 |
| **DAX (new)** | 593 | 52.1% | **-0.002** | 1.00 | -0.05 |

**Only the market it was developed on shows a strong edge.** Dow is weakly
positive; S&P and DAX are flat. All four have win rate >50% at 1:1, so there
may be a small genuine directional effect — but in 3 of 4 markets it is too
small to clear costs. This is the signature of an edge that is largely
**instrument-specific and partly a product of having been developed on
Nasdaq**, not a universal market law. Real NQ futures (Yahoo, n=26) was +0.07R —
too small a sample to confirm or refute (CI ≈ ±0.39R).

Per-year on Nasdaq: 2022 +0.44, 2023 +0.19, 2024 +0.15, 2025 +0.25,
**2026 +0.07 (t=0.63)** — possible decay, or just a small sample.

## What this means in dollars (Nasdaq, the best case)

0.235R × 0.5 trades/day = **0.12R/day**. Median risk ~50 pts ⇒ 1R ≈ $100
per MNQ contract.

| Account (risking 1%/trade) | realistic $/day |
|---|---|
| $2,000 | ~$2 |
| $5,000 | ~$6 |
| $10,000 | ~$12 |
| **~$45,000** | **~$200** |

**$200/day requires risking ~$1,700 per trade and surviving a ~$21,000
drawdown** — i.e. a $40-60k account, *assuming the edge is real and
persists*. It cannot be reached by scaling a small account; leverage that
big on a small account gets liquidated by the 12.6R drawdown first.

## Verdict

**Not tradeable.** The evidence for and against:

FOR: significant out-of-sample in time (t=2.64 on unseen 2025-26), beat
10/10 random-side controls, survives realistic slippage, >50% win rate on
all four markets, positive every year.

AGAINST: **fails on 2 of 3 markets it wasn't developed on** (S&P -0.038,
DAX -0.002), weak and insignificant in 2026 (+0.07R, t=0.63), unconfirmed
on real NQ futures, and the strong result appears only on the instrument
used for development — the classic fingerprint of instrument-level
overfitting.

Most likely truth: a small real short-horizon reversion effect exists after
liquidity sweeps, but it is weak, market-specific, possibly decaying, and
too small to support meaningful income without a large account.

**Do not trade this with real money.** The right next step is forward paper
testing (papertest.py), which costs nothing and generates the only evidence
that can't be overfit.
