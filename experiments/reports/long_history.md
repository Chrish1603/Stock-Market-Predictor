# long_history

**Question:** Do the risk results hold up over 30 years, including the 2000-02 and 2008 crashes?

Run 2026-11-03 · SPY only, data from 1993-01-29 · out-of-sample forecasts from 1997-05-29 to 2026-10-30

The live system's data starts in 2010, so its backtests miss the 2000-02 and 2008 bear markets. This rebuilds the SPY risk models alone from 1993-01-29, with 8 walk-forward periods. It has no cross-sectional ranks and no breadth input (both need the 2010+ stock universe).

## Volatility forecasts

Mean walk-forward R² of log realised volatility (5 and 20 days): **0.370**.

| Alternative | Its R² | Δ chosen − alternative | 95% CI | p (Holm) | Better / worse periods | Verdict |
|---|---|---|---|---|---|---|
| persistence | 0.114 | +0.256 | [+0.169, +0.340] | < 0.001 | 15 / 1 | better |
| har | 0.262 | +0.108 | [+0.051, +0.156] | < 0.001 | 15 / 1 | better |

**5-day R² by test period** (chosen model: lightgbm)

| Period | lightgbm | persistence | har |
|---|---|---|---|
| 1997-05 – 2000-12 | 0.020 | -0.079 | -0.044 |
| 2000-12 – 2004-07 | 0.465 | 0.277 | 0.398 |
| 2004-07 – 2008-01 | 0.331 | 0.164 | 0.216 |
| 2008-01 – 2011-08 | 0.520 | 0.437 | 0.487 |
| 2011-08 – 2015-02 | 0.498 | 0.094 | 0.276 |
| 2015-02 – 2018-09 | 0.383 | 0.148 | 0.277 |
| 2018-09 – 2022-03 | 0.586 | 0.261 | 0.365 |
| 2022-03 – 2025-10 | 0.511 | 0.194 | 0.325 |

**20-day R² by test period** (chosen model: lightgbm)

| Period | lightgbm | persistence | har |
|---|---|---|---|
| 1997-05 – 2000-12 | -0.319 | -0.433 | -0.383 |
| 2000-12 – 2004-06 | 0.566 | 0.291 | 0.506 |
| 2004-06 – 2008-01 | 0.393 | 0.036 | 0.214 |
| 2008-01 – 2011-07 | 0.412 | 0.422 | 0.549 |
| 2011-07 – 2015-01 | 0.412 | -0.084 | 0.276 |
| 2015-01 – 2018-08 | 0.292 | -0.039 | 0.189 |
| 2018-08 – 2022-02 | 0.329 | -0.069 | 0.178 |
| 2022-02 – 2025-09 | 0.523 | 0.203 | 0.364 |

## Volatility targeting

| Period | Strategy | CAGR | Volatility | Sharpe | Max drawdown | Avg holding |
|---|---|---|---|---|---|---|
| All (1997–2026) | Buy & hold | 9.6% | 19.3% | 0.57 | -55.2% | 100% |
| All (1997–2026) | **Vol target, trailing vol** (default) | 8.0% | 13.5% | 0.64 | -43.5% | 85% |
| All (1997–2026) | Vol target, model forecast | 7.8% | 14.6% | 0.59 | -45.5% | 90% |
| All (1997–2026) | Model forecast + drop filter | 7.2% | 14.0% | 0.57 | -43.9% | 89% |
| Before 2010 (1997–2009) | Buy & hold | 4.1% | 22.1% | 0.29 | -55.2% | 100% |
| Before 2010 (1997–2009) | **Vol target, trailing vol** (default) | 3.2% | 14.4% | 0.29 | -43.5% | 80% |
| Before 2010 (1997–2009) | Vol target, model forecast | 2.9% | 16.2% | 0.26 | -45.5% | 86% |
| Before 2010 (1997–2009) | Model forecast + drop filter | 2.5% | 15.3% | 0.24 | -43.9% | 85% |
| 2010 on (2010–2026) | Buy & hold | 13.9% | 17.0% | 0.85 | -33.7% | 100% |
| 2010 on (2010–2026) | **Vol target, trailing vol** (default) | 11.8% | 12.7% | 0.94 | -18.9% | 89% |
| 2010 on (2010–2026) | Vol target, model forecast | 11.7% | 13.2% | 0.90 | -20.4% | 93% |
| 2010 on (2010–2026) | Model forecast + drop filter | 10.9% | 13.0% | 0.86 | -21.7% | 92% |

**`trailing_vol_target` vs other strategies, whole period** (Sharpe difference, block bootstrap):

| Alternative | Its Sharpe | Δ Sharpe | 95% CI | p (Holm) | Verdict |
|---|---|---|---|---|---|
| buy_hold | 0.57 | +0.07 | [-0.05, +0.21] | 0.337 | no clear difference |
| model_vol_target | 0.59 | +0.05 | [-0.02, +0.13] | 0.337 | no clear difference |
| model_vol_target_drop_filter | 0.57 | +0.07 | [-0.00, +0.15] | 0.165 | no clear difference |

**Through each crisis** (return / worst drawdown, peak to trough of the S&P 500):

| Crisis | Dates | Buy & hold | Vol target, trailing vol | Vol target, model forecast | Model forecast + drop filter | Avg holding (default) |
|---|---|---|---|---|---|---|
| 1998 LTCM / Russia | 1998-07-17 – 1998-10-08 | -17% / -19% | -13% / -14% | -15% / -17% | -20% / -20% | 59% |
| Dot-com bust | 2000-03-24 – 2002-10-09 | -46% / -47% | -41% / -42% | -43% / -43% | -43% / -43% | 72% |
| Financial crisis | 2007-10-09 – 2009-03-09 | -53% / -55% | -32% / -34% | -43% / -45% | -37% / -38% | 62% |
| 2011 debt crisis | 2011-04-29 – 2011-10-03 | -17% / -18% | -14% / -15% | -16% / -16% | -16% / -16% | 77% |
| 2015-16 sell-off | 2015-05-21 – 2016-02-11 | -11% / -13% | -12% / -13% | -13% / -14% | -13% / -14% | 88% |
| 2018 Q4 | 2018-09-20 – 2018-12-24 | -15% / -19% | -12% / -15% | -15% / -17% | -15% / -16% | 77% |
| COVID crash | 2020-02-19 – 2020-03-23 | -28% / -33% | -16% / -17% | -18% / -20% | -17% / -18% | 50% |
| 2022 bear market | 2022-01-03 – 2022-10-12 | -23% / -24% | -18% / -19% | -19% / -20% | -21% / -22% | 68% |

## Market regime

Out of sample, 1997-05-29 to 2026-10-05 (29.2 years), without the breadth input. "Stressed" spells: 128. What happened over the next 20 trading days:

| Regime | Share of days | Volatility | Chance of a 5% drop | Average return |
|---|---|---|---|---|
| Calm | 38% | 11% | 9% | +0.6% |
| Normal | 46% | 17% | 21% | +0.8% |
| Stressed | 16% | 28% | 33% | +1.5% |

## Method

- Volatility: walk-forward periods; the chosen model is paired with each baseline on the same horizon and period, with a cluster bootstrap over periods and Holm correction.
- Allocation: daily returns from out-of-sample forecasts only, 0.1% cost per unit traded; the Sharpe comparison uses a 20-day moving-block bootstrap. Drawdowns are descriptive (no interval).
- Crisis windows are fixed peak-to-trough dates, chosen before looking at results.

Reproduce: `smp long-history`
