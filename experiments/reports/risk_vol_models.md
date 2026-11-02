# risk_vol_models

**Question:** Do the volatility forecasts beat standard baselines, and do earnings dates help them?

Run 2026-11-03 · `risk` · metric `vol.r2` (higher is better) · control `default`

R² of log annualised realised volatility, 5 and 20 days ahead. Baselines: "persistence" (next period's volatility equals the last month's) and HAR (a linear blend of weekly, monthly and quarterly volatility).

## Variants vs control

| Variant | Mean | Δ vs control | 95% CI | p (Holm) | Better / worse folds | Verdict |
|---|---|---|---|---|---|---|
| **default** (control) | 0.1146 | – | – | – | – | – |
| no_earnings | 0.0911 | -0.0236 | [-0.0435, -0.0010] | 0.028 | 16 / 32 | worse |

## Chosen model vs alternatives (control run)

| Alternative | Its mean | Δ chosen − alternative | 95% CI | p (Holm) | Better / worse folds | Verdict |
|---|---|---|---|---|---|---|
| persistence | -0.1973 | +0.3119 | [+0.2166, +0.4247] | < 0.001 | 61 / 3 | better |
| har | 0.0632 | +0.0514 | [+0.0378, +0.0639] | < 0.001 | 36 / 12 | better |

## Breakdown (mean over folds)

| Ticker | Horizon | default | no_earnings |
|---|---|---|---|
| AAPL | 5d | 0.1357 | 0.1063 |
| AAPL | 20d | 0.0992 | 0.0992 |
| MSFT | 5d | 0.1607 | 0.1048 |
| MSFT | 20d | 0.1078 | 0.1078 |
| NVDA | 5d | 0.0532 | 0.0202 |
| NVDA | 20d | -0.1698 | -0.2400 |
| SPY | 5d | 0.4042 | 0.4042 |
| SPY | 20d | 0.1260 | 0.1260 |

## Method

- Scores are walk-forward cross-validation folds; each variant is paired with the control on the same ticker, horizon and fold.
- 95% intervals: cluster bootstrap that resamples whole time periods (folds), because scores from the same period move together.
- p-values are Holm-corrected across the comparisons in each table; a verdict needs p < 0.05.
- Each run picks its best model on these same folds, which flatters every variant slightly.

Reproduce: `smp experiment run experiments/risk_vol_models.yaml`
