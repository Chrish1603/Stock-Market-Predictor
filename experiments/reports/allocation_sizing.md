# allocation_sizing

**Question:** Does volatility targeting beat buy-and-hold, and which sizing settings work best?

Run 2026-11-03 · `allocation` · metric `portfolio Sharpe` (higher is better) · control `default`

Equal-weight portfolio of the 4 tickers, 2013 to today, using only out-of-sample forecasts. The default targets 15% volatility with no leverage and sizes with last month's volatility.

## Variants vs control

| Variant | Mean | Max drawdown | Δ vs control | 95% CI | p (Holm) | Verdict |
|---|---|---|---|---|---|---|
| **default** (control) | 1.40 | -17.8% | – | – | – | – |
| target_10 | 1.37 | -12.9% | -0.02 | [-0.08, +0.03] | 1.000 | no clear difference |
| target_20 | 1.41 | -21.0% | +0.02 | [-0.03, +0.06] | 1.000 | no clear difference |
| leverage_1_5 | 1.38 | -17.7% | -0.02 | [-0.06, +0.03] | 1.000 | no clear difference |
| no_band | 1.42 | -16.5% | +0.02 | [-0.01, +0.05] | 0.648 | no clear difference |

## `trailing_vol_target` vs other strategies (control run)

`trailing_vol_target`: Sharpe 1.40, max drawdown -17.8%.

| Alternative | Its Sharpe | Its max drawdown | Δ Sharpe | 95% CI | p (Holm) | Verdict |
|---|---|---|---|---|---|---|
| buy_hold | 1.31 | -37.5% | +0.09 | [-0.10, +0.26] | 0.380 | no clear difference |
| model_vol_target | 1.29 | -22.8% | +0.10 | [-0.02, +0.23] | 0.227 | no clear difference |
| model_vol_target_drop_filter | 1.27 | -22.6% | +0.13 | [-0.01, +0.26] | 0.146 | no clear difference |

## Method

- Daily returns of the equal-weight portfolio from out-of-sample forecasts only (2013 to today).
- 95% intervals: moving-block bootstrap (20-day blocks) of the Sharpe difference.
- p-values are Holm-corrected across the comparisons in each table; a verdict needs p < 0.05.
- Each run picks its best model on these same folds, which flatters every variant slightly.

Reproduce: `smp experiment run experiments/allocation_sizing.yaml`
