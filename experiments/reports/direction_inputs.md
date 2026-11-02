# direction_inputs

**Question:** Is up/down direction predictable at all, and do the optional inputs help?

Run 2026-11-03 · `direction` · metric `classifier.roc_auc` (higher is better) · control `default`

The control is the current config (price, market, earnings and cross-sectional rank inputs). "Chosen model vs baseline" tests whether direction beats a coin flip at all (the baseline always predicts the most common class, so its ROC-AUC is 0.5).

## Variants vs control

| Variant | Mean | Δ vs control | 95% CI | p (Holm) | Better / worse folds | Verdict |
|---|---|---|---|---|---|---|
| **default** (control) | 0.5202 | – | – | – | – | – |
| no_earnings | 0.5199 | -0.0003 | [-0.0047, +0.0044] | 1.000 | 43 / 29 | no clear difference |
| no_universe | 0.5213 | +0.0011 | [-0.0031, +0.0057] | 1.000 | 53 / 43 | no clear difference |
| plus_rates | 0.5199 | -0.0003 | [-0.0054, +0.0044] | 1.000 | 45 / 51 | no clear difference |
| plus_sectors | 0.5204 | +0.0002 | [-0.0023, +0.0025] | 1.000 | 35 / 37 | no clear difference |

## Chosen model vs alternatives (control run)

| Alternative | Its mean | Δ chosen − alternative | 95% CI | p (Holm) | Better / worse folds | Verdict |
|---|---|---|---|---|---|---|
| baseline | 0.5000 | +0.0202 | [+0.0076, +0.0318] | < 0.001 | 64 / 32 | better |

## Breakdown (mean over folds)

| Ticker | Horizon | default | no_earnings | no_universe | plus_rates | plus_sectors |
|---|---|---|---|---|---|---|
| AAPL | 1d | 0.5157 | 0.5152 | 0.5136 | 0.5196 | 0.5233 |
| AAPL | 5d | 0.4927 | 0.5015 | 0.4958 | 0.4876 | 0.4924 |
| AAPL | 20d | 0.5201 | 0.5368 | 0.5267 | 0.5169 | 0.5241 |
| MSFT | 1d | 0.5233 | 0.5192 | 0.5141 | 0.5272 | 0.5201 |
| MSFT | 5d | 0.5376 | 0.5408 | 0.5416 | 0.5381 | 0.5353 |
| MSFT | 20d | 0.5467 | 0.5565 | 0.5446 | 0.5280 | 0.5476 |
| NVDA | 1d | 0.5063 | 0.5045 | 0.5074 | 0.5019 | 0.5047 |
| NVDA | 5d | 0.5198 | 0.5125 | 0.5173 | 0.5159 | 0.5204 |
| NVDA | 20d | 0.5439 | 0.5157 | 0.5608 | 0.5570 | 0.5406 |
| SPY | 1d | 0.5098 | 0.5098 | 0.5093 | 0.5159 | 0.5098 |
| SPY | 5d | 0.5262 | 0.5262 | 0.5340 | 0.5185 | 0.5262 |
| SPY | 20d | 0.4999 | 0.4999 | 0.4902 | 0.5122 | 0.4999 |

## Method

- Scores are walk-forward cross-validation folds; each variant is paired with the control on the same ticker, horizon and fold.
- 95% intervals: cluster bootstrap that resamples whole time periods (folds), because scores from the same period move together.
- p-values are Holm-corrected across the comparisons in each table; a verdict needs p < 0.05.
- Each run picks its best model on these same folds, which flatters every variant slightly.

Reproduce: `smp experiment run experiments/direction_inputs.yaml`
