# risk_drop_models

**Question:** Are the "chance of a 5% drop" probabilities better than quoting the historical drop rate?

Run 2026-11-03 · `risk` · metric `drop.brier` (lower is better) · control `default`

Brier score (mean squared error of the probabilities; lower is better). The baseline always predicts the historical drop rate. Where nothing beats it, the chosen model is the baseline itself.

## Variants vs control

| Variant | Mean | Δ vs control | 95% CI | p (Holm) | Better / worse folds | Verdict |
|---|---|---|---|---|---|---|
| **default** (control) | 0.1394 | – | – | – | – | – |

## Chosen model vs alternatives (control run)

| Alternative | Its mean | Δ chosen − alternative | 95% CI | p (Holm) | Better / worse folds | Verdict |
|---|---|---|---|---|---|---|
| baseline | 0.1418 | -0.0024 | [-0.0044, -0.0005] | 0.016 | 9 / 30 | better |

## Breakdown (mean over folds)

| Ticker | Horizon | default |
|---|---|---|
| AAPL | 5d | 0.0892 |
| AAPL | 20d | 0.2255 |
| MSFT | 5d | 0.0633 |
| MSFT | 20d | 0.1935 |
| NVDA | 5d | 0.1456 |
| NVDA | 20d | 0.2562 |
| SPY | 5d | 0.0258 |
| SPY | 20d | 0.1161 |

## Method

- Scores are walk-forward cross-validation folds; each variant is paired with the control on the same ticker, horizon and fold.
- 95% intervals: cluster bootstrap that resamples whole time periods (folds), because scores from the same period move together.
- p-values are Holm-corrected across the comparisons in each table; a verdict needs p < 0.05.
- Each run picks its best model on these same folds, which flatters every variant slightly.

Reproduce: `smp experiment run experiments/risk_drop_models.yaml`
