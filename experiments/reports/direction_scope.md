# direction_scope

**Question:** Does one pooled model trained on 63 stocks beat a separate model per stock?

Run 2026-11-03 · `direction` · metric `classifier.roc_auc` (higher is better) · control `per_ticker`

Both are scored on the 4 predicted tickers. Pooled folds are split by date across all stocks, so fold periods line up closely but not exactly with the per-stock folds.

## Variants vs control

| Variant | Mean | Δ vs control | 95% CI | p (Holm) | Better / worse folds | Verdict |
|---|---|---|---|---|---|---|
| **per_ticker** (control) | 0.5202 | – | – | – | – | – |
| pooled | 0.5149 | -0.0053 | [-0.0287, +0.0156] | 0.641 | 49 / 47 | no clear difference |

## Breakdown (mean over folds)

| Ticker | Horizon | per_ticker | pooled |
|---|---|---|---|
| AAPL | 1d | 0.5157 | 0.4949 |
| AAPL | 5d | 0.4927 | 0.5070 |
| AAPL | 20d | 0.5201 | 0.5297 |
| MSFT | 1d | 0.5233 | 0.5186 |
| MSFT | 5d | 0.5376 | 0.5199 |
| MSFT | 20d | 0.5467 | 0.5011 |
| NVDA | 1d | 0.5063 | 0.5190 |
| NVDA | 5d | 0.5198 | 0.5132 |
| NVDA | 20d | 0.5439 | 0.4886 |
| SPY | 1d | 0.5098 | 0.5178 |
| SPY | 5d | 0.5262 | 0.5325 |
| SPY | 20d | 0.4999 | 0.5368 |

## Method

- Scores are walk-forward cross-validation folds; each variant is paired with the control on the same ticker, horizon and fold.
- 95% intervals: cluster bootstrap that resamples whole time periods (folds), because scores from the same period move together.
- p-values are Holm-corrected across the comparisons in each table; a verdict needs p < 0.05.
- Each run picks its best model on these same folds, which flatters every variant slightly.

Reproduce: `smp experiment run experiments/direction_scope.yaml`
