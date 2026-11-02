# Stock Market Predictor

Predicts next-day stock moves with machine learning. An automated pipeline downloads, cleans and processes daily
prices, trains models, and writes predictions. A Streamlit dashboard shows the results.

> Educational project, not financial advice. Daily returns are close to random, so expect models to be only
> slightly better than a coin flip, if at all.

## What it does

| Stage | Module | Output |
|---|---|---|
| Ingest | `smp.ingest` | Daily adjusted OHLCV from Yahoo Finance in `data/raw/`. Re-runs only fetch new days, and refetch full history if past prices were re-adjusted (splits/dividends). |
| Clean | `smp.clean` | Sorted, de-duplicated, validated prices in `data/processed/clean/`, plus `quality_report.json` (filled values, dropped rows, fixed OHLC bars, flagged extreme moves, gaps). |
| Features | `smp.features` | 31 features per day (returns, moving-average ratios, MACD, RSI, stochastic, volatility, ATR, Bollinger %B, volume, SPY and VIX context, calendar) and two targets: next-day log return and up/down direction. |
| Train | `smp.train` | For each ticker and target: baseline, linear model, LightGBM and XGBoost compared with walk-forward cross-validation. The best non-baseline model is evaluated on the last 12 months (hold-out) and backtested, then refit on all data. Saved to `models/<TICKER>/`. |
| Predict | `smp.predict` | `models/predictions.parquet`: P(up) and predicted return for the next trading day. |
| Track record | `smp.history` | Every prediction is appended to `data/history/predictions.parquet` and scored once the next close is known. Only the first prediction per ticker and day is kept, so the record can't be rewritten. |

### Avoiding misleading results
- Features at day *t* only use data up to *t*'s close (`tests/test_features.py::test_no_lookahead`).
- Time-ordered splits only, with a gap between train and test; scaling is fit inside each fold.
- Every model is compared with a naive baseline (most-common class / zero return).
- The backtest charges 0.1% per position change and is compared with buy-and-hold.
- Today's price bar is ignored until 16:30 New York time, so features and scoring only use final closing prices.
- Live predictions are scored on outcomes that didn't exist when they were made (`smp track-record`).

## Setup

Requires Python 3.10+. On macOS, LightGBM and XGBoost need OpenMP: `brew install libomp`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # add ,tune for Optuna hyperparameter tuning
```

## Usage

```bash
smp run-all                    # ingest -> clean -> features -> train -> predict
smp run-all --skip-train       # refresh data and predictions with existing models
smp run-all --tune             # tune LightGBM with Optuna (pip install -e ".[tune]")
smp ingest | clean | features | train | predict   # run a single stage
smp track-record               # how past live predictions turned out

streamlit run app/dashboard.py # open the dashboard
```

Settings (tickers, dates, hold-out length, costs, paths) live in [config/config.yaml](config/config.yaml).
Logs are written to `logs/pipeline.log`.

### Dashboard
- **Overview**: next-day prediction for each ticker.
- **Track record**: live hit rate of past predictions vs always guessing "up", and each prediction's outcome.
- **Price & indicators**: candlesticks with SMA 20/50 and Bollinger band, volume, RSI and MACD.
- **Model performance**: CV and hold-out metrics for every model vs baseline, ROC curve and confusion matrix
  (direction) or predicted-vs-actual scatter (return), feature importance.
- **Backtest**: growth of $1 and drawdown, strategy vs buy-and-hold.
- **Data quality**: the cleaning report.

The sidebar button fetches new data and re-predicts without retraining.

## Automation

**Local (macOS launchd)**: runs weekdays at 22:30 local time.
```bash
# edit the path inside the plist first
cp scripts/com.smp.pipeline.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.smp.pipeline.plist
```

**Local (cron)**: `30 22 * * 1-5 /path/to/Stock-Market-Predictor/scripts/run_pipeline.sh`

**GitHub Actions**: [.github/workflows/pipeline.yml](.github/workflows/pipeline.yml) runs lint and tests on every
push, and runs the full pipeline on weekdays at 22:30 UTC (or on demand), uploading models and predictions as a
build artifact.

## Example results (hold-out 29 Oct 2025 to 29 Oct 2026)

| Ticker | Direction model | ROC-AUC | Accuracy | Backtest return | Buy & hold |
|---|---|---|---|---|---|
| AAPL | XGBoost | 0.545 | 55.6% | +28.9% | +28.3% |
| MSFT | LightGBM | 0.543 | 53.6% | -3.7% | -2.8% |
| NVDA | LightGBM | 0.516 | 48.8% | -2.0% | -0.9% |
| SPY | LightGBM | 0.422 | 46.0% | -12.3% | +11.1% |

None of the strategies reliably beats buy-and-hold. Treat this as a framework for testing ideas (new features,
data sources, longer horizons) rather than a trading signal.

## Tests

```bash
pytest          # cleaning rules, indicators, no-lookahead check, splits, backtest, end-to-end pipeline, dashboard
ruff check .
```

## Project layout

```
config/config.yaml     settings
src/smp/               pipeline package (config, ingest, clean, features, models, train, evaluate, predict, pipeline, cli)
app/dashboard.py       Streamlit dashboard
scripts/               cron/launchd helpers
tests/                 pytest suite
```
