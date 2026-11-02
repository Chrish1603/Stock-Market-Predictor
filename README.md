# Stock Market Predictor

Predicts stock moves 1, 5 and 20 trading days ahead with machine learning. An automated pipeline downloads, cleans and processes daily
prices, trains models, and writes predictions. A Streamlit dashboard shows the results.

> Educational project, not financial advice. Daily returns are close to random, so expect models to be only
> slightly better than a coin flip, if at all.

## What it does

| Stage | Module | Output |
|---|---|---|
| Ingest | `smp.ingest` | Daily adjusted OHLCV from Yahoo Finance in `data/raw/`. Re-runs only fetch new days, and refetch full history if past prices were re-adjusted (splits/dividends). |
| Clean | `smp.clean` | Sorted, de-duplicated, validated prices in `data/processed/clean/`, plus `quality_report.json` (filled values, dropped rows, fixed OHLC bars, flagged extreme moves, gaps). |
| Features | `smp.features` | 34 features per stock by default: price (returns, moving-average ratios, MACD, RSI, stochastic, volatility, ATR, Bollinger %B, volume), market (S&P 500, VIX), earnings (days to next / since last report, last EPS surprise) and calendar. Optional: Treasury-yield and sector-fund inputs (see below). Plus targets for each horizon (1, 5, 20 days): log return and up/down direction. |
| Train | `smp.train` | For each ticker, horizon and target: baseline, linear model, LightGBM and XGBoost compared with walk-forward cross-validation. The best non-baseline model is evaluated on the last 12 months (hold-out) and backtested, then refit on all data. Saved to `models/<TICKER>/h<N>/`. |
| Predict | `smp.predict` | `models/predictions.parquet`: P(up) and predicted return for each horizon. |
| Track record | `smp.history` | Every prediction is appended to `data/history/predictions.parquet` and scored once its target day has closed. Only the first prediction per ticker and day is kept, so the record can't be rewritten. |

### Avoiding misleading results
- Features at day *t* only use data up to *t*'s close (`tests/test_features.py::test_no_lookahead`).
- Time-ordered splits only, with a gap between train and test at least as long as the horizon (so overlapping
  multi-day labels can't leak); scaling is fit inside each fold.
- Longer-horizon backtests hold each position for the full horizon instead of trading daily.
- Every model is compared with a naive baseline (most-common class / zero return).
- The backtest charges 0.1% per position change and is compared with buy-and-hold.
- Optional interest-rate inputs are changes, not levels: levels drift for years and let trees memorise eras.
- Earnings dates are only treated as known up to 20 business days ahead, and an EPS surprise only from its
  reaction day (same day for pre-noon reports, else the next trading day).
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

Settings (tickers, horizons, sector funds, earnings on/off, dates, hold-out length, costs, paths) live in [config/config.yaml](config/config.yaml).
Logs are written to `logs/pipeline.log`.

### Dashboard
- **Overview**: next-day prediction for each ticker.
- **Track record**: live hit rate of past predictions vs always guessing "up", and each prediction's outcome.
- **Price & indicators**: candlesticks with SMA 20/50 and Bollinger band, volume, RSI and MACD.
- **Model performance**: CV and hold-out metrics for every model vs baseline, ROC curve and confusion matrix
  (direction) or predicted-vs-actual scatter (return), feature importance.
- **Backtest**: growth of $1 and drawdown, strategy vs buy-and-hold.
- **Data quality**: the cleaning report.

Pick the horizon in the sidebar; every tab follows it. The sidebar button fetches new data and re-predicts
without retraining.

## Automation

**Local (macOS launchd)**: the included plist runs at 11:00 Tue-Sat New Zealand time, just after the Mon-Fri US
close. Adjust the times for your time zone so it runs after 16:30 New York time.
```bash
# edit the path inside the plist first
cp scripts/com.smp.pipeline.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.smp.pipeline.plist
```

**Local (cron)**: `30 22 * * 1-5 /path/to/Stock-Market-Predictor/scripts/run_pipeline.sh`

**GitHub Actions**: [.github/workflows/pipeline.yml](.github/workflows/pipeline.yml) runs lint and tests on every
push, and runs the full pipeline on weekdays at 22:30 UTC (or on demand), uploading models and predictions as a
build artifact.

## Example results: 1-day horizon (hold-out 29 Oct 2025 to 29 Oct 2026)

| Ticker | Direction model | ROC-AUC | Accuracy | Backtest return | Buy & hold |
|---|---|---|---|---|---|
| AAPL | XGBoost | 0.545 | 55.6% | +28.9% | +28.3% |
| MSFT | LightGBM | 0.543 | 53.6% | -3.7% | -2.8% |
| NVDA | LightGBM | 0.516 | 48.8% | -2.0% | -0.9% |
| SPY | LightGBM | 0.422 | 46.0% | -12.3% | +11.1% |

None of the strategies reliably beats buy-and-hold. Treat this as a framework for testing ideas (new features,
data sources, longer horizons) rather than a trading signal.

## Which inputs help

Each optional input group was tested by retraining all 12 models (4 tickers x 3 horizons) with it switched on or off.
Mean walk-forward CV ROC-AUC (0.5 = guessing):

| Inputs | Mean CV AUC | Better than all-inputs on |
|---|---|---|
| Price + market + **earnings** (default) | **0.5181** | 8/12 |
| Price + market only | 0.5172 | 5/12 |
| All (+ rates + sector funds) | 0.5137 | – |

Rates and sector funds added noise, so they're off by default. Turn them back on in
[config/config.yaml](config/config.yaml). The differences are small (under 0.005), so re-test after changing tickers.

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
