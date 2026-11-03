import pandas as pd
import pytest
import yaml

from conftest import make_ohlcv
from smp.config import Config, ConfigError, DataConfig, PathsConfig, RiskConfig, TrainConfig, load_config


def _write(tmp_path, cfg: dict):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_real_config_is_valid():
    load_config()


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"train": {"horizon": [1]}}, "train.horizon: unknown setting"),
        ({"train": {"horizons": [0, 5]}}, "1 or more"),
        ({"train": {"horizons": [5, 5]}}, "must not repeat"),
        ({"data": {"tickers": []}}, "at least one ticker"),
        ({"data": {"tickers": ["AAPL", "aapl"]}}, "must not repeat"),
        ({"data": {"start": "2010-02-30"}}, "is not a date"),
        ({"data": {"sectors": {"TSLA": "XLY"}}}, "aren't predicted"),
        ({"risk": {"target_vol": -0.1}}, "greater than 0"),
        ({"risk": {"allocation_horizon": 10}}, "must be one of horizons"),
        ({"backtest": {"cost": 0.5}}, "less than 0.1"),
        (["not", "a", "mapping"], "must be a mapping"),
    ],
)
def test_bad_settings_are_rejected_with_the_field_named(tmp_path, bad, message):
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, bad))


def test_missing_and_broken_files(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("data: [unclosed")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(broken)


def test_tickers_are_normalised():
    assert DataConfig(tickers=[" aapl ", "msft"]).tickers == ["AAPL", "MSFT"]


def test_prediction_survives_one_bad_ticker_and_reports_it(tmp_path):
    """A model whose inputs no longer exist is skipped and reported; other tickers are still predicted."""
    from smp import pipeline
    from smp.features import features_path
    from smp.ingest import raw_path
    from smp.predict import PredictionError, predict_all, predictions_path

    cfg = Config(
        data=DataConfig(tickers=["AAA", "BBB"], market_tickers=[], earnings=False),
        train=TrainConfig(horizons=[1], n_splits=3, holdout_days=120),
        risk=RiskConfig(horizons=[5], allocation_horizon=5),
        paths=PathsConfig().resolve(tmp_path),
    )
    cfg.paths.raw.mkdir(parents=True)
    for i, t in enumerate(["AAA", "BBB"]):
        make_ohlcv(n=800, seed=i).to_parquet(raw_path(cfg, t))
    pipeline.run(cfg, ["clean", "features", "train"])

    f = pd.read_parquet(features_path(cfg, "BBB"))
    f.drop(columns=["rsi_14"]).to_parquet(features_path(cfg, "BBB"))  # e.g. inputs changed without retraining
    a = pd.read_parquet(features_path(cfg, "AAA"))
    a.iloc[-1, a.columns.get_loc("vol_20")] = float("nan")  # newest day incomplete
    a.to_parquet(features_path(cfg, "AAA"))

    with pytest.raises(PredictionError) as err:
        predict_all(cfg)
    text = str(err.value)
    assert "BBB 1d: model needs inputs that are no longer built (rsi_14)" in text and "retrain" in text
    assert "AAA 1d: inputs incomplete" in text
    preds = pd.read_parquet(predictions_path(cfg))
    assert list(preds["ticker"]) == ["AAA"]  # the good ticker was still predicted and saved
