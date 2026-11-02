"""Load and validate the YAML configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


class DataConfig(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "SPY"])
    market_tickers: list[str] = Field(default_factory=lambda: ["SPY", "^VIX"])
    start: str = "2010-01-01"
    end: str | None = None
    ffill_limit: int = 2
    outlier_z: float = 6.0

    @property
    def all_tickers(self) -> list[str]:
        """Modelled tickers plus market-context tickers, without duplicates."""
        return list(dict.fromkeys(self.tickers + self.market_tickers))


class TrainConfig(BaseModel):
    n_splits: int = 5
    holdout_days: int = 252
    gap: int = 1
    tune: bool = False
    n_trials: int = 20
    random_state: int = 42


class BacktestConfig(BaseModel):
    cost: float = 0.001
    prob_threshold: float = 0.5


class PathsConfig(BaseModel):
    raw: Path = Path("data/raw")
    clean: Path = Path("data/processed/clean")
    features: Path = Path("data/processed/features")
    models: Path = Path("models")
    logs: Path = Path("logs")

    def resolve(self, root: Path) -> PathsConfig:
        """Make every relative path absolute against `root`."""
        return PathsConfig(**{k: (v if v.is_absolute() else root / v) for k, v in self})


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def load_config(path: str | Path | None = None) -> Config:
    """Read the config file; relative paths are resolved against the project root."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config(**raw)
    cfg.paths = cfg.paths.resolve(PROJECT_ROOT)
    return cfg


def ticker_to_filename(ticker: str) -> str:
    """'^VIX' -> 'VIX', 'BRK-B' -> 'BRK-B'; safe for file names."""
    return ticker.replace("^", "").replace("/", "_")
