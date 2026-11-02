"""Load and validate the YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


class DataConfig(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "SPY"])
    market_tickers: list[str] = Field(default_factory=lambda: ["SPY", "^VIX"])
    sectors: dict[str, str] = Field(default_factory=dict)  # ticker -> sector fund, e.g. AAPL: XLK
    earnings: bool = True  # download earnings dates for modelled tickers (ETFs have none)
    universe: list[str] = Field(default_factory=list)  # extra stocks used only as training data
    cross_sectional: bool = True  # rank features across all tickers each day (needs a universe)
    start: str = "2010-01-01"
    end: str | None = None
    ffill_limit: int = 2
    outlier_z: float = 6.0

    @property
    def feature_tickers(self) -> list[str]:
        """Tickers that get feature files: the ones we predict plus the training universe."""
        return list(dict.fromkeys(self.tickers + self.universe))

    @property
    def all_tickers(self) -> list[str]:
        """Every ticker to download, without duplicates."""
        return list(dict.fromkeys(self.feature_tickers + self.market_tickers + list(self.sectors.values())))


class TrainConfig(BaseModel):
    horizons: list[int] = Field(default_factory=lambda: [1, 5, 20])  # trading days ahead to predict
    scope: Literal["per_ticker", "pooled"] = "per_ticker"  # one model per ticker, or one shared model
    n_splits: int = 5
    holdout_days: int = 252
    gap: int = 1
    tune: bool = False
    n_trials: int = 20
    random_state: int = 42


class RiskConfig(BaseModel):
    enabled: bool = True
    horizons: list[int] = Field(default_factory=lambda: [5, 20])  # days ahead for risk forecasts
    drop_threshold: float = 0.05  # a "big drop": close falls this far below today's within the horizon
    target_vol: float = 0.15  # annualised volatility the allocation aims for
    max_weight: float = 1.0  # cap on position size (1.0 = no leverage)
    rebalance_band: float = 0.10  # only trade when the target weight moves by more than this
    allocation_horizon: int = 20  # which vol forecast sizes positions
    market_ticker: str = "SPY"  # ticker used for the market regime


class BacktestConfig(BaseModel):
    cost: float = 0.001
    prob_threshold: float = 0.5


class PathsConfig(BaseModel):
    raw: Path = Path("data/raw")
    clean: Path = Path("data/processed/clean")
    features: Path = Path("data/processed/features")
    models: Path = Path("models")
    logs: Path = Path("logs")
    history: Path = Path("data/history")

    def resolve(self, root: Path) -> PathsConfig:
        """Make every relative path absolute against `root`."""
        return PathsConfig(**{k: (v if v.is_absolute() else root / v) for k, v in self})


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
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
