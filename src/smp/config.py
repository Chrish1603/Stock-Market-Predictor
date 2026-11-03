"""Load and validate the YAML configuration."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

CONFIG_RELPATH = Path("config") / "config.yaml"


def find_project_root() -> Path:
    """The folder holding config/config.yaml.

    Order: SMP_PROJECT_ROOT, then the source checkout this package lives in (editable install), then the
    current folder and its parents (a regular install run from the repo, e.g. on Streamlit Cloud).
    """
    if env := os.environ.get("SMP_PROJECT_ROOT"):
        return Path(env).resolve()
    source = Path(__file__).resolve().parents[2]
    if (source / CONFIG_RELPATH).exists():
        return source
    cwd = Path.cwd().resolve()
    for folder in (cwd, *cwd.parents):
        if (folder / CONFIG_RELPATH).exists():
            return folder
    return source


PROJECT_ROOT = find_project_root()
DEFAULT_CONFIG = PROJECT_ROOT / CONFIG_RELPATH


class ConfigError(ValueError):
    """The config file is missing, unreadable or has invalid values."""


class Strict(BaseModel):
    """Unknown keys are errors, so a typo in config.yaml can't be silently ignored."""

    model_config = ConfigDict(extra="forbid")


def _horizons(v: list[int]) -> list[int]:
    if not v:
        raise ValueError("needs at least one horizon")
    if any(h < 1 for h in v):
        raise ValueError("horizons must be whole numbers of trading days, 1 or more")
    if len(set(v)) != len(v):
        raise ValueError("horizons must not repeat")
    return v


def _tickers(v: list[str]) -> list[str]:
    cleaned = [t.strip().upper() for t in v]
    if any(not t for t in cleaned):
        raise ValueError("ticker symbols must not be blank")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("ticker symbols must not repeat")
    return cleaned


class DataConfig(Strict):
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "SPY"])
    market_tickers: list[str] = Field(default_factory=lambda: ["SPY", "^VIX"])
    sectors: dict[str, str] = Field(default_factory=dict)  # ticker -> sector fund, e.g. AAPL: XLK
    earnings: bool = True  # download earnings dates for modelled tickers (ETFs have none)
    universe: list[str] = Field(default_factory=list)  # extra stocks used only as training data
    cross_sectional: bool = True  # rank features across all tickers each day (needs a universe)
    start: str = "2010-01-01"
    end: str | None = None
    ffill_limit: int = Field(2, ge=0, le=10)
    outlier_z: float = Field(6.0, gt=0)

    @field_validator("tickers")
    @classmethod
    def _check_tickers(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("needs at least one ticker to predict")
        return _tickers(v)

    @field_validator("market_tickers", "universe")
    @classmethod
    def _check_lists(cls, v: list[str]) -> list[str]:
        return _tickers(v)

    @field_validator("start", "end")
    @classmethod
    def _check_date(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                date.fromisoformat(v)
            except ValueError:
                raise ValueError(f"'{v}' is not a date like 2010-01-01") from None
        return v

    @model_validator(mode="after")
    def _check_sectors(self) -> DataConfig:
        if unknown := set(self.sectors) - set(self.tickers):
            raise ValueError(f"sectors set for tickers that aren't predicted: {sorted(unknown)}")
        return self

    @property
    def feature_tickers(self) -> list[str]:
        """Tickers that get feature files: the ones we predict plus the training universe."""
        return list(dict.fromkeys(self.tickers + self.universe))

    @property
    def all_tickers(self) -> list[str]:
        """Every ticker to download, without duplicates."""
        return list(dict.fromkeys(self.feature_tickers + self.market_tickers + list(self.sectors.values())))


class TrainConfig(Strict):
    horizons: list[int] = Field(default_factory=lambda: [1, 5, 20])  # trading days ahead to predict
    scope: Literal["per_ticker", "pooled"] = "per_ticker"  # one model per ticker, or one shared model
    n_splits: int = Field(5, ge=2, le=20)
    holdout_days: int = Field(252, ge=20)
    gap: int = Field(1, ge=0)
    tune: bool = False
    n_trials: int = Field(20, ge=1)
    random_state: int = 42

    _check_horizons = field_validator("horizons")(classmethod(lambda cls, v: _horizons(v)))


class RiskConfig(Strict):
    enabled: bool = True
    horizons: list[int] = Field(default_factory=lambda: [5, 20])  # days ahead for risk forecasts
    drop_threshold: float = Field(0.05, gt=0, lt=1)  # "big drop": close falls this far below today's
    target_vol: float = Field(0.15, gt=0, le=2)  # annualised volatility the allocation aims for
    max_weight: float = Field(1.0, gt=0, le=5)  # cap on position size (1.0 = no leverage)
    rebalance_band: float = Field(0.10, ge=0, lt=1)  # only trade when the target moves more than this
    allocation_horizon: int = 20  # which vol forecast sizes positions
    sizing_vol: Literal["trailing", "model"] = "trailing"  # trailing won the 2013-2026 backtest
    market_ticker: str = "SPY"  # ticker used for the market regime

    _check_horizons = field_validator("horizons")(classmethod(lambda cls, v: _horizons(v)))

    @model_validator(mode="after")
    def _check_allocation_horizon(self) -> RiskConfig:
        if self.enabled and self.allocation_horizon not in self.horizons:
            raise ValueError(f"allocation_horizon {self.allocation_horizon} must be one of horizons {self.horizons}")
        return self


class BacktestConfig(Strict):
    cost: float = Field(0.001, ge=0, lt=0.1)  # fraction of the trade, e.g. 0.001 = 0.1%
    prob_threshold: float = Field(0.5, gt=0, lt=1)


class PathsConfig(Strict):
    raw: Path = Path("data/raw")
    clean: Path = Path("data/processed/clean")
    features: Path = Path("data/processed/features")
    models: Path = Path("models")
    logs: Path = Path("logs")
    history: Path = Path("data/history")

    def resolve(self, root: Path) -> PathsConfig:
        """Make every relative path absolute against `root`."""
        return PathsConfig(**{k: (v if v.is_absolute() else root / v) for k, v in self})


class NotifyConfig(Strict):
    macos: bool = False  # desktop notification after each local run
    ntfy_topic: str | None = None  # push to phones via https://ntfy.sh/<topic> (opt-in, public service)
    ntfy_server: str = "https://ntfy.sh"


class DashboardConfig(Strict):
    # Where a hosted dashboard downloads its data snapshot when no local pipeline output exists
    data_url: str | None = None
    # Local runs adopt the track record in that snapshot (written by the daily GitHub run, the official
    # record) before adding their own predictions, so local and hosted dashboards show the same record.
    sync_track_record: bool = True


class Config(Strict):
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    notify: NotifyConfig = Field(default_factory=NotifyConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


def data_root() -> Path:
    """Base for relative data/model paths: SMP_DATA_ROOT if set (e.g. a published snapshot), else the project."""
    root = os.environ.get("SMP_DATA_ROOT")
    return Path(root) if root else PROJECT_ROOT


def _friendly(error: dict) -> str:
    if error["type"] == "extra_forbidden":
        return "unknown setting (typo?)"
    return error["msg"].removeprefix("Value error, ")


def load_config(path: str | Path | None = None) -> Config:
    """Read the config file; relative paths are resolved against the project root."""
    path = Path(path) if path else DEFAULT_CONFIG
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path} (set SMP_PROJECT_ROOT or run from the project)") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a mapping of sections (data, train, ...)")
    try:
        cfg = Config(**raw)
    except ValidationError as exc:
        lines = [f"  {'.'.join(str(x) for x in e['loc']) or '(top level)'}: {_friendly(e)}" for e in exc.errors()]
        raise ConfigError(f"invalid settings in {path}:\n" + "\n".join(lines)) from None
    cfg.paths = cfg.paths.resolve(data_root())
    return cfg


def ticker_to_filename(ticker: str) -> str:
    """'^VIX' -> 'VIX', 'BRK-B' -> 'BRK-B'; safe for file names."""
    return ticker.replace("^", "").replace("/", "_")
