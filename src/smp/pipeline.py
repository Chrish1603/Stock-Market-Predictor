"""Run the pipeline stages in order: ingest -> clean -> features -> train -> predict."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from smp.clean import clean_all
from smp.config import Config
from smp.features import build_all
from smp.ingest import ingest_all
from smp.predict import predict_all
from smp.train import train_all

log = logging.getLogger("smp")

STAGES = {
    "ingest": ingest_all,
    "clean": clean_all,
    "features": build_all,
    "train": train_all,
    "predict": predict_all,
}


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if any(getattr(h, "_smp", False) for h in root.handlers):
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_dir / "pipeline.log")):
        handler.setFormatter(fmt)
        handler._smp = True
        root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("yfinance", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run_stage(cfg: Config, name: str):
    start = time.perf_counter()
    log.info("=== stage: %s ===", name)
    result = STAGES[name](cfg)
    log.info("=== %s done in %.1fs ===", name, time.perf_counter() - start)
    return result


def run(cfg: Config, stages: list[str]) -> dict:
    return {name: run_stage(cfg, name) for name in stages}


def run_all(cfg: Config, skip_ingest: bool = False, skip_train: bool = False) -> dict:
    stages = [s for s in STAGES if not (skip_ingest and s == "ingest") and not (skip_train and s == "train")]
    return run(cfg, stages)


def refresh(cfg: Config) -> dict:
    """Fetch new data and update predictions without retraining (used by the dashboard)."""
    return run(cfg, ["ingest", "clean", "features", "predict"])
