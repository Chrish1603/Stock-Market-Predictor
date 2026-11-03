"""Run the pipeline stages in order: ingest -> clean -> features -> train -> sync -> predict -> notify."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from smp.clean import clean_all
from smp.config import Config
from smp.features import build_all
from smp.ingest import ingest_all
from smp.notify import notify
from smp.predict import predict_all
from smp.safeio import atomic_write_json, run_lock
from smp.sync import sync_track_record
from smp.train import train_all

log = logging.getLogger("smp")

STAGES = {
    "ingest": ingest_all,
    "clean": clean_all,
    "features": build_all,
    "train": train_all,
    "sync": sync_track_record,  # adopt the official track record before adding today's predictions
    "predict": predict_all,
    "notify": notify,
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


class PipelineError(RuntimeError):
    """One or more stages failed (the others still ran)."""

    def __init__(self, failures: list[dict]):
        self.failures = failures
        super().__init__("; ".join(f"{f['stage']}: {f['error']}" for f in failures))


def run_stage(cfg: Config, name: str):
    start = time.perf_counter()
    log.info("=== stage: %s ===", name)
    result = STAGES[name](cfg)
    log.info("=== %s done in %.1fs ===", name, time.perf_counter() - start)
    return result


def _write_status(cfg: Config, started: str, stages: list[str], failures: list[dict], finished: bool) -> None:
    atomic_write_json(
        cfg.paths.logs / "last_run.json",
        {
            "started": started,
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds") if finished else None,
            "stages": stages,
            "failures": failures,
        },
    )


def run(cfg: Config, stages: list[str], keep_going: bool = False) -> dict:
    """Run stages in order under a lock. With keep_going, a failed stage is logged and the rest still run
    (the summary reports it); PipelineError is raised at the end. Without it, the first failure is raised."""
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results: dict = {}
    failures: list[dict] = []
    with run_lock(cfg.paths.logs):
        for name in stages:
            if name == "notify":  # let the summary see failures so far
                _write_status(cfg, started, stages, failures, finished=False)
            try:
                results[name] = run_stage(cfg, name)
            except Exception as exc:
                log.exception("stage %s failed", name)
                failures.append({"stage": name, "error": f"{type(exc).__name__}: {exc}"[:300]})
                if not keep_going:
                    _write_status(cfg, started, stages, failures, finished=True)
                    raise
        _write_status(cfg, started, stages, failures, finished=True)
    if failures:
        raise PipelineError(failures)
    return results


def run_all(cfg: Config, skip_ingest: bool = False, skip_train: bool = False) -> dict:
    """Every stage; a failure doesn't stop the rest, and the summary is always sent."""
    stages = [s for s in STAGES if not (skip_ingest and s == "ingest") and not (skip_train and s == "train")]
    return run(cfg, stages, keep_going=True)


def refresh(cfg: Config) -> dict:
    """Fetch new data and update predictions without retraining (used by the dashboard)."""
    return run(cfg, ["ingest", "clean", "features", "sync", "predict"], keep_going=True)
