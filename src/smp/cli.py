"""Command-line interface: `smp <command>`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from smp import pipeline
from smp.config import load_config

app = typer.Typer(help="Stock Market Predictor pipeline.", no_args_is_help=True)
ConfigOpt = typer.Option(None, "--config", "-c", help="Path to config YAML.")


def _cfg(config: Optional[Path], tune: bool = False):
    cfg = load_config(config)
    if tune:
        cfg.train.tune = True
    pipeline.setup_logging(cfg.paths.logs)
    return cfg


@app.command()
def ingest(config: Optional[Path] = ConfigOpt):
    """Download or update raw market data."""
    pipeline.run(_cfg(config), ["ingest"])


@app.command()
def clean(config: Optional[Path] = ConfigOpt):
    """Validate and clean raw data."""
    pipeline.run(_cfg(config), ["clean"])


@app.command()
def features(config: Optional[Path] = ConfigOpt):
    """Build features and targets."""
    pipeline.run(_cfg(config), ["features"])


@app.command()
def train(config: Optional[Path] = ConfigOpt, tune: bool = typer.Option(False, help="Tune LightGBM with Optuna.")):
    """Train and evaluate models."""
    pipeline.run(_cfg(config, tune), ["train"])


@app.command()
def predict(config: Optional[Path] = ConfigOpt):
    """Write the latest next-day predictions."""
    pipeline.run(_cfg(config), ["predict"])


@app.command("run-all")
def run_all(
    config: Optional[Path] = ConfigOpt,
    tune: bool = typer.Option(False, help="Tune LightGBM with Optuna."),
    skip_ingest: bool = typer.Option(False, help="Use existing raw data."),
    skip_train: bool = typer.Option(False, help="Reuse existing models."),
):
    """Run the full pipeline: ingest -> clean -> features -> train -> predict."""
    pipeline.run_all(_cfg(config, tune), skip_ingest=skip_ingest, skip_train=skip_train)


if __name__ == "__main__":
    app()
