"""Command-line interface: `smp <command>`."""

from __future__ import annotations

from pathlib import Path

import typer

from smp import pipeline
from smp.config import load_config

app = typer.Typer(help="Stock Market Predictor pipeline.", no_args_is_help=True)
ConfigOpt = typer.Option(None, "--config", "-c", help="Path to config YAML.")


def _cfg(config: Path | None, tune: bool = False):
    cfg = load_config(config)
    if tune:
        cfg.train.tune = True
    pipeline.setup_logging(cfg.paths.logs)
    return cfg


@app.command()
def ingest(config: Path | None = ConfigOpt):
    """Download or update raw market data."""
    pipeline.run(_cfg(config), ["ingest"])


@app.command()
def clean(config: Path | None = ConfigOpt):
    """Validate and clean raw data."""
    pipeline.run(_cfg(config), ["clean"])


@app.command()
def features(config: Path | None = ConfigOpt):
    """Build features and targets."""
    pipeline.run(_cfg(config), ["features"])


@app.command()
def train(config: Path | None = ConfigOpt, tune: bool = typer.Option(False, help="Tune LightGBM with Optuna.")):
    """Train and evaluate models."""
    pipeline.run(_cfg(config, tune), ["train"])


@app.command()
def predict(config: Path | None = ConfigOpt):
    """Write the latest next-day predictions."""
    pipeline.run(_cfg(config), ["predict"])


@app.command("track-record")
def track_record(config: Path | None = ConfigOpt):
    """Show how past live predictions turned out."""
    from smp.history import load_history, summarize

    summary = summarize(load_history(load_config(config)))
    if summary.empty:
        typer.echo("No predictions recorded yet. Run `smp predict`.")
        return
    typer.echo(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


@app.command("run-all")
def run_all(
    config: Path | None = ConfigOpt,
    tune: bool = typer.Option(False, help="Tune LightGBM with Optuna."),
    skip_ingest: bool = typer.Option(False, help="Use existing raw data."),
    skip_train: bool = typer.Option(False, help="Reuse existing models."),
):
    """Run the full pipeline: ingest -> clean -> features -> train -> predict."""
    pipeline.run_all(_cfg(config, tune), skip_ingest=skip_ingest, skip_train=skip_train)


if __name__ == "__main__":
    app()
