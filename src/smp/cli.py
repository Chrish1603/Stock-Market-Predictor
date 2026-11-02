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


@app.command()
def publish(
    out: Path = typer.Option(Path("site-data"), help="Folder to write the dashboard snapshot to."),
    config: Path | None = ConfigOpt,
):
    """Write a small data snapshot that a hosted dashboard can read (SMP_DATA_ROOT=<out>)."""
    from smp.config import data_root
    from smp.publish import build_snapshot

    cfg = _cfg(config)
    m = build_snapshot(cfg, out, data_root())
    typer.echo(f"Wrote {m['files']} files ({m['bytes'] / 1e6:.1f} MB), data as of {m['as_of']}, to {out}")


experiment_app = typer.Typer(help="Compare config variants with confidence intervals.", no_args_is_help=True)
app.add_typer(experiment_app, name="experiment")


@experiment_app.command("run")
def experiment_run(
    spec: Path = typer.Argument(..., help="Experiment YAML, e.g. experiments/direction_inputs.yaml"),
    fresh: bool = typer.Option(False, help="Retrain every variant even if results are up to date."),
):
    """Train every variant, compare with the control and write a report."""
    from smp.experiment import REPORTS_DIR, load_spec, run_experiment

    pipeline.setup_logging(load_config().paths.logs)
    result = run_experiment(spec, fresh)
    typer.echo(f"\nReport: {REPORTS_DIR / (load_spec(spec)['name'] + '.md')}")
    for n, v in result["variants"].items():
        if "verdict" in v:
            typer.echo(f"  {n} vs {result['control']}: {v['verdict']} (diff {v['diff']:+.4f}, p={v['p_holm']:.3f})")


@experiment_app.command("list")
def experiment_list():
    """Show the experiment files and whether each has a report."""
    from smp.experiment import PROJECT_ROOT, REPORTS_DIR

    for f in sorted((PROJECT_ROOT / "experiments").glob("*.yaml")):
        has = (REPORTS_DIR / f"{f.stem}.md").exists()
        typer.echo(f"  {f.relative_to(PROJECT_ROOT)}  {'(report)' if has else '(not run)'}")


@experiment_app.command("index")
def experiment_index():
    """Rebuild docs/RESULTS.md from all reports."""
    from smp.experiment import build_index

    typer.echo(f"Wrote {build_index()}")


if __name__ == "__main__":
    app()
