"""Command-line interface: `smp <command>`."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer

from smp import pipeline
from smp.config import load_config

app = typer.Typer(help="Stock Market Predictor pipeline.", no_args_is_help=True)
ConfigOpt = typer.Option(None, "--config", "-c", help="Path to config YAML.")


def _run(cfg, stages: list[str] | None = None, **kwargs) -> None:
    """Run stages (or the full pipeline); turn failures into a clear message and exit code 1."""
    from smp.safeio import PipelineBusy

    try:
        if stages is None:
            pipeline.run_all(cfg, **kwargs)
        else:
            pipeline.run(cfg, stages)
    except PipelineBusy as exc:
        typer.echo(f"Not started: {exc}.", err=True)
        raise typer.Exit(1) from None
    except pipeline.PipelineError as exc:
        typer.echo(f"Finished with {len(exc.failures)} failed stage(s): {exc}", err=True)
        raise typer.Exit(1) from None


def _cfg(config: Path | None, tune: bool = False):
    cfg = load_config(config)
    if tune:
        cfg.train.tune = True
    pipeline.setup_logging(cfg.paths.logs)
    return cfg


@app.command()
def ingest(config: Path | None = ConfigOpt):
    """Download or update raw market data."""
    _run(_cfg(config), ["ingest"])


@app.command()
def clean(config: Path | None = ConfigOpt):
    """Validate and clean raw data."""
    _run(_cfg(config), ["clean"])


@app.command()
def features(config: Path | None = ConfigOpt):
    """Build features and targets."""
    _run(_cfg(config), ["features"])


@app.command()
def train(config: Path | None = ConfigOpt, tune: bool = typer.Option(False, help="Tune LightGBM with Optuna.")):
    """Train and evaluate models."""
    _run(_cfg(config, tune), ["train"])


@app.command()
def predict(config: Path | None = ConfigOpt):
    """Write the latest next-day predictions."""
    _run(_cfg(config), ["predict"])


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
    _run(_cfg(config, tune), skip_ingest=skip_ingest, skip_train=skip_train)


@app.command()
def health(
    config: Path | None = ConfigOpt,
    snapshot: Path | None = typer.Option(None, help="Check a published snapshot's manifest.json instead."),
):
    """Check that data is current. Exits 1 if anything is stale or missing (for scripts and CI)."""
    from smp.health import data_problems, expected_latest_close, is_stale

    if snapshot is not None:
        as_of = json.loads(snapshot.read_text()).get("as_of") if snapshot.exists() else None
        current = as_of is not None and not is_stale(date.fromisoformat(as_of))
        typer.echo(f"snapshot as of {as_of}; expected {expected_latest_close()}: {'current' if current else 'stale'}")
        raise typer.Exit(0 if current else 1)
    problems = data_problems(load_config(config))
    for p in problems:
        typer.echo(f"problem: {p}")
    if not problems:
        typer.echo(f"ok: data is current (latest expected close {expected_latest_close()})")
    raise typer.Exit(1 if problems else 0)


@app.command("notify")
def notify_cmd(config: Path | None = ConfigOpt):
    """Build today's summary and send it (macOS / ntfy, if configured)."""
    _run(_cfg(config), ["notify"])
    typer.echo((load_config(config).paths.logs / "summary.md").read_text())


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
