"""Build a small data snapshot for a hosted dashboard.

The snapshot mirrors the project layout (data/..., models/...) so pointing SMP_DATA_ROOT at it
makes the dashboard work unchanged. It holds only what the dashboard reads: cleaned prices and
features for the predicted tickers, reports, predictions, history and model metrics, without the
trained model files or the training universe.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from smp.clean import clean_path, report_path
from smp.config import Config, PathsConfig
from smp.features import features_path
from smp.history import history_path

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
DASHBOARD_FEATURES = ["vol_20", "sma_ratio_200", "days_to_earnings", "has_earnings"]  # what the app reads


def _copy(src: Path, root: Path, out: Path) -> int:
    if not src.exists():
        return 0
    dest = out / src.relative_to(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest.stat().st_size


class SnapshotError(RuntimeError):
    """Publishing would damage the official track record; nothing was written."""


def check_records_kept(cfg: Config, previous: Path) -> None:
    """Refuse to publish if any prediction in the previous snapshot's track records (under `previous`,
    laid out like a snapshot) is missing from this run's. Guards against a run that lost the record."""
    from smp.history import HistoryError, _read, load_record, record_keys
    from smp.sync import records

    prev_dir = Path(previous) / PathsConfig().history  # snapshots use the default layout
    problems = []
    for name, (path, backup) in records(cfg).items():
        prev_main, prev_backup = prev_dir / path.name, prev_dir / backup.name
        if not prev_main.exists() and not prev_backup.exists():
            continue  # the previous snapshot had no such record (e.g. a record type added since)
        try:
            before = record_keys(load_record(prev_main, prev_backup))
        except HistoryError as exc:
            log.warning("can't read the previous %s track record, so it can't be checked: %s", name, exc)
            continue
        if not before:
            continue
        try:
            now = record_keys(_read(path)) if path.exists() else set()
        except Exception as exc:
            problems.append(f"{name} track record {path} is unreadable ({exc})")
            continue
        if missing := before - now:
            t, h, d = sorted(missing, key=lambda k: k[2])[0]
            problems.append(
                f"{name} track record would lose {len(missing)} of {len(before)} published prediction(s), "
                f"from {d.date()} ({t} {h}d) on"
            )
    if problems:
        raise SnapshotError("refusing to publish: " + "; ".join(problems))


def build_snapshot(cfg: Config, out: Path, root: Path, previous: Path | None = None) -> dict:
    """Copy the dashboard's inputs from a pipeline run (under `root`) into `out`. With `previous` (the
    last published snapshot), first check that no published prediction would be lost."""
    if previous is not None:
        check_records_kept(cfg, previous)
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files: list[Path] = [report_path(cfg), history_path(cfg)]
    files += [f for f in cfg.paths.history.glob("*") if f.is_file() and f != history_path(cfg)]
    files += [clean_path(cfg, t) for t in dict.fromkeys(cfg.data.tickers + cfg.data.market_tickers)]
    files += [f for f in cfg.paths.models.rglob("*") if f.is_file() and f.suffix in {".json", ".parquet"}]
    size = sum(_copy(f, root, out) for f in files)
    import pandas as pd

    for t in cfg.data.tickers:  # slim feature files: only the columns the dashboard uses
        src = features_path(cfg, t)
        if src.exists():
            f = pd.read_parquet(src)
            dest = out / src.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            f[[c for c in DASHBOARD_FEATURES if c in f]].to_parquet(dest)
            size += dest.stat().st_size
            files.append(src)
    as_of = None
    preds = cfg.paths.models / "risk_predictions.parquet"
    if preds.exists():
        as_of = str(pd.read_parquet(preds)["as_of"].max().date())
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": as_of,
        "files": len(files),
        "bytes": size,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2))
    log.info("snapshot: %d files, %.1f MB, data as of %s -> %s", len(files), size / 1e6, as_of, out)
    return manifest


def fetch_snapshot(url: str, dest: Path | None = None, timeout: int = 60) -> Path:
    """Download and unpack a published snapshot (a .tar.gz, e.g. GitHub's branch archive); returns its root,
    the folder holding manifest.json. Only plain files and folders inside `dest` are extracted."""
    import io
    import ssl
    import tarfile
    import tempfile
    import urllib.request

    import certifi

    ctx = ssl.create_default_context(cafile=certifi.where())  # python.org builds may lack system certificates
    with urllib.request.urlopen(url, timeout=timeout, context=ctx if url.startswith("https") else None) as resp:
        payload = resp.read()
    dest = Path(dest or tempfile.mkdtemp(prefix="smp-snapshot-")).resolve()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not (member.isfile() or member.isdir()) or (dest not in target.parents and target != dest):
                continue
            tar.extract(member, dest)
    try:
        return next(dest.rglob(MANIFEST)).parent
    except StopIteration:
        raise FileNotFoundError(f"{url} is not a dashboard snapshot (no {MANIFEST})") from None
