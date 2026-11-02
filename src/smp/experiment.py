"""Reproducible experiments: compare config variants with confidence intervals.

An experiment is a YAML file in experiments/ that lists variants as overrides to config.yaml.
Each variant is trained in its own folder (live models are never touched), then every variant is
compared with the control fold by fold: same ticker, horizon and walk-forward period. Intervals
come from a cluster bootstrap over time periods, and p-values are Holm-corrected across variants,
so a variant only counts as "better" if the difference is unlikely to be luck.

    smp experiment run experiments/direction_inputs.yaml
    smp experiment index          # rebuild docs/RESULTS.md from all reports
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from smp.config import DEFAULT_CONFIG, PROJECT_ROOT, Config
from smp.stats import block_bootstrap_sharpe_diff, cluster_bootstrap_mean, holm, verdict

log = logging.getLogger(__name__)

RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "reports"
RESULTS_PAGE = PROJECT_ROOT / "docs" / "RESULTS.md"
KINDS = ("direction", "risk", "allocation")
LOWER_IS_BETTER = {"rmse", "brier", "log_loss", "mae", "mae_vol_points", "max_drawdown_abs", "volatility"}


# --- spec and config -----------------------------------------------------------------------------


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_spec(path: str | Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text())
    for key in ("name", "question", "kind", "metric", "control", "variants"):
        if key not in spec:
            raise ValueError(f"experiment spec is missing '{key}'")
    if spec["kind"] not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if spec["control"] not in spec["variants"]:
        raise ValueError("control must be one of the variants")
    return spec


def higher_is_better(metric: str) -> bool:
    return metric.split(".")[-1] not in LOWER_IS_BETTER


def variant_settings(spec: dict, name: str) -> dict:
    base = yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}
    return deep_merge(deep_merge(base, spec.get("base", {})), spec["variants"][name] or {})


def variant_config(spec: dict, name: str) -> Config:
    cfg = Config(**variant_settings(spec, name))
    cfg.paths = cfg.paths.resolve(PROJECT_ROOT)
    out = RUNS_DIR / spec["name"] / name
    cfg.paths.features, cfg.paths.models = out / "features", out / "models"
    return cfg


def _fingerprint(spec: dict, name: str) -> str:
    payload = json.dumps({"kind": spec["kind"], "settings": variant_settings(spec, name)}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --- running -------------------------------------------------------------------------------------


def ensure_data(cfg: Config) -> None:
    """Download and clean any ticker (or earnings file) a variant needs that isn't on disk yet."""
    from smp.clean import clean_all, clean_path
    from smp.ingest import earnings_path, ingest_earnings, ingest_ticker

    missing = [t for t in cfg.data.all_tickers if not clean_path(cfg, t).exists()]
    for t in missing:
        ingest_ticker(cfg, t)
    if missing:
        clean_all(cfg)
    if cfg.data.earnings:
        for t in cfg.data.feature_tickers:
            if not earnings_path(cfg, t).exists():
                ingest_earnings(cfg, t)


def run_variant(spec: dict, name: str, fresh: bool = False) -> Config:
    from smp.allocation import backtest_all
    from smp.features import build_all
    from smp.risk import train_risk_all
    from smp.train import train_direction

    cfg = variant_config(spec, name)
    marker = RUNS_DIR / spec["name"] / name / "done.json"
    fp = _fingerprint(spec, name)
    if not fresh and marker.exists() and json.loads(marker.read_text()).get("fingerprint") == fp:
        log.info("%s / %s: up to date, reusing results", spec["name"], name)
        return cfg
    start = time.perf_counter()
    log.info("%s / %s: training", spec["name"], name)
    ensure_data(cfg)
    build_all(cfg)
    if spec["kind"] == "direction":
        train_direction(cfg)
    else:
        train_risk_all(cfg)
        if spec["kind"] == "allocation":
            backtest_all(cfg)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"fingerprint": fp, "seconds": round(time.perf_counter() - start)}))
    log.info("%s / %s: done in %.0fs", spec["name"], name, time.perf_counter() - start)
    return cfg


# --- collecting scores ---------------------------------------------------------------------------


def fold_scores(cfg: Config, spec: dict, model: str | None = None) -> pd.DataFrame:
    """One row per (ticker, horizon, fold) with the metric for the selected model (or `model`)."""
    from smp.risk import risk_dir
    from smp.train import model_dir

    task, metric = spec["metric"].split(".")
    horizons = cfg.train.horizons if spec["kind"] == "direction" else cfg.risk.horizons
    rows = []
    for t in cfg.data.tickers:
        for h in horizons:
            d = model_dir(cfg, t, h) if spec["kind"] == "direction" else risk_dir(cfg, t, h)
            p = d / "metrics.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text())[task]
            name = model or m["selected"]
            for i, f in enumerate(m["cv_folds"].get(name, [])):
                rows.append({"ticker": t, "horizon": h, "fold": i, "model": name, "value": f.get(metric, np.nan)})
    return pd.DataFrame(rows)


def portfolio_returns(cfg: Config, strategy: str) -> pd.Series:
    curves = pd.read_parquet(cfg.paths.models / "_portfolio" / "allocation" / "curves.parquet")
    eq = curves[f"{strategy}_equity"]
    return eq.pct_change().fillna(eq.iloc[0] - 1)


def _compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Paired comparison a - b on matching (ticker, horizon, fold)."""
    key = ["ticker", "horizon", "fold"]
    m = a.merge(b, on=key, suffixes=("_a", "_b")).dropna(subset=["value_a", "value_b"])
    res = cluster_bootstrap_mean(m["value_a"] - m["value_b"], m["fold"])
    return {
        "diff": res["mean"],
        "ci_low": res["ci_low"],
        "ci_high": res["ci_high"],
        "p_value": res["p_value"],
        "n_pairs": res["n"],
        "n_periods": res["n_clusters"],
        "wins": int((m["value_a"] > m["value_b"]).sum()),
        "losses": int((m["value_a"] < m["value_b"]).sum()),
    }


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def _compare_sharpe(a: pd.Series, b: pd.Series) -> dict:
    res = block_bootstrap_sharpe_diff(a, b)
    return {
        "diff": res["mean"],
        "ci_low": res["ci_low"],
        "ci_high": res["ci_high"],
        "p_value": res["p_value"],
        "n_days": res["n"],
    }


# --- report --------------------------------------------------------------------------------------


def build_report(spec: dict, cfgs: dict[str, Config]) -> dict:
    """Compare every variant with the control, plus optional within-control comparisons."""
    hib = higher_is_better(spec["metric"])
    control = spec["control"]
    result: dict = {
        "name": spec["name"],
        "question": spec["question"],
        "kind": spec["kind"],
        "metric": spec["metric"],
        "control": control,
        "date": str(date.today()),
        "variants": {},
        "models": {},
        "strategies": {},
    }

    if spec["kind"] in ("direction", "risk"):
        scores = {n: fold_scores(c, spec) for n, c in cfgs.items()}
        for n, s in scores.items():
            entry = {"mean": float(s["value"].mean()) if len(s) else float("nan"), "n": int(len(s))}
            if n != control:
                entry |= _compare(s, scores[control])
            result["variants"][n] = entry
        breakdown = pd.DataFrame({n: s.groupby(["ticker", "horizon"])["value"].mean() for n, s in scores.items()})
        result["breakdown"] = breakdown.reset_index().to_dict(orient="records")
        for other in spec.get("compare_models", []):  # chosen model vs named alternatives, in the control run
            alt = fold_scores(cfgs[control], spec, model=other)
            result["models"][other] = {"mean": float(alt["value"].mean())} | _compare(scores[control], alt)
    else:
        strategy = spec.get("strategy", "trailing_vol_target")
        rets = {n: portfolio_returns(c, strategy) for n, c in cfgs.items()}
        for n, r in rets.items():
            eq = (1 + r).cumprod()
            entry = {"mean": _sharpe(r), "max_drawdown": float((eq / eq.cummax() - 1).min()), "n": int(len(r))}
            if n != control:
                entry |= _compare_sharpe(r, rets[control])
            result["variants"][n] = entry
        for other in spec.get("compare_strategies", []):
            r_other = portfolio_returns(cfgs[control], other)
            eq = (1 + r_other).cumprod()
            result["strategies"][other] = {
                "mean": _sharpe(r_other),
                "max_drawdown": float((eq / eq.cummax() - 1).min()),
            }
            result["strategies"][other] |= _compare_sharpe(rets[control], r_other)
        eq = (1 + rets[control]).cumprod()
        result["control_strategy"] = {
            "name": strategy,
            "sharpe": _sharpe(rets[control]),
            "max_drawdown": float((eq / eq.cummax() - 1).min()),
        }

    for group in ("variants", "models", "strategies"):
        compared = {k: v for k, v in result[group].items() if "p_value" in v}
        adjusted = holm({k: v["p_value"] for k, v in compared.items()})
        for k, v in compared.items():
            # Sharpe is always higher-is-better; otherwise follow the metric
            v["p_holm"] = adjusted[k]
            v["verdict"] = verdict(v["diff"], adjusted[k], True if spec["kind"] == "allocation" else hib)
    return result


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _p_phrase(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _num(x: float, digits: int = 4) -> str:
    return "–" if x is None or np.isnan(x) else f"{x:.{digits}f}"


def _sig(x: float, digits: int = 4) -> str:
    return "–" if x is None or np.isnan(x) else f"{x:+.{digits}f}"


def render_markdown(spec: dict, result: dict) -> str:
    metric, control, kind = result["metric"], result["control"], result["kind"]
    alloc = kind == "allocation"
    d = 2 if alloc else 4
    shown = "portfolio Sharpe" if alloc else metric
    better = "higher" if alloc or higher_is_better(metric) else "lower"
    lines = [
        f"# {spec['name']}",
        "",
        f"**Question:** {spec['question']}",
        "",
        f"Run {result['date']} · `{kind}` · metric `{shown}` ({better} is better) · control `{control}`",
        "",
    ]
    if spec.get("description"):
        lines += [spec["description"].strip(), ""]

    lines += ["## Variants vs control", ""]
    extra_h = " Max drawdown |" if alloc else ""
    wl_h = "" if alloc else " Better / worse folds |"
    lines += [
        f"| Variant | Mean |{extra_h} Δ vs control | 95% CI | p (Holm) |{wl_h} Verdict |",
        "|---|---|" + ("---|" if alloc else "") + "---|---|---|" + ("" if alloc else "---|") + "---|",
    ]
    for n, v in result["variants"].items():
        dd = f" {v['max_drawdown']:.1%} |" if alloc else ""
        if n == control:
            lines.append(f"| **{n}** (control) | {_num(v['mean'], d)} |{dd} – | – | – |{'' if alloc else ' – |'} – |")
        else:
            wl = "" if alloc else f" {v['wins']} / {v['losses']} |"
            lines.append(
                f"| {n} | {_num(v['mean'], d)} |{dd} {_sig(v['diff'], d)} | [{_sig(v['ci_low'], d)}, "
                f"{_sig(v['ci_high'], d)}] | {_p(v['p_holm'])} |{wl} {v['verdict']} |"
            )

    if result["models"]:
        lines += [
            "",
            "## Chosen model vs alternatives (control run)",
            "",
            "| Alternative | Its mean | Δ chosen − alternative | 95% CI | p (Holm) | Better / worse folds | Verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for n, v in result["models"].items():
            lines.append(
                f"| {n} | {_num(v['mean'])} | {_sig(v['diff'])} | [{_sig(v['ci_low'])}, {_sig(v['ci_high'])}]"
                f" | {_p(v['p_holm'])} | {v['wins']} / {v['losses']} | {v['verdict']} |"
            )
    if result["strategies"]:
        cs = result["control_strategy"]
        lines += [
            "",
            f"## `{cs['name']}` vs other strategies (control run)",
            "",
            f"`{cs['name']}`: Sharpe {cs['sharpe']:.2f}, max drawdown {cs['max_drawdown']:.1%}.",
            "",
            "| Alternative | Its Sharpe | Its max drawdown | Δ Sharpe | 95% CI | p (Holm) | Verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for n, v in result["strategies"].items():
            lines.append(
                f"| {n} | {v['mean']:.2f} | {v['max_drawdown']:.1%} | {v['diff']:+.2f} | "
                f"[{v['ci_low']:+.2f}, {v['ci_high']:+.2f}] | {_p(v['p_holm'])} | {v['verdict']} |"
            )
    if result.get("breakdown"):
        names = list(result["variants"])
        lines += [
            "",
            "## Breakdown (mean over folds)",
            "",
            "| Ticker | Horizon | " + " | ".join(names) + " |",
            "|---|---|" + "---|" * len(names),
        ]
        for r in result["breakdown"]:
            lines.append(
                f"| {r['ticker']} | {r['horizon']}d | " + " | ".join(_num(r.get(n, np.nan)) for n in names) + " |"
            )

    method = (
        [
            "- Daily returns of the equal-weight portfolio from out-of-sample forecasts only (2013 to today).",
            "- 95% intervals: moving-block bootstrap (20-day blocks) of the Sharpe difference.",
        ]
        if alloc
        else [
            "- Scores are walk-forward cross-validation folds; each variant is paired with the control on the same "
            "ticker, horizon and fold.",
            "- 95% intervals: cluster bootstrap that resamples whole time periods (folds), because scores from the "
            "same period move together.",
        ]
    )
    lines += [
        "",
        "## Method",
        "",
        *method,
        "- p-values are Holm-corrected across the comparisons in each table; a verdict needs p < 0.05.",
        "- Each run picks its best model on these same folds, which flatters every variant slightly.",
        "",
        f"Reproduce: `smp experiment run experiments/{spec['name']}.yaml`",
        "",
    ]
    return "\n".join(lines)


def run_experiment(path: str | Path, fresh: bool = False) -> dict:
    spec = load_spec(path)
    cfgs = {n: run_variant(spec, n, fresh) for n in spec["variants"]}
    result = build_report(spec, cfgs)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{spec['name']}.json").write_text(json.dumps(result, indent=2, default=float))
    (REPORTS_DIR / f"{spec['name']}.md").write_text(render_markdown(spec, result))
    build_index()
    return result


def build_index() -> Path:
    """docs/RESULTS.md: one section per experiment report, with its headline verdicts."""
    reports = sorted(REPORTS_DIR.glob("*.json"))
    lines = [
        "# Experiment results",
        "",
        "Generated by `smp experiment index` from [experiments/reports/](../experiments/reports/). "
        "Each experiment trains every variant in its own folder and compares it with a control, with "
        "95% intervals and Holm-corrected p-values.",
        "",
    ]
    for p in reports:
        r = json.loads(p.read_text())
        lines += [
            f"## [{r['name']}](../experiments/reports/{r['name']}.md)",
            "",
            f"{r['question']} *(run {r['date']})*",
            "",
        ]
        for n, v in r.get("variants", {}).items():
            if "verdict" in v:
                lines.append(
                    f"- **{n}** vs `{r['control']}`: {v['verdict']} (Δ {v['diff']:+.4f}, {_p_phrase(v['p_holm'])})"
                )
        for group, label in (("models", "chosen model vs"), ("strategies", "default strategy vs")):
            for n, v in r.get(group, {}).items():
                lines.append(f"- {label} **{n}**: {v['verdict']} (Δ {v['diff']:+.4f}, {_p_phrase(v['p_holm'])})")
        lines.append("")
    RESULTS_PAGE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PAGE.write_text("\n".join(lines))
    return RESULTS_PAGE
