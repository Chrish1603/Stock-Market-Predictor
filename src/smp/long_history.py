"""Long-history check: does the risk side hold up from 1993, through the 2000-02 and 2008 crashes?

The live system starts in 2010 (the stock universe and earnings data), so its backtests miss the two
worst bear markets of the last 30 years. This study rebuilds the market ticker's (SPY's) risk models
alone from its first trading day, in its own folder (live data and models are untouched), and reports:

- volatility forecasts vs "same as last month" and HAR, per walk-forward period
- volatility targeting vs buy-and-hold: overall, before and after 2010, and through each crisis
- the market regime's next-20-day outcomes

Differences from the live SPY models: no cross-sectional ranks (they need the 2010+ stock universe)
and no breadth input to the regime. Crisis windows are the S&P 500's peak-to-trough dates, fixed
up front, not chosen from the results.

    smp long-history
"""

from __future__ import annotations

import json
import logging
from datetime import date

import numpy as np
import pandas as pd

from smp.config import Config, PathsConfig, load_config
from smp.experiment import REPORTS_DIR, RUNS_DIR, _compare, _compare_sharpe, _p, build_index, fold_scores
from smp.stats import holm, verdict

log = logging.getLogger(__name__)

NAME = "long_history"
QUESTION = "Do the risk results hold up over 30 years, including the 2000-02 and 2008 crashes?"
START = "1993-01-29"  # SPY's first trading day
N_SPLITS = 8  # more periods than live (5): out-of-sample forecasts then start in the late 1990s
SPLIT_YEAR = 2010  # the live system's data starts here
CRISES = [  # S&P 500 peak and trough closes
    ("1998 LTCM / Russia", "1998-07-17", "1998-10-08"),
    ("Dot-com bust", "2000-03-24", "2002-10-09"),
    ("Financial crisis", "2007-10-09", "2009-03-09"),
    ("2011 debt crisis", "2011-04-29", "2011-10-03"),
    ("2015-16 sell-off", "2015-05-21", "2016-02-11"),
    ("2018 Q4", "2018-09-20", "2018-12-24"),
    ("COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 bear market", "2022-01-03", "2022-10-12"),
]
STRATEGY_LABELS = {
    "buy_hold": "Buy & hold",
    "trailing_vol_target": "Vol target, trailing vol",
    "model_vol_target": "Vol target, model forecast",
    "model_vol_target_drop_filter": "Model forecast + drop filter",
}


def run_dir():
    return RUNS_DIR / NAME


def long_config(base: Config | None = None) -> Config:
    """The live settings, narrowed to the market ticker and extended back to START, in its own folder."""
    base = base or load_config()
    s = base.model_dump()
    mkt = base.risk.market_ticker
    s["data"] |= {
        "tickers": [mkt],
        "market_tickers": list(dict.fromkeys([mkt, *base.data.market_tickers])),
        "universe": [],
        "sectors": {},
        "earnings": False,
        "cross_sectional": False,
        "start": START,
    }
    s["train"]["n_splits"] = N_SPLITS
    s["notify"] = {}
    s.pop("paths")
    cfg = Config(**s)
    d = run_dir()
    cfg.paths = PathsConfig(
        raw=d / "raw",
        clean=d / "clean",
        features=d / "features",
        models=d / "models",
        logs=base.paths.logs,
        history=d / "history",
    )
    return cfg


# --- summaries (pure functions on backtest curves and regime history) ----------------------------


def strategy_runs(curves: pd.DataFrame) -> dict[str, pd.DataFrame]:
    names = [c.removesuffix("_return") for c in curves.columns if c.endswith("_return")]
    return {n: pd.DataFrame({"weight": curves[f"{n}_weight"], "return": curves[f"{n}_return"]}) for n in names}


def window_stats(run: pd.DataFrame, start=None, end=None) -> dict | None:
    """Performance over [start, end] with equity restarted at 1; None if the window isn't covered."""
    from smp.allocation import performance

    r = run.loc[start:end]
    if len(r) < 2:
        return None
    r = r.assign(equity=(1 + r["return"]).cumprod())
    return performance(r) | {"start": str(r.index[0].date()), "end": str(r.index[-1].date())}


def crisis_table(runs: dict[str, pd.DataFrame]) -> list[dict]:
    """Return and worst drawdown of each strategy through each crisis the backtest fully covers."""
    any_run = next(iter(runs.values()))
    first, last = any_run.index[0], any_run.index[-1]
    rows = []
    for label, start, end in CRISES:
        if pd.Timestamp(start) < first or pd.Timestamp(end) > last:
            continue
        row = {"crisis": label, "start": start, "end": end}
        for n, run in runs.items():
            s = window_stats(run, start, end)
            row[n] = {"return": s["total_return"], "max_drawdown": s["max_drawdown"], "avg_weight": s["avg_weight"]}
        rows.append(row)
    return rows


def count_spells(labels: pd.Series, value: str) -> int:
    """Number of separate runs of consecutive days labelled `value`."""
    is_v = labels.eq(value)
    return int((is_v & ~is_v.shift(1, fill_value=False)).sum())


def fold_periods(cfg: Config, horizon: int) -> list[dict]:
    """Date range and R² of each walk-forward test period, for the chosen model and the baselines."""
    from smp.risk import risk_dir

    d = risk_dir(cfg, cfg.risk.market_ticker, horizon)
    oos = pd.read_parquet(d / "oos_predictions.parquet")
    vol = json.loads((d / "metrics.json").read_text())["vol"]
    out = []
    for i, g in oos.groupby("fold"):
        row = {"fold": int(i), "start": str(g.index.min().date()), "end": str(g.index.max().date())}
        for name in (vol["selected"], "persistence", "har"):
            row[name] = vol["cv_folds"][name][int(i)]["r2"]
        out.append(row | {"selected": vol["selected"]})
    return out


# --- run and report ------------------------------------------------------------------------------


def build_report(cfg: Config) -> dict:
    from smp.allocation import allocation_path
    from smp.regime import market_dir, validate

    mkt = cfg.risk.market_ticker
    spec = {"kind": "risk", "metric": "vol.r2"}
    chosen = fold_scores(cfg, spec)
    result: dict = {
        "name": NAME,
        "question": QUESTION,
        "kind": "risk",
        "metric": "vol.r2",
        "control": "default",
        "date": str(date.today()),
        "ticker": mkt,
        "start": START,
        "variants": {},
        "models": {},
        "strategies": {},
    }

    # volatility forecasts vs baselines, folds paired across both horizons (as in risk_vol_models)
    for other in ("persistence", "har"):
        alt = fold_scores(cfg, spec, model=other)
        result["models"][other] = {"mean": float(alt["value"].mean())} | _compare(chosen, alt)
    result["vol_r2"] = float(chosen["value"].mean())
    result["periods"] = {h: fold_periods(cfg, h) for h in cfg.risk.horizons}

    # allocation
    curves = pd.read_parquet(allocation_path(cfg, mkt) / "curves.parquet")
    runs = strategy_runs(curves)
    live = "model_vol_target" if cfg.risk.sizing_vol == "model" else "trailing_vol_target"
    result["control_strategy"] = {"name": live}
    for other in ("buy_hold", "model_vol_target", "model_vol_target_drop_filter"):
        if other != live and other in runs:
            v = _compare_sharpe(runs[live]["return"], runs[other]["return"])
            result["strategies"][other] = v | {"mean": window_stats(runs[other])["sharpe"]}
    split = pd.Timestamp(f"{SPLIT_YEAR}-01-01")
    result["allocation"] = {
        "all": {n: window_stats(r) for n, r in runs.items()},
        f"before_{SPLIT_YEAR}": {n: window_stats(r, end=split - pd.Timedelta(days=1)) for n, r in runs.items()},
        f"from_{SPLIT_YEAR}": {n: window_stats(r, start=split) for n, r in runs.items()},
    }
    result["control_strategy"] |= {
        "sharpe": result["allocation"]["all"][live]["sharpe"],
        "max_drawdown": result["allocation"]["all"][live]["max_drawdown"],
    }
    result["crises"] = crisis_table(runs)

    # regime
    hist = pd.read_parquet(market_dir(cfg) / "regime_history.parquet").dropna(subset=["regime"])
    result["regime"] = {
        "start": str(hist.index.min().date()),
        "end": str(hist.index.max().date()),
        "years": round(len(hist) / 252, 1),
        "outcomes": validate(hist),
        "stressed_spells": count_spells(hist["regime"], "stressed"),
        f"outcomes_before_{SPLIT_YEAR}": validate(hist.loc[: split - pd.Timedelta(days=1)]),
    }

    for group in ("models", "strategies"):
        adjusted = holm({k: v["p_value"] for k, v in result[group].items()})
        for k, v in result[group].items():
            v["p_holm"] = adjusted[k]
            v["verdict"] = verdict(v["diff"], adjusted[k], True)  # R² and Sharpe: higher is better
    return result


def _pct(x, digits=1) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}%}"


def render_markdown(r: dict) -> str:
    live = r["control_strategy"]["name"]
    shown = [s for s in STRATEGY_LABELS if s in r["allocation"]["all"]]
    a = r["allocation"]
    lines = [
        f"# {NAME}",
        "",
        f"**Question:** {QUESTION}",
        "",
        f"Run {r['date']} · {r['ticker']} only, data from {r['start']} · out-of-sample forecasts from "
        f"{a['all'][live]['start']} to {a['all'][live]['end']}",
        "",
        "The live system's data starts in 2010, so its backtests miss the 2000-02 and 2008 bear markets. This "
        f"rebuilds the {r['ticker']} risk models alone from {r['start']}, with {N_SPLITS} walk-forward periods. "
        "It has no cross-sectional ranks and no breadth input (both need the 2010+ stock universe).",
        "",
        "## Volatility forecasts",
        "",
        f"Mean walk-forward R² of log realised volatility (5 and 20 days): **{r['vol_r2']:.3f}**.",
        "",
        "| Alternative | Its R² | Δ chosen − alternative | 95% CI | p (Holm) | Better / worse periods | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, v in r["models"].items():
        lines.append(
            f"| {n} | {v['mean']:.3f} | {v['diff']:+.3f} | [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}] | "
            f"{_p(v['p_holm'])} | {v['wins']} / {v['losses']} | {v['verdict']} |"
        )
    for h, periods in r["periods"].items():
        sel = periods[0]["selected"]
        lines += [
            "",
            f"**{h}-day R² by test period** (chosen model: {sel})",
            "",
            f"| Period | {sel} | persistence | har |"
            if sel not in ("persistence", "har")
            else "| Period | persistence | har |",
            "|---|---|---|---|" if sel not in ("persistence", "har") else "|---|---|---|",
        ]
        for p in periods:
            cells = [p[sel]] if sel not in ("persistence", "har") else []
            cells += [p["persistence"], p["har"]]
            lines.append(f"| {p['start'][:7]} – {p['end'][:7]} | " + " | ".join(f"{c:.3f}" for c in cells) + " |")

    lines += [
        "",
        "## Volatility targeting",
        "",
        "| Period | Strategy | CAGR | Volatility | Sharpe | Max drawdown | Avg holding |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in (
        ("all", "All"),
        (f"before_{SPLIT_YEAR}", f"Before {SPLIT_YEAR}"),
        (f"from_{SPLIT_YEAR}", f"{SPLIT_YEAR} on"),
    ):
        for s in shown:
            st = a[key].get(s)
            if st is None:
                continue
            name = f"**{STRATEGY_LABELS[s]}** (default)" if s == live else STRATEGY_LABELS[s]
            lines.append(
                f"| {label} ({st['start'][:4]}–{st['end'][:4]}) | {name} | {_pct(st['cagr'])} | "
                f"{_pct(st['volatility'])} | {st['sharpe']:.2f} | {_pct(st['max_drawdown'])} | "
                f"{_pct(st['avg_weight'], 0)} |"
            )
    lines += [
        "",
        f"**`{live}` vs other strategies, whole period** (Sharpe difference, block bootstrap):",
        "",
        "| Alternative | Its Sharpe | Δ Sharpe | 95% CI | p (Holm) | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for n, v in r["strategies"].items():
        lines.append(
            f"| {n} | {v['mean']:.2f} | {v['diff']:+.2f} | [{v['ci_low']:+.2f}, {v['ci_high']:+.2f}] | "
            f"{_p(v['p_holm'])} | {v['verdict']} |"
        )
    lines += [
        "",
        "**Through each crisis** (return / worst drawdown, peak to trough of the S&P 500):",
        "",
        "| Crisis | Dates | " + " | ".join(STRATEGY_LABELS[s] for s in shown) + " | Avg holding (default) |",
        "|---|---|" + "---|" * len(shown) + "---|",
    ]
    for c in r["crises"]:
        cells = [f"{_pct(c[s]['return'], 0)} / {_pct(c[s]['max_drawdown'], 0)}" for s in shown]
        lines.append(
            f"| {c['crisis']} | {c['start']} – {c['end']} | "
            + " | ".join(cells)
            + f" | {_pct(c[live]['avg_weight'], 0)} |"
        )

    g = r["regime"]
    lines += [
        "",
        "## Market regime",
        "",
        f"Out of sample, {g['start']} to {g['end']} ({g['years']} years), without the breadth input. "
        f'"Stressed" spells: {g["stressed_spells"]}. What happened over the next 20 trading days:',
        "",
        "| Regime | Share of days | Volatility | Chance of a 5% drop | Average return |",
        "|---|---|---|---|---|",
    ]
    for name, o in g["outcomes"].items():
        lines.append(
            f"| {name.capitalize()} | {_pct(o['share_of_days'], 0)} | {_pct(o['next_20d_vol'], 0)} | "
            f"{_pct(o['next_20d_drop_rate'], 0)} | {o['next_20d_return']:+.1%} |"
        )
    lines += [
        "",
        "## Method",
        "",
        "- Volatility: walk-forward periods; the chosen model is paired with each baseline on the same horizon and "
        "period, with a cluster bootstrap over periods and Holm correction.",
        "- Allocation: daily returns from out-of-sample forecasts only, 0.1% cost per unit traded; the Sharpe "
        "comparison uses a 20-day moving-block bootstrap. Drawdowns are descriptive (no interval).",
        "- Crisis windows are fixed peak-to-trough dates, chosen before looking at results.",
        "",
        "Reproduce: `smp long-history`",
        "",
    ]
    return "\n".join(lines)


def run_long_history() -> dict:
    """Download, build, train and backtest the long history, then write the report."""
    from smp.allocation import backtest_all
    from smp.clean import clean_all
    from smp.features import build_all
    from smp.ingest import ingest_ticker
    from smp.regime import build_regime
    from smp.risk import train_risk_all

    cfg = long_config()
    for t in cfg.data.all_tickers:
        ingest_ticker(cfg, t)
    clean_all(cfg)
    build_all(cfg)
    train_risk_all(cfg)
    backtest_all(cfg)
    build_regime(cfg)
    result = build_report(cfg)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{NAME}.json").write_text(json.dumps(result, indent=2, default=float))
    (REPORTS_DIR / f"{NAME}.md").write_text(render_markdown(result))
    build_index()
    return result
