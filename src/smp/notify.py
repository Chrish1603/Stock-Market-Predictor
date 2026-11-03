"""Daily summary after each pipeline run, sent to wherever it's configured.

Content: market regime (flagged when it changes), and per stock the suggested holding, volatility
forecast, drop chance and upcoming earnings, plus what changed since the previous run.

Delivery (all optional except the log file):
- logs/summary.md (always) and the GitHub Actions run page when running in Actions
- macOS desktop notification (notify.macos: true)
- phone push via ntfy.sh (notify.ntfy_topic, or the SMP_NTFY_TOPIC environment variable)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import urllib.request

import numpy as np
import pandas as pd

from smp.config import Config
from smp.features import TRADING_DAYS, features_path
from smp.safeio import atomic_write_json

log = logging.getLogger(__name__)

STATE_FILE = "last_summary.json"
WEIGHT_CHANGE = 0.05  # report holding changes of at least 5 percentage points


def _vol_words(pct: float) -> str:
    if np.isnan(pct):
        return ""
    if pct >= 0.8:
        return "well above normal"
    if pct >= 0.6:
        return "above normal"
    return "normal" if pct > 0.4 else "below normal"


def build_summary(cfg: Config) -> dict:
    """Collect today's numbers and compare them with the previous run's."""
    rp_path = cfg.paths.models / "risk_predictions.parquet"
    if not rp_path.exists():
        raise FileNotFoundError("no risk predictions; run the pipeline first")
    rp = pd.read_parquet(rp_path)
    rp = rp[rp["horizon"] == cfg.risk.allocation_horizon]
    regime_path = cfg.paths.models / "_market" / "regime.json"
    regime = json.loads(regime_path.read_text()) if regime_path.exists() else None

    stocks = []
    for r in rp.itertuples():
        f = features_path(cfg, r.ticker)
        pct, earnings_in = float("nan"), None
        if f.exists():
            feats = pd.read_parquet(f)
            hist = (feats["vol_20"] * np.sqrt(TRADING_DAYS)).dropna()
            pct = float((hist < r.forecast_vol).mean()) if len(hist) else float("nan")
            last = feats.iloc[-1]
            if last.get("has_earnings", 0) == 1 and last.get("days_to_earnings", 99) <= 5:
                earnings_in = int(last["days_to_earnings"])
        stocks.append(
            {
                "ticker": r.ticker,
                "weight": float(r.suggested_weight),
                "vol": float(r.forecast_vol),
                "vol_pct": pct,
                "p_drop": float(r.p_drop),
                "earnings_in": earnings_in,
            }
        )
    as_of = str(pd.to_datetime(rp["as_of"]).max().date()) if len(rp) else None
    return {
        "as_of": as_of,
        "regime": regime["regime"] if regime else None,
        "market_vol": regime["forecast_vol"] if regime else None,
        "stocks": stocks,
    }


def changes(now: dict, before: dict | None) -> list[str]:
    if not before:
        return []
    out = []
    if before.get("regime") and now.get("regime") and before["regime"] != now["regime"]:
        out.append(f"Market regime changed: {before['regime']} → {now['regime']}")
    prev = {s["ticker"]: s for s in before.get("stocks", [])}
    for s in now["stocks"]:
        p = prev.get(s["ticker"])
        if p and abs(s["weight"] - p["weight"]) >= WEIGHT_CHANGE:
            out.append(f"{s['ticker']} holding {p['weight']:.0%} → {s['weight']:.0%}")
    return out


def render(now: dict, diff: list[str], problems: list[str] | None = None) -> tuple[str, str, str]:
    """(title, plain text for notifications, markdown for logs and GitHub). Problems always lead."""
    problems = problems or []
    regime = (now["regime"] or "unknown").capitalize()
    title = diff[0] if diff and diff[0].startswith("Market regime") else f"{regime} market · {now['as_of']}"
    if problems:
        title = f"⚠ {len(problems)} problem{'s' if len(problems) != 1 else ''}: {problems[0]}"
    lines = []
    for s in now["stocks"]:
        bits = [
            f"hold {s['weight']:.0%}",
            f"vol {s['vol']:.0%}" + (f" ({_vol_words(s['vol_pct'])})" if _vol_words(s["vol_pct"]) else ""),
            f"{s['p_drop']:.0%} chance of a 5% drop",
        ]
        if s["earnings_in"] is not None:
            bits.append(f"earnings in {s['earnings_in']}d")
        lines.append(f"{s['ticker']}: " + ", ".join(bits))
    text = "\n".join(
        ([" · ".join(problems), ""] if problems else []) + ([" · ".join(diff), ""] if diff else []) + lines
    )
    md = [f"### {title}", ""]
    if problems:
        md += ["**Problems:**", *[f"- ⚠ {x}" for x in problems], ""]
    if now["market_vol"] is not None:
        md.append(
            f"S&P 500 20-day volatility forecast: **{now['market_vol']:.0%}**. Based on the close of {now['as_of']}."
        )
        md.append("")
    if diff:
        md += ["**Changes since last run:** " + "; ".join(diff), ""]
    md += [f"- {line}" for line in lines]
    md += ["", "_Educational project, not financial advice._"]
    return title, text, "\n".join(md)


def send_macos(title: str, text: str) -> bool:
    if platform.system() != "Darwin":
        return False
    body = text.replace("\\", "").replace('"', "'")[:240]
    script = f'display notification "{body}" with title "Stock Market Predictor" subtitle "{title}"'
    return subprocess.run(["osascript", "-e", script], capture_output=True).returncode == 0


def send_ntfy(server: str, topic: str, title: str, text: str) -> bool:
    req = urllib.request.Request(
        f"{server.rstrip('/')}/{topic}",
        data=text.encode(),
        method="POST",
        headers={"Title": title.encode("ascii", "ignore").decode(), "Tags": "chart_with_upwards_trend"},
    )
    try:
        import ssl

        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            return resp.status == 200
    except Exception as exc:  # never fail the pipeline because a notification failed
        log.warning("ntfy notification failed: %s", exc)
        return False


def run_problems(cfg: Config) -> list[str]:
    """Stage failures from the last pipeline run plus stale or missing data."""
    from smp.health import data_problems

    problems = []
    status = cfg.paths.logs / "last_run.json"
    if status.exists():
        try:
            run = json.loads(status.read_text())
            problems += [f"stage '{f['stage']}' failed: {f['error']}" for f in run.get("failures", [])]
        except (json.JSONDecodeError, KeyError, TypeError):
            problems.append("last run status file is unreadable")
    try:
        problems += data_problems(cfg)
    except Exception as exc:  # a broken check must not stop the alert
        problems.append(f"data check failed: {exc}")
    return problems


def notify(cfg: Config) -> dict:
    """Build the summary, compare with the last run, write it out and send it. Problems are always sent."""
    problems = run_problems(cfg)
    try:
        now = build_summary(cfg)
    except FileNotFoundError as exc:
        if not problems:
            log.warning("no summary: %s", exc)
            return {}
        now = {"as_of": None, "regime": None, "market_vol": None, "stocks": []}
        problems.append(str(exc))
    state = cfg.paths.history / STATE_FILE
    try:
        before = json.loads(state.read_text()) if state.exists() else None
    except json.JSONDecodeError:
        before = None
    diff = changes(now, before)
    title, text, md = render(now, diff, problems)

    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    (cfg.paths.logs / "summary.md").write_text(md + "\n")
    if gh := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(gh, "a") as f:
            f.write(md + "\n")
    sent = []
    if cfg.notify.macos and send_macos(title, text):
        sent.append("macOS")
    topic = os.environ.get("SMP_NTFY_TOPIC") or cfg.notify.ntfy_topic
    if topic and send_ntfy(cfg.notify.ntfy_server, topic, title, text):
        sent.append("ntfy")
    if now["stocks"]:  # keep the last good state for change tracking
        cfg.paths.history.mkdir(parents=True, exist_ok=True)
        atomic_write_json(state, now)
    log.info("summary: %s%s", title, f" (sent via {', '.join(sent)})" if sent else "")
    return {"title": title, "text": text, "changes": diff, "problems": problems, "sent": sent}
