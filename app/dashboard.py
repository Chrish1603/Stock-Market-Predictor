"""Streamlit dashboard: predictions, price charts, model performance, backtest, data quality.

Run with:  streamlit run app/dashboard.py
Reads pipeline outputs only; training happens in the pipeline (`smp run-all`).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve

# The repo root is the folder above app/. Set it before importing smp, which reads it at import time
# (needed when smp is installed as a regular package, as on Streamlit Cloud).
os.environ.setdefault("SMP_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))


def drop_stale_smp_modules() -> None:
    """Streamlit keeps imported modules between reruns, and Streamlit Cloud pulls code updates without
    restarting. A new module importing from an old one already loaded then fails, so forget every smp
    module when any of its source files is newer than the import."""
    import sys
    import time

    if (pkg := sys.modules.get("smp")) is not None:
        loaded_at = getattr(pkg, "_dashboard_loaded_at", 0.0)
        newest = max((p.stat().st_mtime for p in Path(pkg.__file__).parent.glob("*.py")), default=0.0)
        if newest > loaded_at:
            for name in [m for m in sys.modules if m == "smp" or m.startswith("smp.")]:
                del sys.modules[name]
    import smp

    if not hasattr(smp, "_dashboard_loaded_at"):
        smp._dashboard_loaded_at = time.time()


drop_stale_smp_modules()

from smp import pipeline  # noqa: E402
from smp.allocation import allocation_path  # noqa: E402
from smp.clean import clean_path, report_path  # noqa: E402
from smp.config import load_config  # noqa: E402
from smp.features import bollinger, features_path, macd, rsi  # noqa: E402
from smp.history import load_history, summarize  # noqa: E402
from smp.predict import predictions_path  # noqa: E402
from smp.regime import market_dir  # noqa: E402
from smp.risk import risk_dir, risk_predictions_path  # noqa: E402
from smp.risk_history import load_risk_history, summarize_risk  # noqa: E402
from smp.train import model_dir  # noqa: E402

st.set_page_config(page_title="Stock Market Predictor", layout="wide")


@st.cache_resource(ttl=3600, show_spinner="Downloading the latest data snapshot...")
def download_snapshot(url: str) -> str:
    """Fetch and unpack a published snapshot (see `smp publish`); refreshed at most hourly."""
    from smp.publish import fetch_snapshot

    return str(fetch_snapshot(url))


def choose_data_root() -> bool:
    """Use local pipeline output if present; otherwise the published snapshot. True means snapshot mode."""
    if os.environ.get("SMP_DATA_ROOT") and not os.environ.get("SMP_SNAPSHOT"):
        return False  # set explicitly by the user
    os.environ.pop("SMP_DATA_ROOT", None)
    local = load_config()
    forced = os.environ.get("SMP_SNAPSHOT_URL")  # always use this snapshot (hosting, testing)
    url = forced or local.dashboard.data_url
    if not url or (not forced and (local.paths.models / "risk_predictions.parquet").exists()):
        return False
    os.environ["SMP_DATA_ROOT"] = download_snapshot(url)
    os.environ["SMP_SNAPSHOT"] = "1"
    return True


try:
    SNAPSHOT = choose_data_root()
except Exception as exc:  # network or GitHub problem: explain instead of showing a traceback
    st.error(
        "Couldn't download the latest data snapshot, so there's nothing to show yet. "
        "This is usually a temporary network or GitHub problem."
    )
    st.caption(f"Details: {type(exc).__name__}: {exc}")
    if st.button("Try again"):
        download_snapshot.clear()
        st.rerun()
    st.stop()
CFG = load_config()

# --- colours: validated reference palette, stepped separately for light and dark -------------
PALETTES = {
    "light": dict(
        s1="#2a78d6",
        s2="#eb6834",
        s3="#1baf7a",
        up="#2a78d6",
        down="#e34948",
        muted="#898781",
        grid="#e1e0d9",
        band="rgba(42,120,214,0.10)",
        seq=["#cde2fb", "#86b6ef", "#2a78d6", "#184f95"],
    ),
    "dark": dict(
        s1="#3987e5",
        s2="#d95926",
        s3="#199e70",
        up="#3987e5",
        down="#e66767",
        muted="#898781",
        grid="#2c2c2a",
        band="rgba(57,135,229,0.15)",
        seq=["#104281", "#256abf", "#5598e7", "#b7d3f6"],
    ),
}
theme_type = getattr(getattr(st.context, "theme", None), "type", None) or "light"
C = PALETTES["dark" if theme_type == "dark" else "light"]
FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
UP_TEXT, DOWN_TEXT = ("#0ca30c", "#e66767") if theme_type == "dark" else ("#006300", "#c03030")

st.markdown(
    f"""
    <style>
      [data-testid="stSidebar"] h1 {{ font-size: 1.3rem; letter-spacing: -0.01em; }}
      .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }}
      [data-testid="stHeader"] {{ background: transparent; }}
      h1, h2, h3 {{ letter-spacing: -0.015em; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; }}
      .stTabs [data-baseweb="tab"] {{ font-size: 0.95rem; padding-left: 0; padding-right: 0; }}
      [data-testid="stMetricValue"] {{ letter-spacing: -0.02em; }}
      .eyebrow {{ text-transform: uppercase; letter-spacing: 0.09em; font-size: 0.72rem; font-weight: 600;
                 opacity: 0.6; }}
      .muted {{ opacity: 0.65; }}
      .hero {{ border-radius: 0.9rem; padding: 1.25rem 1.5rem; margin: 0.25rem 0 1.25rem 0; }}
      .hero-title {{ font-size: 1.9rem; font-weight: 650; letter-spacing: -0.02em; line-height: 1.2; }}
      .chips {{ display: flex; gap: 2.2rem; flex-wrap: wrap; margin-top: 0.9rem; }}
      .chip-label {{ font-size: 0.75rem; opacity: 0.65; }}
      .chip-value {{ font-size: 1.05rem; font-weight: 600; }}
      .card-head {{ display: flex; justify-content: space-between; align-items: baseline; }}
      .ticker {{ font-size: 1.15rem; font-weight: 650; letter-spacing: -0.01em; }}
      .price {{ font-size: 0.95rem; opacity: 0.8; margin-left: 0.5rem; }}
      .delta {{ font-size: 0.85rem; font-weight: 600; }}
      .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 0.25rem; }}
      .stat-label {{ font-size: 0.74rem; opacity: 0.65; }}
      .stat-value {{ font-size: 1.35rem; font-weight: 650; letter-spacing: -0.02em; line-height: 1.3; }}
      .stat-note {{ font-size: 0.74rem; opacity: 0.65; margin-top: 0.15rem; }}
      .bar {{ height: 5px; border-radius: 3px; background: rgba(128,128,128,0.18); margin-top: 0.35rem;
              overflow: hidden; }}
      .bar > span {{ display: block; height: 100%; border-radius: 3px; background: {C["s1"]}; }}
      .card-text {{ font-size: 0.86rem; line-height: 1.45; margin-top: 0.75rem; opacity: 0.85; }}
      .card-foot {{ font-size: 0.74rem; opacity: 0.55; margin-top: 0.6rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)
# Status colours are reserved for regime state and always shown with an icon + label.
REGIME_STYLE = {
    "calm": ("#0ca30c", "●", "Calm"),
    "normal": (C["muted"], "◐", "Normal"),
    "stressed": ("#d03b3b", "▲", "Stressed"),
}


FEATURE_NAMES = {
    "macd": "MACD",
    "macd_hist": "MACD histogram",
    "rsi_14": "RSI (14)",
    "stoch_k": "Stochastic %K",
    "atr_14": "Average daily range (ATR)",
    "bb_pctb": "Position in Bollinger band",
    "hl_range": "Today's high-low range",
    "volume_z": "Volume vs normal",
    "obv_change_10": "On-balance volume, 10d",
    "day_of_week": "Day of week",
    "month": "Month",
    "ewm_vol_20": "Volatility (EWMA)",
    "downside_vol_20": "Downside volatility",
    "parkinson_vol_20": "Range-based volatility",
    "vol_ratio_5_60": "Volatility, 5d vs 60d",
    "drawdown_60": "Drop from 60-day high",
    "days_to_earnings": "Days to earnings",
    "days_since_earnings": "Days since earnings",
    "last_eps_surprise": "Last earnings surprise",
    "has_earnings": "Has earnings",
    "vix_level": "VIX level",
    "vix_change_5": "VIX change, 5d",
    "yield_curve_change_20": "Yield curve change, 20d",
    "yield_curve_inverted": "Yield curve inverted",
    "sector_sma_ratio_50": "Sector vs its 50-day average",
}
METRIC_NAMES = {
    "roc_auc": "ROC-AUC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "log_loss": "Log loss",
    "up_rate": "Share of up days",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "directional_accuracy": "Direction right",
}


def feature_label(name: str) -> str:
    """Readable name for a model input, e.g. 'sma_ratio_50' -> 'Price vs 50-day average'."""
    if name in FEATURE_NAMES:
        return FEATURE_NAMES[name]
    if name.startswith("xs_rank_"):
        return f"Rank among stocks: {feature_label(name[8:]).lower()}"
    patterns = [
        (r"ret_lag_1", "Return yesterday"),
        (r"ret_lag_(\d+)", "Return {} days ago"),
        (r"ret_(\d+)", "Return, past {}d"),
        (r"sma_ratio_(\d+)", "Price vs {}-day average"),
        (r"ema_ratio_(\d+)", "Price vs {}-day EMA"),
        (r"vol_(\d+)", "Volatility, {}d"),
        (r"spy_ret_(\d+)", "S&P 500 return, {}d"),
        (r"(tnx|irx)_change_(\d+)", "{} yield change, {}d"),
        (r"sector_ret_(\d+)", "Sector return, {}d"),
        (r"rel_strength_(\d+)", "Strength vs sector, {}d"),
    ]
    for pattern, text in patterns:
        if m := re.fullmatch(pattern, name):
            parts = [{"tnx": "10-year", "irx": "3-month"}.get(g, g) for g in m.groups()]
            return text.format(*parts)
    return name


def as_text(df: pd.DataFrame, formats: dict[str, str]) -> pd.DataFrame:
    """Format columns as strings with '–' for missing values (st.dataframe ignores Styler na_rep)."""
    out = df.copy().astype(object)
    for col in out.columns:
        fmt = formats.get(col)
        out[col] = [
            "–"
            if v is None or v is pd.NaT or (isinstance(v, float) and np.isnan(v))
            else (fmt.format(v) if fmt else str(v))
            for v in df[col]
        ]
    return out


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    """House style: transparent background, left-aligned title, hairline grid, legend below."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=52, b=8),
        font=dict(family=FONT, size=12),
        title=dict(x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=15, family=FONT)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        legend=dict(orientation="h", y=-0.1, yanchor="top", x=0, font=dict(size=12)),
    )
    fig.update_xaxes(showgrid=False, linecolor=C["grid"], tickfont=dict(color=C["muted"]), ticks="")
    fig.update_yaxes(gridcolor=C["grid"], gridwidth=1, zeroline=False, tickfont=dict(color=C["muted"]))
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme="streamlit")


# --- data loading ------------------------------------------------------------------------------


@st.cache_data
def load_prices(ticker: str) -> pd.DataFrame | None:
    p = clean_path(CFG, ticker)
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    p = predictions_path(CFG)
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_metrics(ticker: str, horizon: int) -> dict | None:
    p = model_dir(CFG, ticker, horizon) / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_artifact(ticker: str, horizon: int, name: str) -> pd.DataFrame | None:
    p = model_dir(CFG, ticker, horizon) / name
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_track_record() -> pd.DataFrame:
    return load_history(CFG)


@st.cache_data
def load_risk_track_record() -> pd.DataFrame:
    return load_risk_history(CFG)


@st.cache_data
def load_json(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_parquet(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_quality() -> dict:
    p = report_path(CFG)
    return json.loads(p.read_text()) if p.exists() else {}


# --- sidebar -----------------------------------------------------------------------------------

st.sidebar.title("Stock Market Predictor")
st.sidebar.caption("Risk forecasts and position sizing for US stocks, with direction models for comparison.")
ticker = st.sidebar.selectbox("Stock", CFG.data.tickers, help="Used by the Price, Direction and Track record tabs")
_latest = load_prices(CFG.data.tickers[0])
if _latest is not None:
    st.sidebar.markdown(f"**Data as of** {_latest.index[-1]:%a %d %b %Y}")
if SNAPSHOT:
    built = (load_json(str(Path(os.environ["SMP_DATA_ROOT"]) / "manifest.json")) or {}).get("generated_at", "?")
    st.sidebar.caption(f"Hosted snapshot, rebuilt each US trading day (last {built[:16].replace('T', ' ')} UTC).")
elif st.sidebar.button(
    "Refresh data", help="Downloads new prices and updates forecasts. Does not retrain.", width="stretch"
):
    from smp.safeio import PipelineBusy

    with st.spinner("Downloading prices and updating forecasts..."):
        pipeline.setup_logging(CFG.paths.logs)
        try:
            pipeline.refresh(CFG)
            st.sidebar.success("Updated")
        except PipelineBusy:
            st.sidebar.info("The scheduled run is in progress; try again in a few minutes.")
        except pipeline.PipelineError as exc:
            st.sidebar.warning(f"Partly updated. Failed: {', '.join(f['stage'] for f in exc.failures)}.")
    st.cache_data.clear()
st.sidebar.caption("Models retrain in the daily scheduled run (`smp run-all`).")
st.sidebar.divider()
st.sidebar.caption("Educational project, not financial advice.")


def horizon_label(h: int) -> str:
    return f"{h} day" if h == 1 else f"{h} days"


prices = load_prices(ticker)
if prices is None:
    st.warning("No data yet. Run `smp run-all` first.")
    st.stop()

# Warn visitors when the daily update has stopped
_rp = load_parquet(str(risk_predictions_path(CFG)))
if _rp is not None and len(_rp):
    from smp.health import business_days_behind, is_stale

    _as_of = pd.to_datetime(_rp["as_of"]).max().date()
    if is_stale(_as_of):
        _n = business_days_behind(_as_of)
        st.warning(
            f"⚠ Data is {_n} trading day{'s' if _n != 1 else ''} out of date (last close {_as_of:%a %d %b}). "
            "The daily update may have failed; forecasts below are from that close.",
        )

tab_today, tab_risk, tab_price, tab_dir, tab_track, tab_quality = st.tabs(
    ["Today", "Risk & allocation", "Price & indicators", "Direction models", "Track record", "Data quality"]
)
tab_model = tab_bt = tab_dir  # model performance and its backtest share one tab

# --- today ------------------------------------------------------------------------------------


def sparkline(series: pd.Series) -> go.Figure:
    lo, hi = float(series.min()), float(series.max())
    pad = (hi - lo) * 0.08 or 1.0
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            line=dict(color=C["s1"], width=2, shape="spline", smoothing=0.4),
            fill="tozeroy",
            fillcolor=C["band"],
            hovertemplate="%{x|%d %b}: %{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=70,
        margin=dict(l=0, r=0, t=2, b=2),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[lo - pad, hi + pad]),
    )
    return fig


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def risk_words(pct: float) -> str:
    if pct >= 0.8:
        return "well above normal"
    if pct >= 0.6:
        return "above normal"
    if pct > 0.4:
        return "normal"
    return "below normal" if pct > 0.2 else "low"


def regime_banner(regime: dict) -> None:
    """Hero card: regime label (icon + word, tinted with its status colour) and its three inputs."""
    color, icon, label = REGIME_STYLE[regime["regime"]]
    breadth = f"{regime['breadth']:.0%}" if regime["breadth"] is not None else "n/a"
    chips = [
        (
            "S&P 500 volatility forecast",
            f"{regime['forecast_vol']:.0%}",
            f"{ordinal(round(regime['vol_percentile'] * 100))} percentile",
        ),
        ("Stocks above 200-day average", breadth, "breadth"),
        ("S&P 500 trend", "Up" if regime["trend_up"] else "Down", "vs 200-day average"),
    ]
    chip_html = "".join(
        f"<div><div class='chip-label'>{a}</div><div class='chip-value'>{b} "
        f"<span class='chip-label'>{c}</span></div></div>"
        for a, b, c in chips
    )
    st.markdown(
        f"<div class='hero' style='background:{color}14;border:1px solid {color}33'>"
        f"<div class='eyebrow'>Market regime · {regime['as_of']}</div>"
        f"<div class='hero-title'><span style='color:{color}'>{icon}</span> {label}</div>"
        f"<div class='chips'>{chip_html}</div></div>",
        unsafe_allow_html=True,
    )


def stat_html(label: str, value: str, note: str = "", bar: float | None = None) -> str:
    bar_html = (
        f"<div class='bar'><span style='width:{max(0.0, min(bar, 1.0)) * 100:.0f}%'></span></div>"
        if bar is not None
        else ""
    )
    return (
        f"<div><div class='stat-label'>{label}</div><div class='stat-value'>{value}</div>{bar_html}"
        f"<div class='stat-note'>{note}</div></div>"
    )


with tab_today:
    rc = CFG.risk
    rpreds = load_parquet(str(risk_predictions_path(CFG)))
    regime = load_json(str(market_dir(CFG) / "regime.json"))
    dpreds = load_predictions()
    if rpreds is None or rpreds.empty:
        st.info("No forecasts yet. Run `smp run-all`.")
    else:
        if regime:
            regime_banner(regime)
        h_alloc, h_dir = rc.allocation_horizon, max(CFG.train.horizons)
        grid: list = []
        for i, t in enumerate(CFG.data.tickers):
            if i % 2 == 0:  # a new row of two cards, so cards line up across the row
                grid = st.columns(2, gap="medium")
            r = rpreds[(rpreds["ticker"] == t) & (rpreds["horizon"] == h_alloc)]
            px_t = load_prices(t)
            if r.empty or px_t is None:
                continue
            r = r.iloc[0]
            close = px_t["Close"]
            feats = load_parquet(str(features_path(CFG, t)))
            hist_vol = (feats["vol_20"] * np.sqrt(252)).dropna() if feats is not None else pd.Series(dtype=float)
            pct = float((hist_vol < r.forecast_vol).mean()) if len(hist_vol) else float("nan")
            rmeta = load_json(str(risk_dir(CFG, t, h_alloc) / "metrics.json"))
            normal_drop = rmeta["drop"]["cv"]["baseline"]["drop_rate"] if rmeta else float("nan")
            d = dpreds[(dpreds["ticker"] == t) & (dpreds["horizon"] == h_dir)] if dpreds is not None else pd.DataFrame()
            dmeta = load_metrics(t, h_dir)
            last = feats.iloc[-1] if feats is not None else None
            earnings_in = (
                int(last["days_to_earnings"])
                if last is not None and last.get("has_earnings", 0) == 1 and last["days_to_earnings"] <= 10
                else None
            )

            # sentence
            ratio = r.p_drop / normal_drop if normal_drop else 1.0
            likely = (
                "more likely than usual"
                if ratio > 1.2
                else "less likely than usual"
                if ratio < 0.8
                else ("about as likely as usual")
            )
            basis = "last month's" if rc.sizing_vol == "trailing" else "the forecast"
            basis_vol = r.trailing_vol if rc.sizing_vol == "trailing" else r.forecast_vol
            text = (
                f"Risk is {risk_words(pct)} for {t}"
                + (
                    f", with earnings in {earnings_in} trading day{'s' if earnings_in != 1 else ''}"
                    if earnings_in
                    else ""
                )
                + f". A {rc.drop_threshold:.0%} drop is {likely}. "
                f"Holding sized to {rc.target_vol:.0%} volatility using {basis} {basis_vol:.0%}."
            )
            if rc.sizing_vol == "trailing" and r.forecast_vol > 1.25 * r.trailing_vol:
                alt = min(rc.target_vol / r.forecast_vol, rc.max_weight)
                text += f" The model expects more turbulence; sizing on its forecast would mean <b>{alt:.0%}</b>."

            change = close.iloc[-1] / close.iloc[-22] - 1
            arrow, tone = ("▲", UP_TEXT) if change >= 0 else ("▼", DOWN_TEXT)
            direction = ""
            if not d.empty and dmeta:
                dd_ = d.iloc[0]
                auc = dmeta["classifier"]["cv"][dmeta["classifier"]["selected"]]["roc_auc"]
                direction = (
                    f"Direction model ({h_dir} days): {'up' if dd_.direction == 'up' else 'down'} {dd_.p_up:.0%} · "
                    f"weak signal, score {auc:.2f} where 0.50 is a coin flip"
                )

            with grid[i % 2].container(border=True):
                st.markdown(
                    f"<div class='card-head'><div><span class='ticker'>{t}</span>"
                    f"<span class='price'>{close.iloc[-1]:,.2f}</span></div>"
                    f"<div class='delta'><span style='color:{tone}'>{arrow} {abs(change):.1%}</span> "
                    f"<span class='muted' style='font-weight:400'>past month</span></div></div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    sparkline(close.iloc[-63:]), width="stretch", config={"displayModeBar": False}, key=f"spark_{t}"
                )
                st.markdown(
                    "<div class='stats'>"
                    + stat_html(
                        f"Volatility, next {h_alloc}d",
                        f"{r.forecast_vol:.0%}",
                        f"{risk_words(pct)} · {ordinal(round(pct * 100))} pct" if not np.isnan(pct) else "",
                        pct if not np.isnan(pct) else None,
                    )
                    + stat_html(
                        f"Chance of a {rc.drop_threshold:.0%} drop",
                        f"{r.p_drop:.0%}",
                        "the usual rate (no model beat it)"
                        if getattr(r, "drop_model", None) == "baseline"
                        else f"usually {normal_drop:.0%}",
                        r.p_drop,
                    )
                    + stat_html(
                        "Suggested holding",
                        f"{r.suggested_weight:.0%}",
                        "the maximum" if r.suggested_weight >= rc.max_weight - 1e-9 else "of a full position",
                        r.suggested_weight / rc.max_weight,
                    )
                    + "</div>"
                    + f"<div class='card-text'>{text}</div>"
                    + (f"<div class='card-foot'>{direction}</div>" if direction else ""),
                    unsafe_allow_html=True,
                )

        as_of = pd.to_datetime(rpreds["as_of"]).max().date()
        st.caption(
            f"Based on the close of {as_of:%A %d %B %Y}. Volatility forecasts beat the standard baselines in testing "
            "and drop chances are calibrated against history; direction calls are close to a coin flip. "
            "Educational project, not financial advice."
        )

# --- risk & allocation -------------------------------------------------------------------------

STRATEGY_LABELS = {
    "buy_hold": "Buy & hold",
    "trailing_vol_target": "Vol target (trailing vol)",
    "model_vol_target": "Vol target (model forecast)",
    "model_vol_target_drop_filter": "Vol target + drop filter",
}
SHOWN_STRATEGIES = ["buy_hold", "trailing_vol_target", "model_vol_target"]

with tab_risk:
    rc = CFG.risk
    regime = load_json(str(market_dir(CFG) / "regime.json"))
    rpreds = load_parquet(str(risk_predictions_path(CFG)))
    if not rc.enabled or rpreds is None:
        st.info("No risk forecasts yet. Run `smp run-all`.")
    else:
        # Risk forecasts per ticker
        if regime:
            regime_banner(regime)
            st.caption(
                "Stressed = vol forecast in its top 20%, or weak breadth in a downtrend. Calm = vol in its bottom 40% "
                "with healthy breadth in an uptrend. A risk signal, not a sell signal: returns after stressed days "
                "were on average higher (rebounds), but swings were much larger."
            )
        st.subheader("Risk forecasts")
        rh = (
            st.segmented_control(
                "Risk horizon", rc.horizons, default=rc.horizons[-1], format_func=horizon_label, key="risk_h"
            )
            or rc.horizons[-1]
        )
        rp = rpreds[rpreds["horizon"] == rh].set_index("ticker")
        cols = st.columns(len(rp))
        for col, (t, r) in zip(cols, rp.iterrows(), strict=False):
            col.metric(
                t,
                f"{r.forecast_vol:.0%} vol",
                f"{r.forecast_vol - r.trailing_vol:+.0%} vs last month",
                delta_color="inverse",
            )
            col.caption(f"{r.p_drop:.0%} chance of a {r.drop_threshold:.0%} drop · hold **{r.suggested_weight:.0%}**")
        st.caption(
            f"Volatility is annualised. Suggested holding = {rc.target_vol:.0%} target vol ÷ "
            f"{'last month' if rc.sizing_vol == 'trailing' else 'forecast'}'s vol, capped at {rc.max_weight:.0%} "
            f"(based on the {rc.allocation_horizon}-day horizon)."
        )

        # Forecast vs realised volatility (out of sample)
        oos_parts = [
            load_parquet(str(risk_dir(CFG, ticker, rh) / f))
            for f in ("oos_predictions.parquet", "holdout_predictions.parquet")
        ]
        rmeta = load_json(str(risk_dir(CFG, ticker, rh) / "metrics.json"))
        if all(x is not None for x in oos_parts) and rmeta:
            oos = pd.concat(oos_parts).sort_index()
            oos = oos[~oos.index.duplicated(keep="last")]
            if not st.toggle("Show full history (2013 onwards)", key="vol_full"):
                oos = oos.loc[oos.index[-1] - pd.DateOffset(years=2) :]
            vm = rmeta["vol"]
            fig = go.Figure()
            for col, name, color, width in (
                ("actual_logvol", "Realised (what happened)", C["muted"], 1),
                ("trailing_logvol", "Last month's vol", C["s2"], 2),
                ("pred_logvol", f"Forecast ({vm['selected']})", C["s1"], 2),
            ):
                fig.add_trace(
                    go.Scatter(x=oos.index, y=np.exp(oos[col]), name=name, line=dict(color=color, width=width))
                )
            fig.update_layout(title=f"{ticker}: {rh}-day volatility forecast vs what happened (out of sample)")
            fig.update_yaxes(tickformat=".0%")
            show(style(fig, height=380))
            sel = vm["selected"]
            st.caption(
                "Last month's volatility (orange) reacts late: it stays high after turbulence ends and low as it "
                "starts. The forecast (blue) adjusts sooner. "
                f"Walk-forward CV R² on log volatility: forecast {vm['cv'][sel]['r2']:.2f}, HAR "
                f"{vm['cv']['har']['r2']:.2f}, last month's vol {vm['cv']['persistence']['r2']:.2f}. "
                f"Average error {vm['cv'][sel]['mae_vol_points']:.1%} points."
            )

        # Allocation backtest
        st.subheader("Volatility-targeted allocation, 2013 to today (out of sample)")
        port_ok = (CFG.paths.models / "_portfolio" / "allocation" / "stats.json").exists()
        show_opts = ([f"All {len(CFG.data.tickers)} (equal weight)"] if port_ok else []) + [ticker]
        choice = st.segmented_control("Show", show_opts, default=show_opts[0], key="alloc_show") or show_opts[0]
        base = CFG.paths.models / "_portfolio" / "allocation" if choice != ticker else allocation_path(CFG, ticker)
        curves, stats = load_parquet(str(base / "curves.parquet")), load_json(str(base / "stats.json"))
        if curves is None or stats is None:
            st.info("No allocation backtest yet. Run `smp train`.")
        else:
            eq = go.Figure()
            dd = go.Figure()
            for s, color in zip(SHOWN_STRATEGIES, (C["s2"], C["s1"], C["s3"]), strict=True):
                e = curves[f"{s}_equity"]
                eq.add_trace(go.Scatter(x=e.index, y=e, name=STRATEGY_LABELS[s], line=dict(color=color, width=2)))
                dd.add_trace(
                    go.Scatter(
                        x=e.index, y=e / e.cummax() - 1, name=STRATEGY_LABELS[s], line=dict(color=color, width=2)
                    )
                )
            eq.update_layout(title="Growth of $1 (log scale)")
            top = float(curves[[f"{s}_equity" for s in SHOWN_STRATEGIES]].max().max())
            ticks = [v for v in (0.5, 1, 2, 5, 10, 20, 50, 100, 200) if v <= top * 1.2]
            eq.update_yaxes(type="log", tickvals=ticks, ticktext=[f"${v:g}" for v in ticks], minor=dict(showgrid=False))
            dd.update_layout(title="Drawdown from peak")
            dd.update_yaxes(tickformat=".0%")
            show(style(eq, height=380))
            show(style(dd, height=280))
            table = pd.DataFrame({STRATEGY_LABELS[s]: stats[s] for s in STRATEGY_LABELS if s in stats}).T
            table = table[["cagr", "volatility", "sharpe", "max_drawdown", "calmar", "avg_weight", "turnover_per_year"]]
            table.columns = ["CAGR", "Volatility", "Sharpe", "Max drawdown", "Calmar", "Avg invested", "Turnover/yr"]
            st.dataframe(
                table.style.format(
                    {
                        "CAGR": "{:.1%}",
                        "Volatility": "{:.1%}",
                        "Sharpe": "{:.2f}",
                        "Max drawdown": "{:.1%}",
                        "Calmar": "{:.2f}",
                        "Avg invested": "{:.0%}",
                        "Turnover/yr": "{:.1f}",
                    }
                ),
                width="stretch",
            )
            st.caption(
                "Uses only forecasts made before each day (walk-forward folds, then the hold-out year), "
                f"{CFG.backtest.cost:.1%} cost per unit traded, no leverage. Vol targeting holds cash part of the "
                "time, so it gives up return in strong bull markets in exchange for much smaller drawdowns."
            )

        # Regime history
        rhist = load_parquet(str(market_dir(CFG) / "regime_history.parquet"))
        rval = load_json(str(market_dir(CFG) / "regime_validation.json"))
        mkt_prices = load_prices(CFG.risk.market_ticker)
        if rhist is not None and rval and mkt_prices is not None:
            st.subheader("Regime history (out of sample)")
            h = rhist.dropna(subset=["regime"])
            px_m = mkt_prices["Close"].reindex(h.index)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.85, 0.15], vertical_spacing=0.03)
            fig.add_trace(
                go.Scatter(x=px_m.index, y=px_m, name=CFG.risk.market_ticker, line=dict(color=C["s1"], width=2)), 1, 1
            )
            codes = h["regime"].map({r: i for i, r in enumerate(REGIME_STYLE)})
            scale = [[i / 2, REGIME_STYLE[r][0]] for i, r in enumerate(REGIME_STYLE)]
            fig.add_trace(
                go.Heatmap(
                    x=h.index,
                    y=["Regime"],
                    z=[codes.to_numpy()],
                    text=[h["regime"].to_numpy()],
                    colorscale=scale,
                    zmin=0,
                    zmax=2,
                    showscale=False,
                    hovertemplate="%{x|%d %b %Y}: %{text}<extra></extra>",
                ),
                2,
                1,
            )
            fig.update_layout(
                title=f"{CFG.risk.market_ticker} with each day's regime (green ● calm, grey ◐ normal, red ▲ stressed)",
                showlegend=False,
            )
            show(style(fig, height=380))
            vt = pd.DataFrame(rval).T.reindex(list(REGIME_STYLE))
            vt.index = [f"{REGIME_STYLE[r][1]} {REGIME_STYLE[r][2]}" for r in vt.index]
            vt.columns = ["Share of days", "Next 20d volatility", "Next 20d return", "Chance of 5% drop in next 20d"]
            st.dataframe(vt.style.format("{:.1%}"), width="stretch")

# --- live track record -------------------------------------------------------------------------

with tab_track:
    st.subheader("Live track record")
    st.caption(
        "Every forecast is saved when it's made and scored once its horizon has passed. Unlike the backtests, "
        "these were made before the outcome existed, so they can't have been tuned to it."
    )

    st.markdown("#### Risk forecasts")
    st.caption(
        "Volatility forecasts are compared with what volatility turned out to be, and with simply quoting last "
        "month's volatility. Drop chances are compared with what happened, and with quoting the historical "
        "drop rate. Lower errors and Brier scores are better."
    )
    rhist = load_risk_track_record()
    risk_track_h = (
        st.segmented_control(
            "Risk horizon",
            CFG.risk.horizons,
            default=CFG.risk.horizons[0],
            format_func=horizon_label,
            key="track_risk_h",
        )
        or CFG.risk.horizons[0]
    )
    if rhist.empty:
        st.info("No risk forecasts recorded yet. Run `smp predict` (or `smp run-all`).")
    else:
        rsum = summarize_risk(rhist)
        rsum = rsum[rsum["horizon"] == risk_track_h].set_index("ticker")
        rsum = rsum.assign(first=rsum["first"].dt.date, last=rsum["last"].dt.date)
        rsum = rsum[
            [
                "predictions",
                "scored",
                "vol_error",
                "trailing_vol_error",
                "closer_share",
                "drops",
                "brier",
                "base_rate_brier",
                "first",
                "last",
            ]
        ]
        rsum.columns = [
            "Forecasts",
            "Scored",
            "Vol off by",
            "Last month's vol off by",
            "Forecast closer",
            "Drops seen",
            "Drop Brier",
            "Historical-rate Brier",
            "First",
            "Latest",
        ]
        rsum.index.name = "Stock"
        st.dataframe(
            as_text(
                rsum,
                {
                    "Vol off by": "{:.0%}",
                    "Last month's vol off by": "{:.0%}",
                    "Forecast closer": "{:.0%}",
                    "Drop Brier": "{:.3f}",
                    "Historical-rate Brier": "{:.3f}",
                },
            ),
            width="stretch",
        )
        rt = rhist[(rhist["ticker"] == ticker) & (rhist["horizon"] == risk_track_h)].sort_values("as_of")
        rscored = rt.dropna(subset=["realised_vol"])
        if len(rscored) < 2:
            pending = rt[rt["realised_vol"].isna()]
            first_due = (
                f" The first is scored after the close on about "
                f"{(pending['as_of'].min() + pd.offsets.BDay(risk_track_h)):%a %d %b}."
                if len(pending) and rscored.empty
                else ""
            )
            st.info(
                f"{ticker}: {len(rt)} {horizon_label(risk_track_h)} forecast{'s' if len(rt) != 1 else ''} recorded, "
                f"{len(rscored)} scored so far.{first_due} The chart appears once a few have been scored."
            )
        else:
            fig = go.Figure()
            for col, name, color, dash in (
                ("realised_vol", "What happened", C["s1"], None),
                ("forecast_vol", "Forecast", C["s2"], None),
                ("trailing_vol", "Last month's vol", C["muted"], "dot"),
            ):
                fig.add_trace(
                    go.Scatter(
                        x=rscored["as_of"], y=rscored[col], name=name, line=dict(color=color, width=2, dash=dash)
                    )
                )
            fig.update_layout(title=f"{ticker} volatility over the next {horizon_label(risk_track_h)}, by forecast day")
            fig.update_yaxes(tickformat=".0%")
            show(style(fig, height=360))
        if not rt.empty:
            recent = rt.sort_values("as_of", ascending=False).assign(
                as_of=lambda d: d["as_of"].dt.date,
                target_date=lambda d: pd.to_datetime(d["target_date"]).dt.date,
                dropped=lambda d: d["dropped"].map({1.0: "yes", 0.0: "no"}).fillna("pending"),
            )
            recent = recent[
                ["as_of", "forecast_vol", "trailing_vol", "realised_vol", "p_drop", "dropped", "target_date"]
            ]
            recent.columns = [
                "Made on",
                "Forecast vol",
                "Last month's vol",
                "What happened",
                "Drop chance",
                "Dropped?",
                "Scored on",
            ]
            st.markdown(f"**{ticker}: every {risk_track_h}-day risk forecast**")
            st.dataframe(
                as_text(
                    recent,
                    {
                        "Forecast vol": "{:.0%}",
                        "Last month's vol": "{:.0%}",
                        "What happened": "{:.0%}",
                        "Drop chance": "{:.0%}",
                    },
                ),
                hide_index=True,
                width="stretch",
            )

    st.markdown("#### Direction calls")
    st.caption("Up/down calls, scored after each horizon's close. Compare the hit rate with always guessing 'up'.")
    hist = load_track_record()
    track_h = (
        st.segmented_control(
            "Horizon", CFG.train.horizons, default=CFG.train.horizons[0], format_func=horizon_label, key="track_h"
        )
        or CFG.train.horizons[0]
    )
    if hist.empty:
        st.info("No predictions recorded yet. Run `smp predict` (or `smp run-all`).")
    else:
        summary = summarize(hist)
        summary = summary[summary["horizon"] == track_h].set_index("ticker")
        summary = summary.assign(first=summary["first"].dt.date, last=summary["last"].dt.date)
        summary = summary[["predictions", "scored", "hit_rate", "always_up_hit_rate", "brier", "first", "last"]]
        summary.columns = [
            "Predictions",
            "Scored",
            "Hit rate",
            "Always 'up' hit rate",
            "Brier score",
            "First",
            "Latest",
        ]
        summary.index.name = "Stock"
        st.dataframe(
            as_text(summary, {"Hit rate": "{:.0%}", "Always 'up' hit rate": "{:.0%}", "Brier score": "{:.3f}"}),
            width="stretch",
        )
        th = hist[(hist["ticker"] == ticker) & (hist["horizon"] == track_h)].sort_values("as_of")
        scored = th.dropna(subset=["actual_return"])
        if len(scored) < 2:
            st.info(
                f"{ticker}: {len(th)} prediction{'s' if len(th) != 1 else ''} recorded, {len(scored)} scored so far. "
                "The chart appears once a few days have been scored. Run the pipeline each trading day."
            )
        else:
            n = np.arange(1, len(scored) + 1)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=scored["as_of"], y=scored["correct"].cumsum() / n, name="Model", line=dict(color=C["s1"], width=2)
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=scored["as_of"],
                    y=scored["actual_direction"].cumsum() / n,
                    name="Always 'up'",
                    line=dict(color=C["s2"], width=2),
                )
            )
            fig.add_hline(y=0.5, line=dict(color=C["muted"], width=1, dash="dot"))
            fig.update_layout(title=f"{ticker} cumulative hit rate")
            fig.update_yaxes(tickformat=".0%")
            show(style(fig, height=360))
        if not th.empty:
            recent = th.sort_values("as_of", ascending=False).assign(
                as_of=lambda d: d["as_of"].dt.date,
                target_date=lambda d: pd.to_datetime(d["target_date"]).dt.date,
                result=lambda d: d["correct"].map({1.0: "✓ right", 0.0: "✗ wrong"}).fillna("pending"),
            )
            recent = recent[["as_of", "direction", "p_up", "target_date", "actual_return", "result"]]
            recent.columns = ["Made on", "Call", "Chance of up", "Scored on", "What happened", "Result"]
            st.markdown(f"**{ticker}: every {track_h}-day prediction**")
            st.dataframe(
                as_text(recent, {"Chance of up": "{:.0%}", "What happened": "{:+.2%}"}),
                hide_index=True,
                width="stretch",
            )

# --- price & indicators ------------------------------------------------------------------------

with tab_price:
    ranges = {"3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "All": None}
    choice = st.segmented_control("Range", list(ranges), default="1Y", key="price_range") or "1Y"

    c = prices["Close"]
    ind = pd.DataFrame(index=prices.index)
    ind["sma20"], ind["sma50"] = c.rolling(20).mean(), c.rolling(50).mean()
    ind = ind.join(bollinger(c)).join(macd(c))
    ind["rsi"] = rsi(c)
    n = ranges[choice]
    px, ix = (prices.iloc[-n:], ind.iloc[-n:]) if n else (prices, ind)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.5, 0.14, 0.17, 0.19],
        subplot_titles=(f"{ticker} price", "Volume", "RSI (14): above 70 stretched, below 30 washed out", "MACD"),
    )
    fig.add_trace(go.Scatter(x=ix.index, y=ix["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(
        go.Scatter(
            x=ix.index,
            y=ix["lower"],
            line=dict(width=0),
            fill="tonexty",
            fillcolor=C["band"],
            name="Bollinger band",
            hoverinfo="skip",
        ),
        1,
        1,
    )
    fig.add_trace(
        go.Candlestick(
            x=px.index,
            open=px["Open"],
            high=px["High"],
            low=px["Low"],
            close=px["Close"],
            name=ticker,
            increasing=dict(line=dict(color=C["up"], width=1), fillcolor=C["up"]),
            decreasing=dict(line=dict(color=C["down"], width=1), fillcolor=C["down"]),
        ),
        1,
        1,
    )
    fig.add_trace(go.Scatter(x=ix.index, y=ix["sma20"], name="SMA 20", line=dict(color=C["s2"], width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=ix.index, y=ix["sma50"], name="SMA 50", line=dict(color=C["s3"], width=2)), 1, 1)
    vol_colors = np.where(px["Close"] >= px["Open"], C["up"], C["down"])
    fig.add_trace(go.Bar(x=px.index, y=px["Volume"], marker_color=vol_colors, name="Volume", showlegend=False), 2, 1)
    fig.add_trace(
        go.Scatter(x=ix.index, y=ix["rsi"], name="RSI", line=dict(color=C["s1"], width=2), showlegend=False), 3, 1
    )
    for level in (30, 70):
        fig.add_hline(y=level, line=dict(color=C["muted"], width=1, dash="dot"), row=3, col=1)
    fig.add_trace(
        go.Bar(
            x=ix.index,
            y=ix["hist"],
            marker_color=np.where(ix["hist"] >= 0, C["up"], C["down"]),
            name="MACD histogram",
            showlegend=False,
        ),
        4,
        1,
    )
    fig.add_trace(go.Scatter(x=ix.index, y=ix["macd"], name="MACD", line=dict(color=C["s1"], width=2)), 4, 1)
    fig.add_trace(go.Scatter(x=ix.index, y=ix["signal"], name="Signal", line=dict(color=C["s2"], width=2)), 4, 1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    show(style(fig, height=820))
    with st.expander("Data table"):
        st.dataframe(px.join(ix).sort_index(ascending=False).round(3), width="stretch")

# --- model performance -------------------------------------------------------------------------

with tab_model:
    st.caption(
        "Up/down and return models. In testing these are close to a coin flip, so treat them as an experiment, "
        "not a signal."
    )
    cc = st.columns([1, 1, 2])
    horizon = (
        cc[0].segmented_control(
            "Horizon", CFG.train.horizons, default=CFG.train.horizons[0], format_func=horizon_label, key="dir_h"
        )
        or CFG.train.horizons[0]
    )
    task = (
        cc[1].segmented_control(
            "Model",
            ["classifier", "regressor"],
            default="classifier",
            key="dir_task",
            format_func=lambda t: "Up / down" if t == "classifier" else "Return size",
        )
        or "classifier"
    )
    H_LABEL = "next-day" if horizon == 1 else f"{horizon}-day"
    metrics = load_metrics(ticker, horizon)
    if metrics is None:
        st.info("No trained model for this ticker. Run `smp train`.")
    else:
        m = metrics[task]
        hold = load_artifact(ticker, horizon, "holdout_predictions.parquet")
        kind = "direction classifier" if task == "classifier" else "return regressor"
        st.subheader(f"{ticker} · {H_LABEL} {kind}")
        if metrics.get("scope") == "pooled":
            scope_note = (
                f"Pooled model shared by all {metrics['n_training_tickers']} stocks, picked by CV across all of them. "
                f"Scores below are for {ticker} only."
            )
        else:
            scope_note = f"Model trained on {ticker} only."
        st.write(
            f"Selected model: **{m['selected']}** (by walk-forward CV {m['selection_metric']}). {scope_note} "
            f"Hold-out period {metrics['holdout_period'][0]} to {metrics['holdout_period'][1]}."
        )
        if m["beats_baseline_cv"]:
            st.success("✓ Beats the naive baseline in cross-validation")
        else:
            st.warning("⚠ Does not beat the naive baseline in cross-validation")

        key_metrics = (
            ["roc_auc", "accuracy", "log_loss"] if task == "classifier" else ["rmse", "r2", "directional_accuracy"]
        )
        order = [m["selected"]] + [n for n in m["cv"] if n != m["selected"]]
        rows = []
        for name in order:
            label = f"{name} (chosen)" if name == m["selected"] else name
            row = {"Model": label}
            for k in key_metrics:
                row[f"{METRIC_NAMES[k]} · CV"] = m["cv"][name][k]
                row[f"{METRIC_NAMES[k]} · hold-out"] = m["holdout"][name][k]
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.3f}"), width="stretch")
        st.caption(
            "CV = average over walk-forward folds (2010 to last year); hold-out = the final 12 months, never used "
            "for training or model choice. "
            + (
                "ROC-AUC 0.50 means no better than chance."
                if task == "classifier"
                else "R² below 0 means worse than always predicting zero."
            )
        )

        left, right = st.columns(2)
        if task == "classifier":
            y, p = hold["actual_direction"].astype(int), hold["p_up"]
            fpr, tpr, _ = roc_curve(y, p)
            roc = go.Figure()
            roc.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color=C["muted"], width=1, dash="dot"))
            )
            roc.add_trace(go.Scatter(x=fpr, y=tpr, name=m["selected"], line=dict(color=C["s1"], width=2)))
            roc.add_annotation(x=0.62, y=0.45, text="chance", showarrow=False, font=dict(color=C["muted"]))
            roc.update_layout(
                showlegend=False,
                title=f"ROC curve, hold-out (area {m['holdout'][m['selected']]['roc_auc']:.2f})",
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                hovermode="closest",
            )
            left.plotly_chart(style(roc), width="stretch", theme="streamlit")

            cm = confusion_matrix(y, (p > CFG.backtest.prob_threshold).astype(int), labels=[0, 1])
            heat = go.Figure(
                go.Heatmap(
                    z=cm,
                    x=["Pred down", "Pred up"],
                    y=["Actual down", "Actual up"],
                    colorscale=[[i / 3, col] for i, col in enumerate(C["seq"])],
                    showscale=False,
                    text=cm,
                    texttemplate="%{text}",
                    hovertemplate="%{y} · %{x}: %{z}<extra></extra>",
                )
            )
            heat.update_layout(title="Confusion matrix (hold-out)", hovermode="closest")
            heat.update_yaxes(autorange="reversed")
            right.plotly_chart(style(heat), width="stretch", theme="streamlit")
        else:
            sc = go.Figure()
            sc.add_trace(
                go.Scatter(
                    x=hold["pred_return"],
                    y=hold["actual_return"],
                    mode="markers",
                    name="Days",
                    marker=dict(size=8, color=C["s1"], opacity=0.6, line=dict(width=1, color="rgba(255,255,255,0.8)")),
                    hovertemplate="predicted %{x:.3%}<br>actual %{y:.3%}<extra></extra>",
                )
            )
            sc.update_layout(
                title=f"Predicted vs actual {H_LABEL} return (hold-out)",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                hovermode="closest",
                showlegend=False,
            )
            sc.update_xaxes(tickformat=".1%")
            sc.update_yaxes(tickformat=".1%")
            left.plotly_chart(style(sc), width="stretch", theme="streamlit")

            dir_acc = m["holdout"][m["selected"]]["directional_accuracy"]
            right.metric("Directional accuracy (hold-out)", f"{dir_acc:.1%}")
            right.metric(
                "Hold-out RMSE",
                f"{m['holdout'][m['selected']]['rmse']:.4f}",
                f"baseline {m['holdout']['baseline']['rmse']:.4f}",
                delta_color="off",
            )

        imp = pd.Series(m["feature_importance"]).head(15).sort_values()
        imp.index = [feature_label(i) for i in imp.index]
        if not imp.empty:
            bar = go.Figure(
                go.Bar(
                    x=imp.values,
                    y=imp.index,
                    orientation="h",
                    marker_color=C["s1"],
                    hovertemplate="%{y}: %{x:.1%}<extra></extra>",
                )
            )
            bar.update_layout(
                title=f"What the {m['selected']} model relies on most (share of importance)", hovermode="closest"
            )
            bar.update_xaxes(tickformat=".1%")
            show(style(bar, height=480))

# --- backtest ----------------------------------------------------------------------------------

with tab_bt:
    st.divider()
    bt = load_artifact(ticker, horizon, f"backtest_{task}.parquet")
    if bt is None or metrics is None:
        st.info("No backtest yet. Run `smp train`.")
    else:
        s = metrics[task]["backtest"]
        rule = (
            f"long when P(up) > {CFG.backtest.prob_threshold:.0%}"
            if task == "classifier"
            else "long when predicted return > 0"
        )
        st.subheader(f"Hold-out backtest: {rule}, else cash")
        held = "Position reviewed every day" if horizon == 1 else f"Position held {horizon} days, then reviewed"
        st.caption(f"{held}. Costs: {CFG.backtest.cost:.2%} per position change. Trades at the close.")
        comp = pd.DataFrame(
            {
                "Total return": [s["strategy_total_return"], s["buy_hold_total_return"]],
                "Sharpe": [s["strategy_sharpe"], s["buy_hold_sharpe"]],
                "Max drawdown": [s["strategy_max_drawdown"], s["buy_hold_max_drawdown"]],
                "Time invested": [s["exposure"], 1.0],
                "Trades": [s["n_trades"], 1],
            },
            index=["Model strategy", "Buy & hold"],
        )
        st.dataframe(
            comp.style.format(
                {
                    "Total return": "{:+.1%}",
                    "Sharpe": "{:.2f}",
                    "Max drawdown": "{:.1%}",
                    "Time invested": "{:.0%}",
                    "Trades": "{:.0f}",
                }
            ),
            width="stretch",
        )

        eq = go.Figure()
        eq.add_trace(
            go.Scatter(x=bt.index, y=bt["strategy_equity"], name="Strategy", line=dict(color=C["s1"], width=2))
        )
        eq.add_trace(
            go.Scatter(x=bt.index, y=bt["buy_hold_equity"], name="Buy & hold", line=dict(color=C["s2"], width=2))
        )
        eq.update_layout(title="Growth of $1")
        eq.update_yaxes(tickprefix="$")
        show(style(eq))

        dd = go.Figure()
        for col, name, color in (("strategy_equity", "Strategy", C["s1"]), ("buy_hold_equity", "Buy & hold", C["s2"])):
            dd.add_trace(
                go.Scatter(x=bt.index, y=bt[col] / bt[col].cummax() - 1, name=name, line=dict(color=color, width=2))
            )
        dd.update_layout(title="Drawdown")
        dd.update_yaxes(tickformat=".0%")
        show(style(dd, height=320))
        with st.expander("Data table"):
            st.dataframe(bt.sort_index(ascending=False), width="stretch")

# --- data quality ------------------------------------------------------------------------------

with tab_quality:
    q = load_quality()
    if not q:
        st.info("No quality report yet. Run `smp clean`.")
    else:
        summary = pd.DataFrame(
            {
                t: {
                    "rows in": r["rows_in"],
                    "rows out": r["rows_out"],
                    "start": r["start"],
                    "end": r["end"],
                    "duplicates removed": r["duplicates_removed"],
                    "values filled": r["missing_values_filled"],
                    "rows dropped": r["rows_dropped"],
                    "OHLC fixed": r["ohlc_fixed"],
                    "outlier days": len(r["outlier_dates"]),
                    "large gaps": len(r["large_gaps"]),
                }
                for t, r in q.items()
            }
        ).T
        st.subheader("Cleaning report")
        k = st.columns(4)
        k[0].metric("Stocks and indices", len(q))
        k[1].metric("Daily rows checked", f"{int(summary['rows in'].sum()):,}")
        k[2].metric("Rows dropped", int(summary["rows dropped"].sum()))
        k[3].metric("Extreme days flagged", int(summary["outlier days"].sum()), help="Kept in the data, not removed")
        st.caption(
            "Every run validates prices: duplicates removed, short gaps filled, impossible bars (high below low) "
            "fixed, and extreme one-day moves flagged but kept, since they are usually real events."
        )
        st.dataframe(summary, width="stretch")
        if ticker in q:
            r = q[ticker]
            st.write(f"**{ticker} flagged extreme-move days** (kept in the data):")
            st.write(", ".join(r["outlier_dates"]) or "none")
            if r["large_gaps"]:
                st.write(f"**{ticker} gaps longer than 5 days:**")
                st.dataframe(pd.DataFrame(r["large_gaps"]), hide_index=True)
