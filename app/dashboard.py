"""Streamlit dashboard: predictions, price charts, model performance, backtest, data quality.

Run with:  streamlit run app/dashboard.py
Reads pipeline outputs only; training happens in the pipeline (`smp run-all`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve

from smp import pipeline
from smp.allocation import allocation_path
from smp.clean import clean_path, report_path
from smp.config import load_config
from smp.features import bollinger, macd, rsi
from smp.history import load_history, summarize
from smp.predict import predictions_path
from smp.regime import market_dir
from smp.risk import risk_dir, risk_predictions_path
from smp.train import model_dir

st.set_page_config(page_title="Stock Market Predictor", layout="wide")
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
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
# Status colours are reserved for regime state and always shown with an icon + label.
REGIME_STYLE = {
    "calm": ("#0ca30c", "●", "Calm"),
    "normal": (C["muted"], "◐", "Normal"),
    "stressed": ("#d03b3b", "▲", "Stressed"),
}


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        font_family=FONT,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"], gridwidth=1, zeroline=False)
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
ticker = st.sidebar.selectbox("Ticker", CFG.data.tickers)
horizon = st.sidebar.radio(
    "Horizon",
    CFG.train.horizons,
    format_func=lambda h: f"{h} trading day{'s' if h > 1 else ''} ahead",
    horizontal=True,
)
H_LABEL = "next-day" if horizon == 1 else f"{horizon}-day"
task = st.sidebar.radio(
    "Model",
    ["classifier", "regressor"],
    format_func=lambda t: "Direction (up/down)" if t == "classifier" else "Return (regression)",
)
if st.sidebar.button("Refresh data & predictions", help="Downloads new prices and re-predicts. Does not retrain."):
    with st.spinner("Updating data..."):
        pipeline.setup_logging(CFG.paths.logs)
        pipeline.refresh(CFG)
    st.cache_data.clear()
    st.sidebar.success("Updated")
st.sidebar.caption("Retrain with `smp run-all` (or `smp train`).")
st.sidebar.caption("Educational project, not financial advice.")

prices = load_prices(ticker)
metrics = load_metrics(ticker, horizon)
if prices is None:
    st.warning("No data yet. Run `smp run-all` first.")
    st.stop()

tab_overview, tab_risk, tab_track, tab_price, tab_model, tab_bt, tab_quality = st.tabs(
    [
        "Overview",
        "Risk & allocation",
        "Track record",
        "Price & indicators",
        "Model performance",
        "Backtest",
        "Data quality",
    ]
)

# --- overview ----------------------------------------------------------------------------------

with tab_overview:
    preds = load_predictions()
    if preds is not None and "horizon" in preds:
        preds = preds[preds["horizon"] == horizon]
    if preds is None or preds.empty:
        st.info("No predictions yet. Run `smp run-all`.")
    else:
        as_of = pd.to_datetime(preds["as_of"]).max().date()
        st.subheader(f"{H_LABEL.capitalize()} predictions (based on close of {as_of})")
        cols = st.columns(len(preds))
        for col, row in zip(cols, preds.itertuples(), strict=False):
            arrow = "▲ up" if row.direction == "up" else "▼ down"
            col.metric(
                row.ticker,
                f"{row.p_up:.1%} P(up)",
                f"{arrow} · {row.pred_return:+.2%} predicted",
                delta_color="normal" if row.direction == "up" else "inverse",
            )
        st.dataframe(
            preds.assign(as_of=pd.to_datetime(preds["as_of"]).dt.date).style.format(
                {"p_up": "{:.1%}", "pred_return": "{:+.3%}"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Stock moves are mostly noise, especially day to day. Hold-out ROC-AUC near 0.5 is normal; "
            "check the Model performance tab before trusting any signal."
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
        # Market regime
        if regime:
            color, icon, label = REGIME_STYLE[regime["regime"]]
            left, right = st.columns([1, 2])
            left.markdown(
                f"<div style='font-size:0.9rem;color:{C['muted']}'>Market regime ({regime['ticker']}, "
                f"{regime['as_of']})</div><div style='font-size:2.2rem;font-weight:600'>"
                f"<span style='color:{color}'>{icon}</span> {label}</div>",
                unsafe_allow_html=True,
            )
            breadth = f"{regime['breadth']:.0%}" if regime["breadth"] is not None else "n/a"
            right.markdown(
                f"- 20-day vol forecast **{regime['forecast_vol']:.0%}**, higher than "
                f"**{regime['vol_percentile']:.0%}** of past days\n"
                f"- Breadth: **{breadth}** of stocks above their 200-day average\n"
                f"- Trend: {regime['ticker']} is **{'above' if regime['trend_up'] else 'below'}** its 200-day average"
            )
            st.caption(
                "Stressed = vol forecast in its top 20%, or weak breadth in a downtrend. Calm = vol in its bottom 40% "
                "with healthy breadth in an uptrend. A risk signal, not a sell signal: returns after stressed days "
                "were on average higher (rebounds), but swings were much larger."
            )

        # Risk forecasts per ticker
        st.subheader("Risk forecasts")
        rh = st.radio(
            "Risk horizon",
            rc.horizons,
            index=len(rc.horizons) - 1,
            horizontal=True,
            format_func=lambda h: f"{h} trading days",
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
                f"Walk-forward CV R² on log volatility: forecast {vm['cv'][sel]['r2']:.2f}, HAR "
                f"{vm['cv']['har']['r2']:.2f}, last month's vol {vm['cv']['persistence']['r2']:.2f}. "
                f"Average error {vm['cv'][sel]['mae_vol_points']:.1%} points."
            )

        # Allocation backtest
        st.subheader("Volatility-targeted allocation, 2013 to today (out of sample)")
        port_ok = (CFG.paths.models / "_portfolio" / "allocation" / "stats.json").exists()
        choice = st.radio(
            "Show", ([f"All {len(CFG.data.tickers)} (equal weight)"] if port_ok else []) + [ticker], horizontal=True
        )
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
            eq.update_yaxes(type="log", tickprefix="$")
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
    hist = load_track_record()
    st.subheader("Live track record")
    st.caption(
        "Every prediction is saved when it's made and scored after the next close. Unlike the hold-out test, "
        "these were made before the outcome existed. Compare the hit rate with always guessing 'up'."
    )
    if hist.empty:
        st.info("No predictions recorded yet. Run `smp predict` (or `smp run-all`).")
    else:
        summary = summarize(hist)
        summary = summary[summary["horizon"] == horizon].set_index("ticker")
        st.dataframe(
            summary.style.format(
                {
                    "hit_rate": "{:.1%}",
                    "always_up_hit_rate": "{:.1%}",
                    "brier": "{:.4f}",
                    "avg_return_when_up_call": "{:+.3%}",
                    "avg_return_when_down_call": "{:+.3%}",
                },
                na_rep="–",
            ),
            width="stretch",
        )
        th = hist[(hist["ticker"] == ticker) & (hist["horizon"] == horizon)].sort_values("as_of")
        scored = th.dropna(subset=["actual_return"])
        if len(scored) < 2:
            st.info(
                f"{ticker}: {len(th)} prediction(s) recorded, {len(scored)} scored so far. "
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
            st.dataframe(
                recent[
                    ["as_of", "direction", "p_up", "pred_return", "target_date", "actual_return", "result"]
                ].style.format({"p_up": "{:.1%}", "pred_return": "{:+.3%}", "actual_return": "{:+.3%}"}, na_rep="–"),
                hide_index=True,
                width="stretch",
            )

# --- price & indicators ------------------------------------------------------------------------

with tab_price:
    ranges = {"3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "All": None}
    choice = st.radio("Range", list(ranges), index=2, horizontal=True)

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
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.17, 0.18],
        subplot_titles=("Price", "Volume", "RSI (14)", "MACD"),
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

        rows = []
        for name in m["cv"]:
            rows.append({"model": name, "split": "CV (mean)", **m["cv"][name]})
            rows.append({"model": name, "split": "Hold-out", **m["holdout"][name]})
        table = pd.DataFrame(rows).set_index(["model", "split"])
        st.dataframe(table.style.format("{:.4f}"), width="stretch")

        left, right = st.columns(2)
        if task == "classifier":
            y, p = hold["actual_direction"].astype(int), hold["p_up"]
            fpr, tpr, _ = roc_curve(y, p)
            roc = go.Figure()
            roc.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color=C["muted"], width=1, dash="dot"))
            )
            roc.add_trace(go.Scatter(x=fpr, y=tpr, name=m["selected"], line=dict(color=C["s1"], width=2)))
            roc.update_layout(
                title="ROC curve (hold-out)",
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
            bar.update_layout(title=f"Top features ({m['selected']})", hovermode="closest")
            bar.update_xaxes(tickformat=".0%")
            show(style(bar, height=480))

# --- backtest ----------------------------------------------------------------------------------

with tab_bt:
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
        k = st.columns(5)
        k[0].metric(
            "Strategy return",
            f"{s['strategy_total_return']:+.1%}",
            f"{s['strategy_total_return'] - s['buy_hold_total_return']:+.1%} vs buy & hold",
        )
        k[1].metric("Buy & hold return", f"{s['buy_hold_total_return']:+.1%}")
        k[2].metric("Sharpe (strategy / B&H)", f"{s['strategy_sharpe']:.2f} / {s['buy_hold_sharpe']:.2f}")
        k[3].metric(
            "Max drawdown (strategy / B&H)", f"{s['strategy_max_drawdown']:.1%} / {s['buy_hold_max_drawdown']:.1%}"
        )
        k[4].metric("Time in market · trades", f"{s['exposure']:.0%} · {s['n_trades']}")

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
        st.dataframe(summary, width="stretch")
        if ticker in q:
            r = q[ticker]
            st.write(f"**{ticker} flagged extreme-move days** (kept in the data):")
            st.write(", ".join(r["outlier_dates"]) or "none")
            if r["large_gaps"]:
                st.write(f"**{ticker} gaps longer than 5 days:**")
                st.dataframe(pd.DataFrame(r["large_gaps"]), hide_index=True)
