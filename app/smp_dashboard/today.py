"""Today tab: market regime and a card per stock."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smp.features import features_path
from smp.regime import market_dir
from smp.risk import risk_dir, risk_predictions_path
from smp_dashboard.charts import fade
from smp_dashboard.data import (
    load_json,
    load_metrics,
    load_parquet,
    load_predictions,
    load_prices,
)
from smp_dashboard.labels import (
    ordinal,
    risk_words,
)
from smp_dashboard.theme import REGIME_STYLE, palette


def sparkline(series: pd.Series) -> go.Figure:
    C = palette()
    lo, hi = float(series.min()), float(series.max())
    pad = (hi - lo) * 0.08 or 1.0
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            line=dict(color=C["s1"], width=2, shape="spline", smoothing=0.4),
            fill="tozeroy",
            fillgradient=fade(C["fill_top"], start=lo - pad, stop=hi),
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


def render(cfg, ticker: str, prices: pd.DataFrame) -> None:
    rc = cfg.risk
    rpreds = load_parquet(str(risk_predictions_path(cfg)))
    regime = load_json(str(market_dir(cfg) / "regime.json"))
    dpreds = load_predictions()
    if rpreds is None or rpreds.empty:
        st.info("No forecasts yet. Run `smp run-all`.")
    else:
        if regime:
            regime_banner(regime)
        h_alloc, h_dir = rc.allocation_horizon, max(cfg.train.horizons)
        grid: list = []
        for i, t in enumerate(cfg.data.tickers):
            if i % 2 == 0:  # a new row of two cards, so cards line up across the row
                grid = st.columns(2, gap="medium")
            r = rpreds[(rpreds["ticker"] == t) & (rpreds["horizon"] == h_alloc)]
            px_t = load_prices(t)
            if r.empty or px_t is None:
                continue
            r = r.iloc[0]
            close = px_t["Close"]
            feats = load_parquet(str(features_path(cfg, t)))
            hist_vol = (feats["vol_20"] * np.sqrt(252)).dropna() if feats is not None else pd.Series(dtype=float)
            pct = float((hist_vol < r.forecast_vol).mean()) if len(hist_vol) else float("nan")
            rmeta = load_json(str(risk_dir(cfg, t, h_alloc) / "metrics.json"))
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
            arrow, tone = ("▲", "tone-up") if change >= 0 else ("▼", "tone-down")
            direction = ""
            if not d.empty and dmeta:
                dd_ = d.iloc[0]
                auc = dmeta["classifier"]["cv"][dmeta["classifier"]["selected"]]["roc_auc"]
                direction = (
                    f"Direction model ({h_dir} days): {'up' if dd_.direction == 'up' else 'down'} {dd_.p_up:.0%} · "
                    f"weak signal, score {auc:.2f} where 0.50 is a coin flip"
                )

            with grid[i % 2].container(border=True, height="stretch", key=f"card-{t}"):
                st.markdown(
                    f"<div class='card-head'><div><span class='ticker'>{t}</span>"
                    f"<span class='price'>{close.iloc[-1]:,.2f}</span></div>"
                    f"<div class='delta'><span class='{tone}'>{arrow} {abs(change):.1%}</span> "
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
