"""Risk & allocation tab: forecasts, calibration, the allocation backtest, past crashes, regime history."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smp.allocation import allocation_path
from smp.experiment import REPORTS_DIR
from smp.insights import calibration_bins, rolling_vol
from smp.regime import market_dir
from smp.risk import risk_dir, risk_predictions_path
from smp_dashboard.charts import fade, show, style
from smp_dashboard.data import (
    coverage,
    load_json,
    load_oos,
    load_parquet,
    load_prices,
)
from smp_dashboard.labels import (
    SHOWN_STRATEGIES,
    STRATEGY_LABELS,
    horizon_label,
    model_label,
)
from smp_dashboard.theme import REGIME_STYLE, RING, palette, roles
from smp_dashboard.today import regime_banner


def pooled_coverage(tickers: list[str], horizon: int) -> dict:
    """How often past moves stayed inside the forecast price ranges, over all the tickers."""
    cs = [coverage(t, horizon) for t in tickers]
    n = sum(c["n"] for c in cs)
    return {k: sum(c[k] * c["n"] for c in cs if c["n"]) / n if n else np.nan for k in ("typical", "wide")} | {"n": n}


def calibration_chart(series: list[tuple[str, pd.DataFrame, str]], title: str, fmt: str) -> go.Figure:
    """Grouped forecasts vs outcomes, with the diagonal where forecasts mean what they say."""
    C = palette()
    fig = go.Figure()
    top = max(float(max(b["predicted"].max(), b["actual"].max())) for _, b, _ in series) * 1.08
    fig.add_trace(
        go.Scatter(
            x=[0, top],
            y=[0, top],
            name="Perfect",
            line=dict(color=C["axis"], width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for name, b, color in series:
        fig.add_trace(
            go.Scatter(
                x=b["predicted"],
                y=b["actual"],
                name=name,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=9, color=color, line=dict(width=2, color=RING)),
                customdata=b["n"],
                hovertemplate=(
                    f"forecast %{{x:{fmt}}} → happened %{{y:{fmt}}}<br>%{{customdata:,}} forecasts<extra>{name}</extra>"
                ),
            )
        )
    fig.add_annotation(
        x=top * 0.97,
        y=top * 0.97,
        text="forecast = what happened",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        textangle=0,
        font=dict(color=C["muted"], size=11),
    )
    fig.update_layout(
        title=title, hovermode="closest", xaxis_title="Forecast (group average)", yaxis_title="What happened"
    )
    fig.update_xaxes(tickformat=fmt, range=[0, top], showspikes=False)
    fig.update_yaxes(tickformat=fmt, range=[0, top])
    return fig


def render(cfg, ticker: str, prices: pd.DataFrame) -> None:
    C = palette()  # looked up on every run: the theme can change between runs
    OURS, ALT, OURS2, ACTUAL = roles(C)
    rc = cfg.risk
    regime = load_json(str(market_dir(cfg) / "regime.json"))
    rpreds = load_parquet(str(risk_predictions_path(cfg)))
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
        rmeta = load_json(str(risk_dir(cfg, ticker, rh) / "metrics.json"))
        oos = load_oos(ticker, rh)
        if oos is not None and rmeta:
            if not st.toggle("Show full history (2013 onwards)", key="vol_full"):
                oos = oos.loc[oos.index[-1] - pd.DateOffset(years=2) :]
            vm = rmeta["vol"]
            fig = go.Figure()
            for col, name, color, width in (
                ("actual_logvol", "What happened", ACTUAL, 1.25),
                ("trailing_logvol", "Last month's vol", ALT, 1.75),
                ("pred_logvol", f"Forecast ({model_label(vm['selected'])})", OURS, 2),
            ):
                fig.add_trace(
                    go.Scatter(
                        x=oos.index,
                        y=np.exp(oos[col]),
                        name=name,
                        line=dict(color=color, width=width),
                        hovertemplate="%{y:.0%}",
                    )
                )
            fig.update_layout(title=f"{ticker}: {rh}-day volatility forecast vs what happened (out of sample)")
            fig.update_yaxes(tickformat=".0%")
            show(style(fig, height=380))
            sel = vm["selected"]
            st.caption(
                "Last month's volatility (orange) reacts late: it stays high after turbulence ends and low as it "
                "starts. The forecast (blue) adjusts sooner. Grey is what volatility turned out to be. "
                f"Walk-forward CV R² on log volatility: forecast {vm['cv'][sel]['r2']:.2f}, HAR "
                f"{vm['cv']['har']['r2']:.2f}, last month's vol {vm['cv']['persistence']['r2']:.2f}. "
                f"Average error {vm['cv'][sel]['mae_vol_points']:.1%} points."
            )

        # Can the numbers be trusted? Every out-of-sample forecast for every stock, grouped
        pooled = [o for t in cfg.data.tickers if (o := load_oos(t, rh)) is not None]
        if pooled:
            allo = pd.concat(pooled)
            st.subheader("Can you trust these numbers?")
            st.caption(
                f"Every out-of-sample {rh}-day forecast for the {len(pooled)} stocks since 2013 ({len(allo):,}), "
                "sorted into 8 equal groups from lowest to highest forecast. Points on the grey diagonal mean the "
                "forecasts meant what they said."
            )
            left, right = st.columns(2, gap="large")
            actual_vol = np.exp(allo["actual_logvol"])
            vol_fig = calibration_chart(
                [
                    ("Last month's vol", calibration_bins(np.exp(allo["trailing_logvol"]), actual_vol, 8), ALT),
                    ("Forecast", calibration_bins(np.exp(allo["pred_logvol"]), actual_vol, 8), OURS),
                ],
                "Volatility: forecast vs what happened",
                ".0%",
            )
            with left:
                show(style(vol_fig, height=400))
            drop_fig = calibration_chart(
                [("Forecast", calibration_bins(allo["p_drop"], allo["actual_drop"], 8), OURS)],
                f"Chance of a {rc.drop_threshold:.0%} drop: forecast vs how often",
                ".0%",
            )
            with right:
                show(style(drop_fig, height=400))
            cov = pooled_coverage(cfg.data.tickers, rh)
            st.caption(
                "Last month's volatility overshoots at the extremes: very high readings were followed by calmer "
                "periods and very low ones by livelier periods. The forecast tracks the diagonal much more closely, "
                f"though it runs slightly low. Price ranges (Price tab): {cov['typical']:.0%} of {rh}-day moves stayed "
                f"inside the typical range (68% if exact) and {cov['wide']:.0%} inside the wide one (95%). They're "
                "slightly too narrow, because big surprises happen more often than a bell curve allows."
            )

        # Allocation backtest
        st.subheader("Volatility-targeted allocation, 2013 to today (out of sample)")
        port_ok = (cfg.paths.models / "_portfolio" / "allocation" / "stats.json").exists()
        show_opts = ([f"All {len(cfg.data.tickers)} (equal weight)"] if port_ok else []) + [ticker]
        choice = st.segmented_control("Show", show_opts, default=show_opts[0], key="alloc_show") or show_opts[0]
        base = cfg.paths.models / "_portfolio" / "allocation" if choice != ticker else allocation_path(cfg, ticker)
        curves, stats = load_parquet(str(base / "curves.parquet")), load_json(str(base / "stats.json"))
        if curves is None or stats is None:
            st.info("No allocation backtest yet. Run `smp train`.")
        else:
            eq = go.Figure()
            dd = go.Figure()
            live = "model_vol_target" if rc.sizing_vol == "model" else "trailing_vol_target"
            colors = {"buy_hold": ALT, live: OURS}
            for s in SHOWN_STRATEGIES:
                e, color = curves[f"{s}_equity"], colors.get(s, OURS2)
                width = 2 if s in colors else 1.5
                eq.add_trace(
                    go.Scatter(
                        x=e.index,
                        y=e,
                        name=STRATEGY_LABELS[s],
                        line=dict(color=color, width=width),
                        hovertemplate="$%{y:,.2f}",
                    )
                )
                # buy & hold's drawdowns as a light wash, so the strategies' lines read on top of it
                wash = s == "buy_hold"
                dd.add_trace(
                    go.Scatter(
                        x=e.index,
                        y=e / e.cummax() - 1,
                        name=STRATEGY_LABELS[s],
                        line=dict(color=color, width=1.25 if wash else width),
                        fill="tozeroy" if wash else None,
                        fillgradient=fade(C["fill_top2"], deep_at_bottom=True) if wash else None,
                        hovertemplate="%{y:.1%}",
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

            # What vol targeting does: steadier volatility, by holding less when markets are rough
            if choice == ticker:
                weight = curves[f"{live}_weight"]
            else:
                ws = [
                    c[f"{live}_weight"]
                    for t in cfg.data.tickers
                    if (c := load_parquet(str(allocation_path(cfg, t) / "curves.parquet"))) is not None
                ]
                weight = pd.concat(ws, axis=1).mean(axis=1) if ws else None
            hold_col, vol_col = st.columns(2, gap="large")
            if weight is not None:
                hf = go.Figure(
                    go.Scatter(
                        x=weight.index,
                        y=weight,
                        name="Invested",
                        line=dict(color=OURS, width=1.5),
                        fill="tozeroy",
                        fillgradient=fade(C["fill_top"], start=0, stop=1),
                        hovertemplate="%{y:.0%} invested",
                    )
                )
                hf.update_layout(title=f"How much {STRATEGY_LABELS[live].lower()} held")
                hf.update_yaxes(tickformat=".0%", range=[0, 1.05])
                with hold_col:
                    show(style(hf, height=280))
            rv = go.Figure()
            for s_, color in (("buy_hold", ALT), (live, OURS)):
                eqs = curves[f"{s_}_equity"]
                rv.add_trace(
                    go.Scatter(
                        x=eqs.index,
                        y=rolling_vol(eqs.pct_change()),
                        name=STRATEGY_LABELS[s_],
                        line=dict(color=color, width=1.75),
                        hovertemplate="%{y:.0%}",
                    )
                )
            rv.add_hline(
                y=rc.target_vol,
                line=dict(color=C["axis"], width=1),
                annotation=dict(text=f"{rc.target_vol:.0%} target", font=dict(color=C["muted"], size=11)),
                annotation_position="top left",
            )
            rv.update_layout(title="Volatility over the past year")
            rv.update_yaxes(tickformat=".0%", rangemode="tozero")
            with vol_col:
                show(style(rv, height=280))
            st.caption(
                "Left: the strategy cuts its holding when volatility rises (2020, 2022) and goes back to fully "
                "invested when markets calm. Right: that keeps its volatility steadier than buy & hold's."
                + (
                    " Each stock is sized to the target on its own; together they run below it, because the stocks "
                    "don't all move together."
                    if choice != ticker
                    else ""
                )
            )
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
                f"{cfg.backtest.cost:.1%} cost per unit traded, no leverage. Vol targeting holds cash part of the "
                "time, so it gives up return in strong bull markets in exchange for much smaller drawdowns."
            )

        # Through past crashes: the long-history study (S&P 500 back to 1993)
        lh = load_json(str(REPORTS_DIR / "long_history.json"))
        if lh and lh.get("crises"):
            live_lh = lh["control_strategy"]["name"]
            crises = lh["crises"][::-1]  # oldest at the top

            def crash_label(c: dict) -> str:
                years = c["start"][:4] + ("–" + c["end"][2:4] if c["end"][:4] != c["start"][:4] else "")
                return c["crisis"] if c["start"][:4] in c["crisis"] else f"{c['crisis']} ({years})"

            labels = [crash_label(c) for c in crises]
            fig = go.Figure()
            seg_x, seg_y = [], []
            for c, lab in zip(crises, labels, strict=True):
                seg_x += [c["buy_hold"]["max_drawdown"], c[live_lh]["max_drawdown"], None]
                seg_y += [lab, lab, None]
            fig.add_trace(
                go.Scatter(
                    x=seg_x,
                    y=seg_y,
                    mode="lines",
                    line=dict(color=C["axis"], width=2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            for s_, color in (("buy_hold", ALT), (live_lh, OURS)):
                fig.add_trace(
                    go.Scatter(
                        x=[c[s_]["max_drawdown"] for c in crises],
                        y=labels,
                        mode="markers",
                        name=STRATEGY_LABELS.get(s_, s_),
                        marker=dict(size=11, color=color, line=dict(width=2, color=RING)),
                        customdata=[c[live_lh]["avg_weight"] for c in crises],
                        hovertemplate="%{x:.0%} worst drawdown<extra>%{fullData.name}</extra>",
                    )
                )
            fig.update_layout(title="Worst drawdown in each crash: buy & hold vs vol targeting", hovermode="y unified")
            fig.update_xaxes(tickformat=".0%", showspikes=False)
            fig.update_yaxes(showgrid=False)
            st.markdown("#### Through past crashes")
            style(fig, height=380)
            fig.update_layout(title_xref="container")  # start at the chart's edge, not after the labels
            show(fig)
            whole = lh["allocation"]["all"]
            st.caption(
                f"{lh['ticker']} only, rebuilt from {lh['start'][:4]} with out-of-sample forecasts from "
                f"{whole[live_lh]['start'][:4]} (see the long_history report). Vol targeting helped most in sudden, "
                "volatile crashes (2008, COVID) and least in the slow 2000–02 decline, when volatility stayed "
                "moderate while prices fell. Worst drawdown over the whole period: "
                f"{whole[live_lh]['max_drawdown']:.0%} vs {whole['buy_hold']['max_drawdown']:.0%} for buy & hold."
            )

        # Regime history
        rhist = load_parquet(str(market_dir(cfg) / "regime_history.parquet"))
        rval = load_json(str(market_dir(cfg) / "regime_validation.json"))
        mkt_prices = load_prices(cfg.risk.market_ticker)
        if rhist is not None and rval and mkt_prices is not None:
            st.subheader("Regime history (out of sample)")
            h = rhist.dropna(subset=["regime"])
            px_m = mkt_prices["Close"].reindex(h.index)
            fig = go.Figure()
            # Calm and stressed spells as light washes behind the price; normal days stay unshaded
            washes = {"calm": C["calm_wash"], "stressed": C["stressed_wash"]}
            spell = (h["regime"] != h["regime"].shift()).cumsum()
            shapes = []  # set all at once: add_vrect per spell takes ~12s for 250+ spells
            for _, g in h.groupby(spell):
                if (r := g["regime"].iloc[0]) in washes:
                    end = h.index[min(h.index.get_loc(g.index[-1]) + 1, len(h) - 1)]
                    shapes.append(
                        dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0=g.index[0],
                            x1=end,
                            y0=0,
                            y1=1,
                            fillcolor=washes[r],
                            line_width=0,
                            layer="below",
                        )
                    )
            fig.update_layout(shapes=shapes)
            for r in ("calm", "normal", "stressed"):  # legend keys for the washes (icon + label)
                _, icon, label = REGIME_STYLE[r]
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        name=f"{icon} {label}" + (" (unshaded)" if r == "normal" else ""),
                        marker=dict(
                            symbol="square",
                            size=12,
                            color=washes.get(r, "rgba(0,0,0,0)"),
                            line=dict(width=1, color=C["axis"]),
                        ),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=px_m.index,
                    y=px_m,
                    name=cfg.risk.market_ticker,
                    line=dict(color=OURS, width=1.75),
                    customdata=h["regime"].map(lambda r: f"{REGIME_STYLE[r][1]} {REGIME_STYLE[r][2]}"),
                    hovertemplate="$%{y:,.0f} · %{customdata}<extra></extra>",
                    showlegend=False,
                )
            )
            fig.update_layout(title=f"{cfg.risk.market_ticker} and each day's market regime")
            fig.update_xaxes(type="date")  # the empty legend-key traces would otherwise make it numeric
            fig.update_yaxes(tickprefix="$")
            show(style(fig, height=400))
            vt = pd.DataFrame(rval).T.reindex(list(REGIME_STYLE))
            vt.index = [f"{REGIME_STYLE[r][1]} {REGIME_STYLE[r][2]}" for r in vt.index]
            vt.columns = ["Share of days", "Next 20d volatility", "Next 20d return", "Chance of 5% drop in next 20d"]
            st.dataframe(vt.style.format("{:.1%}"), width="stretch")
