"""Direction models tab: up/down and return models, and their hold-out backtest."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve

from smp_dashboard.charts import fade, show, style
from smp_dashboard.data import (
    load_artifact,
    load_metrics,
)
from smp_dashboard.labels import (
    METRIC_NAMES,
    feature_label,
    horizon_label,
    model_label,
)
from smp_dashboard.theme import palette, roles


def render(cfg, ticker: str, prices: pd.DataFrame) -> None:
    C = palette()  # looked up on every run: the theme can change between runs
    OURS, ALT = roles(C)[:2]
    st.caption(
        "Up/down and return models. In testing these are close to a coin flip, so treat them as an experiment, "
        "not a signal."
    )
    cc = st.columns([1, 1, 2])
    horizon = (
        cc[0].segmented_control(
            "Horizon", cfg.train.horizons, default=cfg.train.horizons[0], format_func=horizon_label, key="dir_h"
        )
        or cfg.train.horizons[0]
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
        sel, metric = m["selected"], m["selection_metric"]
        metric_name = METRIC_NAMES.get(metric, metric)
        st.write(
            f"Selected model: **{model_label(sel)}**, chosen by walk-forward cross-validation on {metric_name}. "
            f"{scope_note} Hold-out period {metrics['holdout_period'][0]} to {metrics['holdout_period'][1]}."
        )
        score, base = m["cv"][sel][metric], m["cv"]["baseline"][metric]
        if m["beats_baseline_cv"]:
            st.markdown(
                f"✓ Better than the naive baseline in cross-validation ({metric_name} {score:.3f} vs {base:.3f})."
            )
        else:
            st.warning(
                f"⚠ No better than the naive baseline in cross-validation ({metric_name} {score:.3f} vs {base:.3f})."
            )

        key_metrics = (
            ["roc_auc", "accuracy", "log_loss"] if task == "classifier" else ["rmse", "r2", "directional_accuracy"]
        )
        order = [m["selected"]] + [n for n in m["cv"] if n != m["selected"]]
        rows = []
        for name in order:
            label = f"{model_label(name)} (chosen)" if name == m["selected"] else model_label(name)
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
            roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color=C["axis"], width=1)))
            roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    name=m["selected"],
                    line=dict(color=OURS, width=2),
                    fill="tonexty",
                    fillcolor=C["band"],
                    hovertemplate="false positives %{x:.0%}, true positives %{y:.0%}<extra></extra>",
                )
            )
            roc.add_annotation(x=0.62, y=0.45, text="chance", showarrow=False, font=dict(color=C["muted"]))
            roc.update_layout(
                showlegend=False,
                title=f"ROC curve, hold-out (area {m['holdout'][m['selected']]['roc_auc']:.2f})",
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                hovermode="closest",
            )
            left.plotly_chart(style(roc), width="stretch", theme="streamlit")

            cm = confusion_matrix(y, (p > cfg.backtest.prob_threshold).astype(int), labels=[0, 1])
            heat = go.Figure(
                go.Heatmap(
                    z=cm,
                    x=["Pred down", "Pred up"],
                    y=["Actual down", "Actual up"],
                    colorscale=[[0, C["seq"][0]], [1, C["seq"][2]]],  # light to mid: no saturated blocks
                    showscale=False,
                    text=[[f"{v}<br>{v / cm.sum():.0%} of days" for v in row] for row in cm],
                    texttemplate="%{text}",
                    textfont=dict(size=15),
                    xgap=2,
                    ygap=2,
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
                    width=0.55,
                    hovertemplate="%{y}: %{x:.1%}<extra></extra>",
                )
            )
            bar.update_layout(
                title=f"What the {model_label(m['selected'])} model relies on most (share of importance)",
                hovermode="closest",
            )
            bar.update_xaxes(tickformat=".1%")
            style(bar, height=480)
            bar.update_layout(title_xref="container")  # start at the chart's edge, not after the long labels
            show(bar)
    st.divider()
    bt = load_artifact(ticker, horizon, f"backtest_{task}.parquet")
    if bt is None or metrics is None:
        st.info("No backtest yet. Run `smp train`.")
    else:
        s = metrics[task]["backtest"]
        rule = (
            f"long when P(up) > {cfg.backtest.prob_threshold:.0%}"
            if task == "classifier"
            else "long when predicted return > 0"
        )
        st.subheader(f"Hold-out backtest: {rule}, else cash")
        held = "Position reviewed every day" if horizon == 1 else f"Position held {horizon} days, then reviewed"
        st.caption(f"{held}. Costs: {cfg.backtest.cost:.2%} per position change. Trades at the close.")
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
            go.Scatter(
                x=bt.index,
                y=bt["strategy_equity"],
                name="Model strategy",
                line=dict(color=OURS, width=2),
                hovertemplate="$%{y:,.2f}",
            )
        )
        eq.add_trace(
            go.Scatter(
                x=bt.index,
                y=bt["buy_hold_equity"],
                name="Buy & hold",
                line=dict(color=ALT, width=2),
                hovertemplate="$%{y:,.2f}",
            )
        )
        eq.update_layout(title="Growth of $1")
        eq.update_yaxes(tickprefix="$")
        show(style(eq))

        dd = go.Figure()
        for col, name, color in (("buy_hold_equity", "Buy & hold", ALT), ("strategy_equity", "Model strategy", OURS)):
            wash = col == "buy_hold_equity"
            dd.add_trace(
                go.Scatter(
                    x=bt.index,
                    y=bt[col] / bt[col].cummax() - 1,
                    name=name,
                    line=dict(color=color, width=1.25 if wash else 2),
                    fill="tozeroy" if wash else None,
                    fillgradient=fade(C["fill_top2"], deep_at_bottom=True) if wash else None,
                    hovertemplate="%{y:.1%}",
                )
            )
        dd.update_layout(title="Drawdown")
        dd.update_yaxes(tickformat=".0%")
        show(style(dd, height=320))
        with st.expander("Data table"):
            st.dataframe(bt.sort_index(ascending=False), width="stretch")
