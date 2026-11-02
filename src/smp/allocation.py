"""Volatility-targeted position sizing, backtested on out-of-sample risk forecasts.

Weight = target_vol / forecast_vol, capped at max_weight: hold more when risk is forecast to be
low and less when it is high. The backtest uses only forecasts made before each day: out-of-fold
predictions from the walk-forward folds, then the hold-out year.

Strategies compared per ticker:
- buy_hold: always 100% invested
- trailing_vol_target: sized with last month's volatility (the textbook version)
- model_vol_target: sized with the model's volatility forecast
- model_vol_target_drop_filter: as above, halved when P(big drop) is well above normal

Finding (Nov 2026, 2013-2026 out of sample): both vol-target versions cut max drawdown by about a
third with similar or better Sharpe, but sizing with the model's forecast did not beat sizing with
trailing volatility, and the drop filter hurt. Live suggestions therefore use trailing volatility
by default (risk.sizing_vol: trailing); the model forecasts drive the risk display.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from smp.config import Config
from smp.features import features_path, target_return
from smp.risk import risk_dir
from smp.train import model_dir

log = logging.getLogger(__name__)

TRADING_DAYS = 252
STRATEGIES = ["buy_hold", "trailing_vol_target", "model_vol_target", "model_vol_target_drop_filter"]
DROP_FILTER_MULTIPLE = 2.0  # "well above normal" = twice the historical drop rate
DROP_FILTER_SCALE = 0.5


def target_weights(annual_vol: pd.Series, target_vol: float, max_weight: float) -> pd.Series:
    return (target_vol / annual_vol).clip(0, max_weight)


def apply_band(weights: pd.Series, band: float) -> pd.Series:
    """Only trade when the target moves more than `band` from the current holding (cuts turnover)."""
    held, out = np.nan, np.empty(len(weights))
    for i, w in enumerate(weights.to_numpy()):
        if np.isnan(held) or abs(w - held) > band:
            held = w
        out[i] = held
    return pd.Series(out, index=weights.index)


def run_weights(daily_log_returns: pd.Series, weights: pd.Series, cost: float) -> pd.DataFrame:
    """weights[t] is set at day t's close and earns the t -> t+1 return; costs on weight changes."""
    w = weights.reindex(daily_log_returns.index).ffill().fillna(0.0)
    asset = np.expm1(daily_log_returns)
    trades = w.diff().abs().fillna(w.iloc[0])
    ret = w * asset - trades * cost
    return pd.DataFrame({"weight": w, "return": ret, "equity": (1 + ret).cumprod()})


def performance(run: pd.DataFrame) -> dict[str, float]:
    ret, eq = run["return"], run["equity"]
    years = len(ret) / TRADING_DAYS
    total = float(eq.iloc[-1] - 1)
    vol = float(ret.std() * np.sqrt(TRADING_DAYS))
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 else 0.0
    max_dd = float((eq / eq.cummax() - 1).min())
    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": float(ret.mean() / ret.std() * np.sqrt(TRADING_DAYS)) if ret.std() > 0 else 0.0,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else float("nan"),
        "avg_weight": float(run["weight"].mean()),
        "turnover_per_year": float(run["weight"].diff().abs().sum() / years) if years > 0 else 0.0,
    }


def out_of_sample_forecasts(cfg: Config, ticker: str, horizon: int) -> pd.DataFrame:
    """Out-of-fold forecasts followed by hold-out forecasts, on one continuous daily index."""
    d = risk_dir(cfg, ticker, horizon)
    oos = pd.concat(
        [pd.read_parquet(d / "oos_predictions.parquet"), pd.read_parquet(d / "holdout_predictions.parquet")]
    )
    return oos[~oos.index.duplicated(keep="last")].sort_index()


def strategy_weights(cfg: Config, fc: pd.DataFrame) -> dict[str, pd.Series]:
    rc, h = cfg.risk, cfg.risk.allocation_horizon
    model = target_weights(np.exp(fc["pred_logvol"]), rc.target_vol, rc.max_weight)
    trailing = target_weights(np.exp(fc["trailing_logvol"]), rc.target_vol, rc.max_weight)
    # historical drop rate known at t: labels need h more days to resolve, so shift by h
    drop_rate = fc["actual_drop"].expanding().mean().shift(h).bfill()
    filtered = model.where(fc["p_drop"] <= DROP_FILTER_MULTIPLE * drop_rate, model * DROP_FILTER_SCALE)
    return {
        "buy_hold": pd.Series(1.0, index=fc.index),
        "trailing_vol_target": apply_band(trailing, rc.rebalance_band),
        "model_vol_target": apply_band(model, rc.rebalance_band),
        "model_vol_target_drop_filter": apply_band(filtered, rc.rebalance_band),
    }


def backtest_ticker(cfg: Config, ticker: str) -> tuple[pd.DataFrame, dict]:
    h = cfg.risk.allocation_horizon
    fc = out_of_sample_forecasts(cfg, ticker, h)
    daily = pd.read_parquet(features_path(cfg, ticker))[target_return(1)]
    daily = daily.loc[fc.index[0] :].dropna()
    fc = fc.reindex(daily.index).ffill()  # carry forecasts over the short purge gap before the hold-out
    runs = {name: run_weights(daily, w, cfg.backtest.cost) for name, w in strategy_weights(cfg, fc).items()}
    curves = pd.concat({n: r for n, r in runs.items()}, axis=1)
    curves.columns = [f"{n}_{c}" for n, c in curves.columns]
    stats = {n: performance(r) for n, r in runs.items()}
    return curves, stats


def allocation_path(cfg: Config, ticker: str):
    return model_dir(cfg, ticker, cfg.risk.allocation_horizon).parent / "allocation"


def backtest_all(cfg: Config) -> dict:
    """Backtest every ticker and an equal-weight portfolio of all of them; save curves and stats."""
    if not cfg.risk.enabled:
        return {}
    results, returns = {}, {}
    for ticker in cfg.data.tickers:
        try:
            curves, stats = backtest_ticker(cfg, ticker)
        except FileNotFoundError:
            log.warning("%s: no risk forecasts, skipping allocation backtest", ticker)
            continue
        out = allocation_path(cfg, ticker)
        out.mkdir(parents=True, exist_ok=True)
        curves.to_parquet(out / "curves.parquet")
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
        results[ticker] = stats
        returns[ticker] = curves[[f"{s}_return" for s in STRATEGIES]]
        live = "model_vol_target" if cfg.risk.sizing_vol == "model" else "trailing_vol_target"
        b, m = stats["buy_hold"], stats[live]
        log.info(
            "%s allocation (%s) %s to %s: Sharpe %.2f vs buy-hold %.2f, max drawdown %.1f%% vs %.1f%%",
            ticker,
            live,
            curves.index[0].date(),
            curves.index[-1].date(),
            m["sharpe"],
            b["sharpe"],
            m["max_drawdown"] * 100,
            b["max_drawdown"] * 100,
        )

    if len(returns) > 1:
        common = pd.concat(returns, axis=1).dropna()
        port = {}
        for s in STRATEGIES:
            r = common.xs(f"{s}_return", axis=1, level=1).mean(axis=1)  # equal weight across tickers
            port[s] = pd.DataFrame({"weight": 1.0, "return": r, "equity": (1 + r).cumprod()})
        stats = {s: performance(p) for s, p in port.items()}
        for s in STRATEGIES:  # weights and turnover are averages of the per-ticker sleeves
            for k in ("avg_weight", "turnover_per_year"):
                stats[s][k] = float(np.mean([results[t][s][k] for t in returns]))
        out = cfg.paths.models / "_portfolio" / "allocation"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({f"{s}_equity": p["equity"] for s, p in port.items()}).to_parquet(out / "curves.parquet")
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
        results["_portfolio"] = stats
    return results


def suggested_weight(cfg: Config, trailing_annual_vol: float, model_annual_vol: float) -> float:
    """Today's suggested position size (before the rebalance band, which depends on what you hold)."""
    vol = model_annual_vol if cfg.risk.sizing_vol == "model" else trailing_annual_vol
    return float(np.clip(cfg.risk.target_vol / vol, 0, cfg.risk.max_weight))
