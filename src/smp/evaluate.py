"""Metrics and a simple long/cash backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)

TRADING_DAYS = 252


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    pred = (proba > threshold).astype(int)
    two_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if two_classes else float("nan"),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "up_rate": float(y_true.mean()),
    }


def regression_metrics(y_true, pred) -> dict[str, float]:
    y_true, pred = np.asarray(y_true, dtype=float), np.asarray(pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)),
        "directional_accuracy": float(((pred > 0) == (y_true > 0)).mean()),
    }


def backtest(log_returns: pd.Series, signal: pd.Series, cost: float = 0.001) -> pd.DataFrame:
    """Long (1) or cash (0) each day.

    signal[t] is decided at day t's close and earns log_returns[t] (the t -> t+1 move).
    A cost is charged on every change of position.
    """
    signal = signal.reindex(log_returns.index).fillna(0).clip(0, 1)
    asset = np.expm1(log_returns)
    trades = signal.diff().abs().fillna(signal.iloc[0])
    strat = signal * asset - trades * cost
    return pd.DataFrame({
        "position": signal,
        "strategy_return": strat,
        "buy_hold_return": asset,
        "strategy_equity": (1 + strat).cumprod(),
        "buy_hold_equity": (1 + asset).cumprod(),
    })


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def _sharpe(returns: pd.Series) -> float:
    std = returns.std()
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0


def backtest_stats(bt: pd.DataFrame) -> dict[str, float]:
    n = len(bt)
    years = n / TRADING_DAYS if n else 1.0
    in_market = bt["position"] > 0
    stats = {}
    for name in ("strategy", "buy_hold"):
        eq, ret = bt[f"{name}_equity"], bt[f"{name}_return"]
        total = float(eq.iloc[-1] - 1) if n else 0.0
        stats[f"{name}_total_return"] = total
        stats[f"{name}_cagr"] = float((1 + total) ** (1 / years) - 1) if n else 0.0
        stats[f"{name}_sharpe"] = _sharpe(ret)
        stats[f"{name}_max_drawdown"] = _max_drawdown(eq)
    stats["exposure"] = float(in_market.mean())
    stats["n_trades"] = int(bt["position"].diff().abs().fillna(bt["position"].iloc[0]).sum())
    stats["win_rate"] = float((bt.loc[in_market, "buy_hold_return"] > 0).mean()) if in_market.any() else 0.0
    return stats
