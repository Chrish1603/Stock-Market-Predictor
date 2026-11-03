"""Uncertainty for model comparisons.

Scores from the same time period are correlated (AAPL and MSFT share a market), so differences are
resampled by time period (cluster bootstrap) rather than treating every score as independent.
Daily-return comparisons use a block bootstrap, which keeps runs of consecutive days together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BOOT = 5000
ALPHA = 0.05


def _centred_p(boot: np.ndarray, estimate: float) -> float:
    """Two-sided p-value for 'true difference = 0': how often the bootstrap, shifted to centre on 0,
    is at least as far from 0 as the observed estimate."""
    return float((np.abs(boot - estimate) >= abs(estimate)).mean())


def cluster_bootstrap_mean(diffs: pd.Series, clusters: pd.Series, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Mean of paired differences with a 95% interval, resampling whole clusters (time periods)."""
    df = pd.DataFrame({"d": np.asarray(diffs, float), "c": np.asarray(clusters)}).dropna()
    if df.empty:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n": 0, "n_clusters": 0}
    sums = df.groupby("c")["d"].agg(["sum", "count"])
    s, n = sums["sum"].to_numpy(), sums["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boot = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    mean = float(df["d"].mean())
    p = _centred_p(boot, mean)
    return {
        "mean": mean,
        "ci_low": float(np.quantile(boot, ALPHA / 2)),
        "ci_high": float(np.quantile(boot, 1 - ALPHA / 2)),
        "p_value": float(min(1.0, max(p, 1 / n_boot))),
        "n": len(df),
        "n_clusters": len(s),
    }


def block_bootstrap_sharpe_diff(
    ret_a: pd.Series, ret_b: pd.Series, block: int = 20, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Sharpe(a) - Sharpe(b) on the same days, with a 95% interval from a moving-block bootstrap."""
    both = pd.concat([ret_a, ret_b], axis=1, keys=["a", "b"]).dropna().to_numpy()

    def sharpe(x):
        sd = x.std(axis=0)
        return np.where(sd > 0, x.mean(axis=0) / sd * np.sqrt(252), 0.0)

    point = sharpe(both)
    n = len(both)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block, size=(n_boot, int(np.ceil(n / block))))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        rows = (starts[i][:, None] + np.arange(block)).ravel()[:n]
        s = sharpe(both[rows])
        diffs[i] = s[0] - s[1]
    d = float(point[0] - point[1])
    p = _centred_p(diffs, d)
    return {
        "mean": d,
        "ci_low": float(np.quantile(diffs, ALPHA / 2)),
        "ci_high": float(np.quantile(diffs, 1 - ALPHA / 2)),
        "p_value": float(min(1.0, max(p, 1 / n_boot))),
        "n": int(n),
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values: controls the chance of any false 'winner' across several comparisons."""
    items = sorted((p, k) for k, p in p_values.items() if not np.isnan(p))
    m, out, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    for k in p_values:
        out.setdefault(k, np.nan)
    return out


def verdict(mean: float, p_adj: float, higher_is_better: bool) -> str:
    if np.isnan(p_adj) or p_adj >= ALPHA:
        return "no clear difference"
    better = mean > 0 if higher_is_better else mean < 0
    return "better" if better else "worse"
