"""One shared model per horizon, trained on every ticker in the universe.

Rows from all tickers are stacked into a panel. Splits are made on dates, so all rows for a
given day fall on the same side of every train/test boundary. The model is then evaluated on
each predicted ticker separately and saved into that ticker's model folder, so prediction,
history and the dashboard work exactly as with per-ticker models.
"""

from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from smp.config import Config
from smp.evaluate import backtest, backtest_stats
from smp.features import feature_columns, features_path, target_direction, target_return
from smp.models import BASELINE, classifiers, feature_importance, regressors
from smp.train import SELECTION, _score, model_dir

log = logging.getLogger(__name__)

ID_COLS = ["Date", "ticker"]
MIN_ROWS_TO_SCORE = 50


def load_panel(cfg: Config) -> pd.DataFrame:
    """Stack every ticker's feature file into one long frame with Date and ticker columns."""
    frames = []
    for t in cfg.data.feature_tickers:
        p = features_path(cfg, t)
        if p.exists():
            f = pd.read_parquet(p)
            f.index.name = "Date"
            frames.append(f.reset_index().assign(ticker=t))
    if not frames:
        raise FileNotFoundError("no feature files; run `smp features` first")
    return pd.concat(frames, ignore_index=True).sort_values(ID_COLS).reset_index(drop=True)


def panel_feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(panel) if c not in ID_COLS]


def date_splits(dates: pd.Series, n_splits: int, gap: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward splits over unique dates, mapped back to row positions."""
    uniq = np.sort(dates.unique())
    pos = np.searchsorted(uniq, dates.to_numpy())
    splits = []
    for tr_d, te_d in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(uniq):
        tr = np.flatnonzero(pos <= tr_d.max())
        te = np.flatnonzero((pos >= te_d.min()) & (pos <= te_d.max()))
        splits.append((tr, te))
    return splits


def split_dev_holdout_by_date(panel: pd.DataFrame, holdout_days: int, horizon: int):
    """Last `holdout_days` dates are the hold-out; the `horizon` dates before it are dropped (their
    labels end inside the hold-out)."""
    uniq = np.sort(panel["Date"].unique())
    hold_start, dev_end = uniq[-holdout_days], uniq[-holdout_days - horizon - 1]
    return panel[panel["Date"] <= dev_end], panel[panel["Date"] >= hold_start]


def _predict(model, task: str, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if task == "classifier" else model.predict(X)


def _safe_score(task: str, y, pred) -> dict[str, float] | None:
    if len(y) < MIN_ROWS_TO_SCORE:
        return None
    return _score(task, y, pred)


def _mean_scores(scores: list[dict | None]) -> dict[str, float]:
    scores = [s for s in scores if s]
    return {k: float(np.nanmean([s[k] for s in scores])) for k in scores[0]} if scores else {}


def train_pooled(cfg: Config, horizon: int, panel: pd.DataFrame | None = None) -> dict:
    """Train both tasks for one horizon on the whole universe. Returns pooled + per-ticker metrics."""
    start = time.perf_counter()
    if panel is None:
        panel = load_panel(cfg)
    cols = panel_feature_columns(panel)
    ret_col, dir_col = target_return(horizon), target_direction(horizon)
    data = panel.dropna(subset=[*cols, ret_col, dir_col, target_return(1)]).reset_index(drop=True)
    tc = cfg.train
    if tc.tune:
        log.warning("tuning is not supported for pooled models; using default parameters")

    dev, hold = split_dev_holdout_by_date(data, tc.holdout_days, horizon)
    dev, hold = dev.reset_index(drop=True), hold.reset_index(drop=True)
    splits = date_splits(dev["Date"], tc.n_splits, max(tc.gap, horizon))
    targets = [t for t in cfg.data.tickers if t in set(data["ticker"])]
    log.info(
        "pooled %dd: %d rows from %d tickers (dev %d, hold-out %d)",
        horizon,
        len(data),
        data["ticker"].nunique(),
        len(dev),
        len(hold),
    )

    per_ticker: dict[str, dict] = {t: {} for t in targets}
    hold_frames = {}
    for t in targets:
        h = hold[hold["ticker"] == t].set_index("Date")
        hold_frames[t] = pd.DataFrame(
            {"actual_return": h[ret_col], "actual_direction": h[dir_col], "_daily": h[target_return(1)]}
        )
    pooled_summary: dict = {}

    for task, target in (("classifier", dir_col), ("regressor", ret_col)):
        candidates = (classifiers if task == "classifier" else regressors)(tc.random_state)
        cv_pooled, cv_ticker, ho_pooled, ho_ticker, ho_pred = {}, {}, {}, {}, {}
        for name, model in candidates.items():
            fold_pooled, fold_ticker = [], {t: [] for t in targets}
            for tr, te in splits:
                m = clone(model).fit(dev.loc[tr, cols], dev.loc[tr, target])
                pred = _predict(m, task, dev.loc[te, cols])
                test = dev.loc[te]
                fold_pooled.append(_score(task, test[target], pred))
                for t in targets:
                    mask = (test["ticker"] == t).to_numpy()
                    fold_ticker[t].append(_safe_score(task, test.loc[mask, target], pred[mask]))
            cv_pooled[name] = _mean_scores(fold_pooled)
            cv_ticker[name] = {t: _mean_scores(fold_ticker[t]) for t in targets}

            m = clone(model).fit(dev[cols], dev[target])
            pred = _predict(m, task, hold[cols])
            ho_pooled[name] = _score(task, hold[target], pred)
            ho_ticker[name], ho_pred[name] = {}, {}
            for t in targets:
                mask = (hold["ticker"] == t).to_numpy()
                ho_ticker[name][t] = _score(task, hold.loc[mask, target], pred[mask])
                ho_pred[name][t] = pred[mask]

        metric, higher = SELECTION[task]
        best = sorted((n for n in candidates if n != BASELINE), key=lambda n: cv_pooled[n][metric], reverse=higher)[0]
        log.info(
            "pooled %dd %s: selected %s (CV %s=%.4f vs baseline %.4f; hold-out %.4f)",
            horizon,
            task,
            best,
            metric,
            cv_pooled[best][metric],
            cv_pooled[BASELINE][metric],
            ho_pooled[best][metric],
        )

        final = clone(candidates[best]).fit(data[cols], data[target])
        pooled_dir = cfg.paths.models / "_pooled" / f"h{horizon}"
        pooled_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, pooled_dir / f"{task}.joblib")
        importance = feature_importance(final, cols)
        pooled_summary[task] = {"selected": best, "cv": cv_pooled, "holdout": ho_pooled}

        for t in targets:
            out = model_dir(cfg, t, horizon)
            out.mkdir(parents=True, exist_ok=True)
            joblib.dump(final, out / f"{task}.joblib")
            col = "p_up" if task == "classifier" else "pred_return"
            hold_frames[t][col] = ho_pred[best][t]
            thresh = cfg.backtest.prob_threshold if task == "classifier" else 0.0
            signal = (hold_frames[t][col] > thresh).astype(float)
            bt = backtest(hold_frames[t]["_daily"], signal, cfg.backtest.cost, hold_days=horizon)
            bt.to_parquet(out / f"backtest_{task}.parquet")
            cv_t = {n: cv_ticker[n][t] for n in candidates}
            base, mine = cv_t[BASELINE].get(metric, np.nan), cv_t[best].get(metric, np.nan)
            per_ticker[t][task] = {
                "selected": best,
                "selection_metric": metric,
                "beats_baseline_cv": bool(mine > base if higher else mine < base),
                "lgbm_params": None,
                "cv": cv_t,
                "holdout": {n: ho_ticker[n][t] for n in candidates},
                "backtest": backtest_stats(bt),
                "feature_importance": importance,
                "pooled_cv": cv_pooled[best],
                "pooled_holdout": ho_pooled[best],
            }

    for t in targets:
        out = model_dir(cfg, t, horizon)
        hold_frames[t].drop(columns="_daily").to_parquet(out / "holdout_predictions.parquet")
        th = data[data["ticker"] == t]
        metrics = {
            "ticker": t,
            "horizon": horizon,
            "scope": "pooled",
            "n_training_tickers": int(data["ticker"].nunique()),
            "features": cols,
            "n_rows": int(len(th)),
            "dev_period": [str(pd.Timestamp(dev["Date"].min()).date()), str(pd.Timestamp(dev["Date"].max()).date())],
            "holdout_period": [
                str(pd.Timestamp(hold["Date"].min()).date()),
                str(pd.Timestamp(hold["Date"].max()).date()),
            ],
            **per_ticker[t],
        }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    summary = {"horizon": horizon, "n_rows": len(data), "n_tickers": int(data["ticker"].nunique()), **pooled_summary}
    (cfg.paths.models / "_pooled" / f"h{horizon}" / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    log.info("pooled %dd done in %.0fs", horizon, time.perf_counter() - start)
    return summary


def train_all_pooled(cfg: Config) -> dict[int, dict]:
    panel = load_panel(cfg)
    results = {}
    for h in cfg.train.horizons:
        try:
            results[h] = train_pooled(cfg, h, panel)
        except Exception as exc:
            log.error("pooled %dd: training failed: %s", h, exc)
    return results
