import json

import pytest
import yaml

import smp.experiment as ex
from conftest import make_ohlcv


def test_deep_merge_and_spec_validation(tmp_path):
    assert ex.deep_merge({"a": {"b": 1, "c": 2}, "d": 3}, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}, "d": 3}
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "name": "x",
                "question": "q",
                "kind": "direction",
                "metric": "m",
                "control": "missing",
                "variants": {"a": {}},
            }
        )
    )
    with pytest.raises(ValueError, match="control"):
        ex.load_spec(bad)
    assert ex.higher_is_better("classifier.roc_auc") and not ex.higher_is_better("vol.rmse")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A project-like folder with synthetic clean data and a base config pointing at it."""
    from smp.clean import clean_all
    from smp.config import Config

    paths = {k: str(tmp_path / k) for k in ("raw", "clean", "features", "models", "logs", "history")}
    base = {
        "data": {"tickers": ["AAA", "BBB"], "market_tickers": [], "universe": [], "earnings": False},
        "train": {"horizons": [1], "n_splits": 3, "holdout_days": 100},
        "risk": {"horizons": [5], "allocation_horizon": 5},
        "paths": paths,
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(base))
    cfg = Config(**base)
    cfg.paths.raw.mkdir(parents=True)
    for i, t in enumerate(["AAA", "BBB"]):
        make_ohlcv(n=700, seed=i).to_parquet(cfg.paths.raw / f"{t}.parquet")
    clean_all(cfg)
    monkeypatch.setattr(ex, "DEFAULT_CONFIG", cfg_file)
    monkeypatch.setattr(ex, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(ex, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(ex, "RESULTS_PAGE", tmp_path / "docs" / "RESULTS.md")
    return tmp_path


def test_direction_experiment_end_to_end(sandbox):
    spec = sandbox / "demo.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "question": "Does a stronger logistic penalty matter?",
                "kind": "direction",
                "metric": "classifier.roc_auc",
                "control": "default",
                "compare_models": ["baseline"],
                "variants": {"default": {}, "two_horizons": {"train": {"horizons": [1, 5]}}},
            }
        )
    )
    result = ex.run_experiment(spec)
    v = result["variants"]["two_horizons"]
    assert v["n_pairs"] == 2 * 3  # 2 tickers x 1 shared horizon x 3 folds
    assert v["verdict"] in {"better", "worse", "no clear difference"}
    assert result["models"]["baseline"]["n_pairs"] == 6
    md = (sandbox / "reports" / "demo.md").read_text()
    assert "## Variants vs control" in md and "two_horizons" in md and "## Breakdown" in md
    assert json.loads((sandbox / "reports" / "demo.json").read_text())["control"] == "default"
    assert "demo" in (sandbox / "docs" / "RESULTS.md").read_text()

    # second run reuses results (fingerprints match) and gives the same numbers
    again = ex.run_experiment(spec)
    assert again["variants"]["two_horizons"]["diff"] == v["diff"]


def test_allocation_experiment(sandbox):
    spec = sandbox / "alloc.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "name": "alloc",
                "question": "Does a lower vol target help?",
                "kind": "allocation",
                "metric": "sharpe",
                "control": "default",
                "strategy": "trailing_vol_target",
                "compare_strategies": ["buy_hold"],
                "base": {"risk": {"allocation_horizon": 5}},
                "variants": {"default": {}, "low_target": {"risk": {"target_vol": 0.08}}},
            }
        )
    )
    result = ex.run_experiment(spec)
    assert "diff" in result["variants"]["low_target"] and "buy_hold" in result["strategies"]
    assert "## `trailing_vol_target` vs other strategies" in (sandbox / "reports" / "alloc.md").read_text()
