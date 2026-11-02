import numpy as np
import pandas as pd

from smp.stats import block_bootstrap_sharpe_diff, cluster_bootstrap_mean, holm, verdict


def test_cluster_bootstrap_detects_real_and_null_differences():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(8), 12)
    real = cluster_bootstrap_mean(pd.Series(0.02 + rng.normal(0, 0.01, 96)), pd.Series(clusters))
    assert real["ci_low"] > 0 and real["p_value"] < 0.01
    null = cluster_bootstrap_mean(pd.Series(rng.normal(0, 0.01, 96)), pd.Series(clusters))
    assert null["ci_low"] < 0 < null["ci_high"] and null["p_value"] > 0.05


def test_cluster_bootstrap_is_wider_when_clusters_are_correlated():
    rng = np.random.default_rng(1)
    clusters = np.repeat(np.arange(8), 12)
    shared = np.repeat(rng.normal(0, 0.02, 8), 12)  # whole time periods move together
    d = pd.Series(shared + rng.normal(0, 0.002, 96))
    clustered = cluster_bootstrap_mean(d, pd.Series(clusters))
    naive = cluster_bootstrap_mean(d, pd.Series(np.arange(96)))
    assert (clustered["ci_high"] - clustered["ci_low"]) > 2 * (naive["ci_high"] - naive["ci_low"])


def test_block_bootstrap_sharpe():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=2000)
    a = pd.Series(rng.normal(0.0008, 0.01, 2000), index=idx)
    same = block_bootstrap_sharpe_diff(a, a)
    assert same["mean"] == 0 and same["ci_low"] == 0 == same["ci_high"]
    worse = block_bootstrap_sharpe_diff(a - 0.001, a)
    assert worse["ci_high"] < 0


def test_holm_and_verdict():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == 0.03 and adj["b"] == 0.08 and adj["c"] == 0.5
    assert verdict(0.01, 0.03, higher_is_better=True) == "better"
    assert verdict(0.01, 0.03, higher_is_better=False) == "worse"
    assert verdict(0.01, 0.08, higher_is_better=True) == "no clear difference"
