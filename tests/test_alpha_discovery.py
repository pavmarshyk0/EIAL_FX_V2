"""Tests protect hypothesis DSL, time ordering, train/test boundary, and report honesty."""
import numpy as np
import pandas as pd
import pytest

from alpha_discovery.factors import validated, factor_series, default_hypotheses
from alpha_discovery.engine import frame_from_rates, rolling_predictions, research
from alpha_discovery.llm import propose_ollama


def rates(n=500):
    rng = np.random.default_rng(17)
    dt = pd.bdate_range("2020-01-01", periods=n)
    r1 = rng.normal(0, .003, n)
    r2 = rng.normal(0, .0025, n)
    return pd.DataFrame({"eurusd": np.exp(np.log(1.13) + np.cumsum(r1)),
                         "eurpln": np.exp(np.log(4.30) + np.cumsum(r2)),
                         "usdpln": np.exp(np.log(4.30) + np.cumsum(r2) - np.log(1.13) - np.cumsum(r1))}, index=dt)


def test_reject_injection_and_invalid_llm_models():
    for candidate in [
        {"family": "python", "source": "eurusd", "lookback": 5, "code": "__import__('os')"},
        {"family": "momentum", "source": "eurusd", "lookback": -1},
        {"family": "vol_ratio", "source": "eurpln", "short": 40, "long": 10},
        {"family": "cross_difference", "source": "eurusd", "other": "eurusd", "lookback": 5},
    ]:
        with pytest.raises(ValueError):
            validated(candidate)
    with pytest.raises(ValueError):
        propose_ollama("gemma3:4b", endpoint="https://arbitrary.example.com")


def test_factor_has_no_future_peeking():
    rr = rates(120)
    log1 = np.log(rr)
    log2 = log1.copy()
    log2.iloc[100:, :] += 5
    for spec in default_hypotheses(36):
        a = factor_series(log1, spec).iloc[:95]
        b = factor_series(log2, spec).iloc[:95]
        pd.testing.assert_series_equal(a, b)


def test_horizon_target_and_cross_pair_identity():
    rr = rates(170)
    lp, x, y = frame_from_rates(rr, horizon=5)
    assert np.isclose(y.iloc[42]["eurusd"], np.log(rr.iloc[47].eurusd / rr.iloc[42].eurusd))
    assert np.isclose(y.iloc[42]["eurpln"] - y.iloc[42]["eurusd"],
                      lp.iloc[47].usdpln - lp.iloc[42].usdpln)
    assert y.tail(5).isna().all().all()


def test_last_future_labels_do_not_change_first_prediction():
    rr = rates(325)
    lp, x, y = frame_from_rates(rr, horizon=5)
    dates = x.index[220:235]
    a = rolling_predictions(x, y["eurusd"], dates, horizon=5, block=7, min_train=120)
    modified = y["eurusd"].copy()
    modified.iloc[215:] = 500
    b = rolling_predictions(x, modified, dates, horizon=5, block=7, min_train=120)
    # First training block ends <= 214, so modification at 215 cannot affect it.
    np.testing.assert_allclose(a.iloc[:7], b.iloc[:7])


def test_report_is_exploratory_and_no_automatic_test_use(tmp_path):
    rr = rates(500)
    dev_end = str(rr.index[380].date())
    rep = research(rr, default_hypotheses(6), tmp_path, dev_end, horizon=1, min_train=110)
    assert "CONFIRMED" not in rep["research_status"].replace("NO CONFIRMED", "")
    assert rep["evaluation_2026_status"] == "LOCKED unless --explore-2026"
    assert (tmp_path / "candidates.json").exists()
    assert not (tmp_path / "exploratory_predictions.csv").exists()
    assert rep["n_candidates_submitted"] == 6


def test_explicit_exploratory_eval_only_after_dev(tmp_path):
    rr = rates(500)
    dev_end = str(rr.index[390].date())
    rep = research(rr, default_hypotheses(4), tmp_path, dev_end=dev_end,
                   min_train=120, explore_after_dev=True)
    assert "EXPLORATORY" in rep["evaluation_2026_status"]
    df = pd.read_csv(tmp_path / "exploratory_predictions.csv")
    assert (df["date"] > dev_end).all()
