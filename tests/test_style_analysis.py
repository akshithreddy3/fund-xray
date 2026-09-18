"""Unit tests for src/style_analysis.py, using synthetic data so they
run offline and can check recovery of known ground-truth coefficients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.style_analysis import (
    compare_style_results,
    newey_west_lag,
    run_constrained_style_analysis,
    run_unconstrained_ols,
)


def _make_synthetic_factors(n: int = 1000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    factors = pd.DataFrame(
        rng.normal(scale=0.01, size=(n, 6)),
        index=dates,
        columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
    )
    return factors


def test_constrained_weights_sum_to_one_and_are_bounded():
    factors = _make_synthetic_factors()
    true_weights = np.array([0.6, 0.1, 0.1, 0.1, 0.1, 0.0])
    rng = np.random.default_rng(1)
    noise = rng.normal(scale=1e-4, size=len(factors))
    y = pd.Series(factors.to_numpy() @ true_weights + noise, index=factors.index)

    result = run_constrained_style_analysis(y, factors)

    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (result.weights >= -1e-8).all()
    assert (result.weights <= 1.0 + 1e-8).all()


def test_constrained_recovers_known_weights_with_low_noise():
    factors = _make_synthetic_factors()
    true_weights = np.array([0.7, 0.3, 0.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(2)
    noise = rng.normal(scale=1e-5, size=len(factors))
    y = pd.Series(factors.to_numpy() @ true_weights + noise, index=factors.index)

    result = run_constrained_style_analysis(y, factors)

    np.testing.assert_allclose(result.weights.to_numpy(), true_weights, atol=1e-2)
    assert result.r_squared > 0.99


def test_unconstrained_ols_recovers_alpha_and_betas():
    factors = _make_synthetic_factors()
    true_betas = np.array([0.9, -0.2, 0.1, 0.0, 0.0, 0.05])
    true_alpha = 0.0002
    rng = np.random.default_rng(3)
    noise = rng.normal(scale=1e-4, size=len(factors))
    y = pd.Series(true_alpha + factors.to_numpy() @ true_betas + noise, index=factors.index)

    result = run_unconstrained_ols(y, factors)

    assert result.alpha == pytest.approx(true_alpha, abs=5e-5)
    np.testing.assert_allclose(result.weights.to_numpy(), true_betas, atol=0.05)
    assert result.standard_errors is not None
    assert result.nw_lags is not None and result.nw_lags >= 1


def test_newey_west_lag_grows_slowly_with_sample_size():
    assert newey_west_lag(100) == 4
    assert newey_west_lag(2500) >= newey_west_lag(100)
    # Rule of thumb should never suggest zero lags.
    assert newey_west_lag(1) >= 1


def test_compare_style_results_flags_boundary_weights():
    factors = _make_synthetic_factors()
    true_weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(4)
    noise = rng.normal(scale=1e-5, size=len(factors))
    y = pd.Series(factors.to_numpy() @ true_weights + noise, index=factors.index)

    constrained = run_constrained_style_analysis(y, factors)
    unconstrained = run_unconstrained_ols(y, factors)
    comparison = compare_style_results(constrained, unconstrained)

    assert comparison.loc["Mkt-RF", "at_boundary"]
    assert set(comparison.columns) == {
        "constrained_weight",
        "unconstrained_beta",
        "difference",
        "at_boundary",
    }
