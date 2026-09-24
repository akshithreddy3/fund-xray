"""Unit tests for src/style_analysis.py, using synthetic data so they
run offline and can check recovery of known ground-truth coefficients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.style_analysis import (
    compare_style_results,
    compute_factor_vif,
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


def test_compute_factor_vif_flags_collinear_factor():
    factors = _make_synthetic_factors()
    # Make SMB an almost-exact linear combination of the other factors:
    # its VIF should blow up, while the independent factors it's built
    # from should stay low.
    factors["SMB"] = 0.5 * factors["HML"] + 0.5 * factors["Mom"] + factors["SMB"] * 1e-6

    vif = compute_factor_vif(factors)

    assert vif["SMB"] > 10.0
    assert vif["RMW"] < 5.0  # untouched, independent factor stays low


def test_compute_factor_vif_near_one_for_independent_factors():
    factors = _make_synthetic_factors()  # i.i.d. normal columns, no correlation by construction
    vif = compute_factor_vif(factors)

    assert (vif < 1.5).all()
