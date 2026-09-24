"""Unit tests for src/risk_metrics.py, using synthetic data so the
Euler variance decomposition and VaR/CVaR calculations can be checked
against known closed-form or hand-computed answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk_metrics import (
    OVERALL_LABEL,
    compute_fixed_beta_cross_check,
    compute_regime_risk_metrics,
    evaluate_fixed_beta_r_squared,
    historical_var_cvar,
    split_by_regime,
)
from src.style_analysis import run_unconstrained_ols


def test_historical_var_cvar_on_known_distribution():
    # 100 evenly spaced returns from -0.10 to +0.09; the 5th percentile
    # (index 4, 0-indexed, ascending) is unambiguous with this design.
    returns = pd.Series(np.arange(-0.10, 0.09, 0.001))
    var_95, cvar_95 = historical_var_cvar(returns, confidence=0.95)

    threshold = np.quantile(returns, 0.05)
    expected_var = -threshold
    expected_cvar = -returns[returns <= threshold].mean()

    assert var_95 == pytest.approx(expected_var)
    assert cvar_95 == pytest.approx(expected_cvar)
    # CVaR (average of the worst days) should be at least as large as VaR
    # (the boundary of those days) for a left-tail loss measure.
    assert cvar_95 >= var_95


def test_historical_var_cvar_matches_hand_computed_example():
    # 21 points chosen so the 5th percentile (index = 0.05*(21-1) = 1.0)
    # lands exactly on a sorted observation with no interpolation, so
    # the expected VaR/CVaR can be computed by hand unambiguously.
    returns = pd.Series([-0.10, -0.05] + [0.01] * 19)
    var_95, cvar_95 = historical_var_cvar(returns, confidence=0.95)
    assert var_95 == pytest.approx(0.05, abs=1e-6)
    assert cvar_95 == pytest.approx(0.075, abs=1e-6)  # mean(0.10, 0.05)


def _make_two_regime_fund_data(n: int = 1000, seed: int = 21):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    factors = pd.DataFrame(
        rng.normal(scale=0.01, size=(n, 6)),
        index=dates,
        columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
    )
    true_beta = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    regime = np.array(["calm"] * (n // 2) + ["stressed"] * (n - n // 2))
    # Stressed regime has much higher idiosyncratic noise -> lower R^2,
    # higher vol, higher VaR/CVaR -- the exact pattern we want to detect.
    idio_scale = np.where(regime == "calm", 1e-4, 1.5e-3)
    noise = rng.normal(scale=idio_scale, size=n)

    fund_returns = pd.Series(factors.to_numpy() @ true_beta + noise, index=dates)
    excess_returns = fund_returns  # rf ~ 0 for this synthetic test
    regime_labels = pd.Series(regime, index=dates)
    return fund_returns, excess_returns, factors, regime_labels


def test_compute_regime_risk_metrics_includes_overall_and_all_regimes():
    fund_returns, excess_returns, factors, regime_labels = _make_two_regime_fund_data()
    metrics = compute_regime_risk_metrics(fund_returns, excess_returns, factors, regime_labels)

    assert set(metrics.keys()) == {OVERALL_LABEL, "calm", "stressed"}
    assert metrics["calm"].n_obs + metrics["stressed"].n_obs == metrics[OVERALL_LABEL].n_obs


def test_stressed_regime_has_higher_vol_and_var_than_calm():
    fund_returns, excess_returns, factors, regime_labels = _make_two_regime_fund_data()
    metrics = compute_regime_risk_metrics(fund_returns, excess_returns, factors, regime_labels)

    assert metrics["stressed"].annualized_vol > metrics["calm"].annualized_vol
    assert metrics["stressed"].var_95 > metrics["calm"].var_95
    assert metrics["stressed"].cvar_95 > metrics["calm"].cvar_95
    # More idiosyncratic noise -> lower R^2 in the stressed regime here.
    assert metrics["stressed"].r_squared < metrics["calm"].r_squared


def test_factor_risk_contribution_sums_to_r_squared():
    fund_returns, excess_returns, factors, regime_labels = _make_two_regime_fund_data()
    metrics = compute_regime_risk_metrics(fund_returns, excess_returns, factors, regime_labels)

    for m in metrics.values():
        total_factor_contribution = float(m.factor_risk_contribution_pct.sum())
        assert total_factor_contribution == pytest.approx(m.r_squared, abs=1e-6)
        assert m.idiosyncratic_variance_share == pytest.approx(1.0 - m.r_squared, abs=1e-6)


def test_recovers_near_true_beta_in_each_regime():
    fund_returns, excess_returns, factors, regime_labels = _make_two_regime_fund_data()
    metrics = compute_regime_risk_metrics(fund_returns, excess_returns, factors, regime_labels)

    for label in ("calm", "stressed"):
        assert metrics[label].factor_betas["Mkt-RF"] == pytest.approx(1.0, abs=0.1)


def test_non_overlapping_regime_dates_are_dropped_and_logged(caplog):
    fund_returns, excess_returns, factors, regime_labels = _make_two_regime_fund_data()
    truncated_regime_labels = regime_labels.iloc[:-50]  # regime data ends earlier than fund data

    with caplog.at_level("WARNING"):
        metrics = compute_regime_risk_metrics(
            fund_returns, excess_returns, factors, truncated_regime_labels
        )

    assert metrics[OVERALL_LABEL].n_obs == len(truncated_regime_labels)
    assert "Dropped" in caplog.text


def test_idiosyncratic_and_factor_vol_recombine_to_total_vol():
    fund_returns, excess_returns, factors, regime_labels = _make_two_regime_fund_data()
    metrics = compute_regime_risk_metrics(fund_returns, excess_returns, factors, regime_labels)

    for m in metrics.values():
        recombined = np.sqrt(m.idiosyncratic_annualized_vol**2 + m.factor_driven_annualized_vol**2)
        assert recombined == pytest.approx(m.annualized_vol, rel=1e-6)


def test_split_by_regime_matches_compute_regime_risk_metrics_grouping():
    fund_returns, excess_returns, factors, regime_labels = _make_two_regime_fund_data()
    subsets = split_by_regime(fund_returns, excess_returns, factors, regime_labels)

    assert set(subsets) == {OVERALL_LABEL, "calm", "stressed"}
    assert len(subsets["calm"]["fund"]) + len(subsets["stressed"]["fund"]) == len(
        subsets[OVERALL_LABEL]["fund"]
    )


def test_evaluate_fixed_beta_r_squared_matches_refit_for_identical_betas():
    fund_returns, excess_returns, factors, regime_labels = _make_two_regime_fund_data()
    subsets = split_by_regime(fund_returns, excess_returns, factors, regime_labels)
    calm = subsets["calm"]

    fit = run_unconstrained_ols(calm["excess"], calm["factors"])
    fixed_r_squared = evaluate_fixed_beta_r_squared(
        calm["excess"], calm["factors"], fit.alpha, fit.weights
    )

    # Applying a regime's own fitted coefficients back to itself (fixed,
    # not refit) should reproduce that regime's own R^2.
    assert fixed_r_squared == pytest.approx(fit.r_squared, abs=1e-6)


def test_fixed_beta_cross_check_reflects_true_beta_shift():
    n = 1000
    rng = np.random.default_rng(11)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    factors = pd.DataFrame(
        rng.normal(scale=0.01, size=(n, 6)),
        index=dates,
        columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
    )
    calm_beta = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    stressed_beta = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])  # genuinely different exposure
    regime = np.array(["calm"] * (n // 2) + ["stressed"] * (n - n // 2))
    beta_by_row = np.where((regime == "calm")[:, None], calm_beta, stressed_beta)
    noise = rng.normal(scale=1e-4, size=n)
    fund_returns = pd.Series(
        np.einsum("ij,ij->i", factors.to_numpy(), beta_by_row) + noise, index=dates
    )
    regime_labels = pd.Series(regime, index=dates)

    result = compute_fixed_beta_cross_check(
        fund_returns, fund_returns, factors, regime_labels,
        reference_regime="calm", target_regime="stressed",
    )

    # Calm beta applied, un-refit, to genuinely-different stressed data
    # should fit far worse than the stressed regime's own refit.
    assert result["fixed_beta_r_squared"] < result["refit_r_squared"] - 0.1
