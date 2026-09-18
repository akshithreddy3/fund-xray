"""Unit tests for src/regime_detection.py.

Uses synthetic two-regime data with a known ground-truth label so the
tests run offline and check something stronger than "it runs": that the
fitted HMM actually recovers the underlying structure and that state
relabeling (calm=lowest vol) is correct regardless of which arbitrary
index hmmlearn assigns internally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regime_detection import (
    build_regime_features,
    fit_regime_hmm,
    validate_against_known_stress_periods,
)


def _make_two_regime_market_data(n: int = 1500, seed: int = 5):
    """Calm regime: low vol, positive drift. Stressed: high vol, negative drift."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=n, freq="B")

    true_regime = np.zeros(n, dtype=int)
    # A handful of well-separated stress blocks so it's easy to reason about.
    stress_blocks = [(300, 340), (900, 990)]
    for start, end in stress_blocks:
        true_regime[start:end] = 1

    returns = np.where(
        true_regime == 0,
        rng.normal(loc=0.0006, scale=0.007, size=n),
        rng.normal(loc=-0.0015, scale=0.025, size=n),
    )
    vix = np.where(
        true_regime == 0,
        rng.normal(loc=14, scale=2, size=n),
        rng.normal(loc=32, scale=5, size=n),
    )
    market_returns = pd.Series(returns, index=dates)
    vix_series = pd.Series(vix, index=dates)
    return market_returns, vix_series, pd.Series(true_regime, index=dates), stress_blocks


def test_build_regime_features_uses_vix_when_provided():
    market_returns, vix, _, _ = _make_two_regime_market_data()
    features, source = build_regime_features(market_returns, vix=vix)
    assert source == "vix"
    assert list(features.columns) == ["return", "vol_signal"]
    assert len(features) == len(market_returns)


def test_build_regime_features_falls_back_to_realized_vol_without_vix():
    market_returns, _, _, _ = _make_two_regime_market_data()
    features, source = build_regime_features(market_returns, vix=None)
    assert source == "realized_vol"
    assert features["vol_signal"].min() >= 0  # a std-based measure can't be negative


def test_hmm_recovers_known_two_regime_structure():
    market_returns, vix, true_regime, _ = _make_two_regime_market_data()
    result = fit_regime_hmm(market_returns, vix=vix, n_regimes=2)

    assert set(result.regime_labels.unique()) <= {"calm", "stressed"}
    # Both labels should actually be used -- a degenerate single-state
    # fit would mean the HMM found no structure at all.
    assert result.regime_labels.nunique() == 2

    predicted_stressed = (result.regime_labels == "stressed").astype(int)
    predicted_stressed = predicted_stressed.reindex(true_regime.index)
    agreement = (predicted_stressed == true_regime).mean()
    assert agreement > 0.9


def test_calm_label_has_lower_volatility_mean_than_stressed():
    market_returns, vix, _, _ = _make_two_regime_market_data()
    result = fit_regime_hmm(market_returns, vix=vix, n_regimes=2)

    assert result.state_means.loc["calm", "vol_signal"] < result.state_means.loc["stressed", "vol_signal"]


def test_three_regime_labels_ordered_by_volatility():
    market_returns, vix, _, _ = _make_two_regime_market_data()
    result = fit_regime_hmm(market_returns, vix=vix, n_regimes=3)

    vol_means = result.state_means["vol_signal"]
    assert vol_means["calm"] < vol_means["elevated"] < vol_means["stressed"]


def test_invalid_n_regimes_raises():
    market_returns, vix, _, _ = _make_two_regime_market_data()
    with pytest.raises(ValueError):
        fit_regime_hmm(market_returns, vix=vix, n_regimes=5)


def test_validate_against_known_stress_periods_matches_synthetic_blocks():
    market_returns, vix, _, stress_blocks = _make_two_regime_market_data()
    result = fit_regime_hmm(market_returns, vix=vix, n_regimes=2)

    dates = market_returns.index
    periods = {
        f"block_{i}": (dates[start].date(), dates[end - 1].date())
        for i, (start, end) in enumerate(stress_blocks)
    }
    validation = validate_against_known_stress_periods(result, stress_periods=periods)

    assert (validation["pct_stressed"] > 0.7).all()
