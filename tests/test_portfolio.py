"""Unit tests for src/portfolio.py, using synthetic exposure vectors so
they run offline (no network) and check known-answer behavior directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    PortfolioHolding,
    compute_blended_exposure,
    compute_portfolio_diversification,
    normalize_weights,
)

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def test_normalize_weights_sums_to_one_and_drops_unavailable():
    holdings = [
        PortfolioHolding("A", weight=60.0),
        PortfolioHolding("B", weight=40.0),
        PortfolioHolding("C", weight=100.0),  # C has no usable data
    ]
    weights = normalize_weights(holdings, available_tickers={"A", "B"})

    assert set(weights) == {"A", "B"}
    assert weights["A"] == pytest.approx(0.6)
    assert weights["B"] == pytest.approx(0.4)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_normalize_weights_raises_on_nonpositive_total():
    holdings = [PortfolioHolding("A", weight=0.0)]
    with pytest.raises(ValueError):
        normalize_weights(holdings, available_tickers={"A"})


def test_compute_blended_exposure_matches_manual_weighted_sum():
    betas = {
        "A": pd.Series({"Mkt-RF": 1.0, "SMB": 0.0}),
        "B": pd.Series({"Mkt-RF": 0.0, "SMB": 1.0}),
    }
    weights = {"A": 0.75, "B": 0.25}
    blended = compute_blended_exposure(betas, weights)

    assert blended["Mkt-RF"] == pytest.approx(0.75)
    assert blended["SMB"] == pytest.approx(0.25)


def test_diversification_identical_holdings_gives_effective_n_one():
    fixed = pd.Series({c: 0.5 for c in FACTOR_COLS})
    betas = {"A": fixed, "B": fixed.copy(), "C": fixed.copy()}
    weights = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}

    result = compute_portfolio_diversification(betas, weights)

    assert result.n_holdings == 3
    assert result.avg_pairwise_similarity == pytest.approx(1.0, abs=1e-6)
    assert result.effective_n == pytest.approx(1.0, abs=1e-6)


def test_diversification_orthogonal_holdings_gives_higher_effective_n():
    betas = {
        "A": pd.Series({"Mkt-RF": 1.0, "SMB": 0.0, "HML": 0.0, "RMW": 0.0, "CMA": 0.0, "Mom": 0.0}),
        "B": pd.Series({"Mkt-RF": 0.0, "SMB": 1.0, "HML": 0.0, "RMW": 0.0, "CMA": 0.0, "Mom": 0.0}),
    }
    weights = {"A": 0.5, "B": 0.5}

    result = compute_portfolio_diversification(betas, weights)

    assert result.avg_pairwise_similarity == pytest.approx(0.0, abs=1e-6)
    assert result.effective_n == pytest.approx(2.0, abs=1e-6)


def test_diversification_single_holding_is_effective_n_one():
    betas = {"A": pd.Series({c: 0.5 for c in FACTOR_COLS})}
    weights = {"A": 1.0}

    result = compute_portfolio_diversification(betas, weights)

    assert result.n_holdings == 1
    assert result.effective_n == pytest.approx(1.0)


def test_diversification_pairwise_matrix_is_symmetric_with_unit_diagonal():
    betas = {
        "A": pd.Series({"Mkt-RF": 1.0, "SMB": 0.2}),
        "B": pd.Series({"Mkt-RF": 0.5, "SMB": 0.8}),
        "C": pd.Series({"Mkt-RF": -0.3, "SMB": 0.1}),
    }
    weights = {"A": 0.4, "B": 0.4, "C": 0.2}

    result = compute_portfolio_diversification(betas, weights)
    matrix = result.pairwise_similarity

    np.testing.assert_allclose(np.diag(matrix.to_numpy()), 1.0)
    pd.testing.assert_frame_equal(matrix, matrix.T)
