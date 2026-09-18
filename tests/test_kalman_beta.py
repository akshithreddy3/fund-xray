"""Unit tests for src/kalman_beta.py, using synthetic data with a known,
dateable structural break so both estimators' behavior around it is
checkable without hitting the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.kalman_beta import (
    compare_lag_around_date,
    kalman_filter_betas,
    rolling_ols_betas,
)


def _make_regime_switch_data(n: int = 800, break_at: int = 500, seed: int = 11):
    """Factor returns plus a fund whose true Mkt-RF beta jumps from 0.5 to 1.5 at `break_at`."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    factors = pd.DataFrame(
        rng.normal(scale=0.01, size=(n, 6)),
        index=dates,
        columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
    )
    true_beta = np.where(np.arange(n) < break_at, 0.5, 1.5)
    noise = rng.normal(scale=1e-4, size=n)
    y = pd.Series(factors["Mkt-RF"].to_numpy() * true_beta + noise, index=dates)
    return y, factors, dates[break_at]


def test_rolling_ols_burn_in_is_nan_not_backfilled():
    y, factors, _ = _make_regime_switch_data()
    result = rolling_ols_betas(y, factors, window=126)

    assert result.betas.iloc[: result.n_burn_in].isna().all().all()
    assert result.betas.iloc[result.n_burn_in :].notna().all().all()
    assert result.n_burn_in == 125


def test_rolling_ols_recovers_beta_within_stable_regime():
    y, factors, break_date = _make_regime_switch_data()
    result = rolling_ols_betas(y, factors, window=126)

    # Well before the break, a full trailing window is pure regime 1 (beta=0.5).
    pre_break_estimate = result.betas.loc[:break_date, "Mkt-RF"].iloc[-1]
    assert pre_break_estimate == pytest.approx(0.5, abs=0.05)


def test_kalman_filter_burn_in_is_excluded_not_backfilled():
    y, factors, _ = _make_regime_switch_data()
    result = kalman_filter_betas(y, factors, warmup_window=126)

    assert result.n_burn_in == 126
    assert len(result.betas) == len(y) - 126
    assert result.betas.index[0] == y.index[126]


def test_kalman_reacts_faster_than_rolling_after_structural_break():
    """The core claim this module exists to support: after a real,
    permanent shift, the Kalman filter should be closer to the new
    true value than the rolling window is, shortly after the break.
    """
    y, factors, break_date = _make_regime_switch_data()
    rolling = rolling_ols_betas(y, factors, window=126)
    kalman = kalman_filter_betas(y, factors, delta=0.02, warmup_window=126)

    check_date = break_date + pd.tseries.offsets.BDay(30)
    true_new_beta = 1.5

    rolling_estimate = rolling.betas.loc[:check_date, "Mkt-RF"].iloc[-1]
    kalman_estimate = kalman.betas.loc[:check_date, "Mkt-RF"].iloc[-1]

    rolling_error = abs(rolling_estimate - true_new_beta)
    kalman_error = abs(kalman_estimate - true_new_beta)
    assert kalman_error < rolling_error


def test_compare_lag_around_date_returns_aligned_columns():
    y, factors, break_date = _make_regime_switch_data()
    rolling = rolling_ols_betas(y, factors, window=126)
    kalman = kalman_filter_betas(y, factors, warmup_window=126)

    table = compare_lag_around_date(rolling, kalman, break_date, factor="Mkt-RF", window_days=20)
    assert list(table.columns) == ["rolling", "kalman"]
    assert not table.empty


def test_kalman_rejects_invalid_delta():
    y, factors, _ = _make_regime_switch_data()
    with pytest.raises(ValueError):
        kalman_filter_betas(y, factors, delta=1.5)
    with pytest.raises(ValueError):
        kalman_filter_betas(y, factors, delta=0.0)
