"""Unit tests for src/drift_score.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.drift_score import (
    compute_baseline_vector,
    compute_style_drift,
    cosine_distance,
    euclidean_distance,
)

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def _make_constant_betas(n: int = 400, seed: int = 9) -> pd.DataFrame:
    """No drift at all: every day has the identical exposure vector."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    fixed = rng.normal(size=len(FACTOR_COLS))
    data = np.tile(fixed, (n, 1))
    betas = pd.DataFrame(data, index=dates, columns=FACTOR_COLS)
    betas.insert(0, "const", 0.0001)
    return betas


def _make_step_change_betas(n: int = 400, break_at: int = 300, seed: int = 9) -> pd.DataFrame:
    """Stable for the baseline period, then a permanent, large shift."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    before = rng.normal(size=len(FACTOR_COLS))
    after = before + np.array([0.5, -0.8, 0.9, -0.3, 0.6, -0.4])
    data = np.where(
        (np.arange(n) < break_at)[:, None],
        np.tile(before, (n, 1)),
        np.tile(after, (n, 1)),
    )
    betas = pd.DataFrame(data, index=dates, columns=FACTOR_COLS)
    betas.insert(0, "const", 0.0001)
    return betas


def test_distance_metrics_are_zero_for_identical_vectors():
    v = np.array([0.5, -0.2, 0.1, 0.0, 0.3, -0.1])
    assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-10)
    assert euclidean_distance(v, v) == pytest.approx(0.0, abs=1e-10)


def test_distance_metrics_are_nonzero_for_different_vectors():
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])
    assert cosine_distance(u, v) == pytest.approx(1.0)  # orthogonal vectors
    assert euclidean_distance(u, v) == pytest.approx(np.sqrt(2))


def test_drift_is_zero_throughout_when_exposure_never_changes():
    """Core sanity check: with no real drift, the score reads (near) zero
    everywhere, including at the very start of the series.
    """
    betas = _make_constant_betas()
    result = compute_style_drift(betas, metric="cosine")

    assert result.drift.iloc[0] == pytest.approx(0.0, abs=1e-8)
    assert (result.drift.abs() < 1e-8).all()
    assert result.flagged_events.empty


def test_drift_detects_permanent_structural_shift():
    betas = _make_step_change_betas()
    result = compute_style_drift(betas, metric="euclidean", n_months=6)

    # Drift well after the break should be much larger than during the
    # (stable) baseline period.
    pre_break_drift = result.drift.iloc[50]
    post_break_drift = result.drift.iloc[-1]
    assert post_break_drift > pre_break_drift
    assert not result.flagged_events.empty
    # The last flagged episode should extend to (or near) the end of the series.
    assert result.flagged_events.iloc[-1]["end"] >= betas.index[-50]


def test_compute_baseline_vector_matches_manual_mean():
    betas = _make_step_change_betas()
    baseline = compute_baseline_vector(betas, n_months=6)

    cutoff = betas.index[0] + pd.DateOffset(months=6)
    expected = betas.loc[: cutoff - pd.Timedelta(days=1), FACTOR_COLS].mean()
    pd.testing.assert_series_equal(baseline, expected, check_names=False)


def test_explicit_mandate_vector_is_used_instead_of_computed_baseline():
    betas = _make_constant_betas()
    mandate = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=FACTOR_COLS)

    result = compute_style_drift(betas, baseline_vector=mandate, metric="euclidean")
    pd.testing.assert_series_equal(result.baseline_vector, mandate, check_names=False)


def test_mandate_vector_missing_factor_raises():
    betas = _make_constant_betas()
    incomplete_mandate = pd.Series([1.0, 0.0], index=["Mkt-RF", "SMB"])
    with pytest.raises(ValueError):
        compute_style_drift(betas, baseline_vector=incomplete_mandate)


def test_invalid_metric_raises():
    betas = _make_constant_betas()
    with pytest.raises(ValueError):
        compute_style_drift(betas, metric="manhattan")


def _make_noisy_baseline_step_change_betas(
    n: int = 400, break_at: int = 300, seed: int = 9
) -> pd.DataFrame:
    """Like `_make_step_change_betas`, but with real day-to-day noise
    during the baseline period too, so its drift distribution has
    nonzero variance (needed to exercise skew/percentile-threshold
    logic, which is degenerate on an exactly-constant baseline).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    before = rng.normal(size=len(FACTOR_COLS))
    after = before + np.array([0.5, -0.8, 0.9, -0.3, 0.6, -0.4])
    data = np.where(
        (np.arange(n) < break_at)[:, None],
        np.tile(before, (n, 1)),
        np.tile(after, (n, 1)),
    )
    data = data + rng.normal(scale=0.03, size=data.shape)
    betas = pd.DataFrame(data, index=dates, columns=FACTOR_COLS)
    betas.insert(0, "const", 0.0001)
    return betas


def test_invalid_threshold_method_raises():
    betas = _make_constant_betas()
    with pytest.raises(ValueError):
        compute_style_drift(betas, threshold_method="bogus")


def test_percentile_threshold_method_runs_and_differs_from_gaussian():
    betas = _make_noisy_baseline_step_change_betas()
    gaussian = compute_style_drift(betas, metric="euclidean", n_months=6, threshold_method="gaussian")
    percentile = compute_style_drift(
        betas, metric="euclidean", n_months=6, threshold_method="percentile", threshold_percentile=95.0
    )

    assert gaussian.threshold_method == "gaussian"
    assert percentile.threshold_method == "percentile"
    # Not asserting a direction (depends on distribution shape) -- just
    # that both are valid, finite, and the method actually changes the
    # computed threshold rather than being ignored.
    assert np.isfinite(gaussian.threshold)
    assert np.isfinite(percentile.threshold)
    assert gaussian.threshold != percentile.threshold


def test_baseline_skew_is_reported():
    betas = _make_noisy_baseline_step_change_betas()
    result = compute_style_drift(betas, metric="euclidean", n_months=6)

    assert np.isfinite(result.baseline_skew)


def test_baseline_skew_is_nan_for_perfectly_constant_baseline():
    betas = _make_constant_betas()
    result = compute_style_drift(betas, metric="cosine")

    assert np.isnan(result.baseline_skew)
