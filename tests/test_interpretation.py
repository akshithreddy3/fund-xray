"""Unit tests for src/interpretation.py.

These construct small synthetic dataclass instances (not full model
fits) and assert on each function's `tag` -- the stable classification
-- rather than exact prose, so tests aren't brittle to wording edits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.drift_score import CrowdingScoreResult, DriftScoreResult
from src.interpretation import (
    classify_significance,
    describe_calm_vs_stressed,
    describe_crowding,
    describe_dominant_factor,
    describe_fixed_beta_cross_check,
    describe_portfolio_concentration,
    describe_style_drift,
    describe_style_drift_with_latest,
)
from src.risk_metrics import RegimeRiskMetrics

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def _make_metrics(idio_share: float, market_beta: float = 1.0, vol: float = 0.15) -> RegimeRiskMetrics:
    factor_betas = pd.Series({c: 0.0 for c in FACTOR_COLS})
    factor_betas["Mkt-RF"] = market_beta
    contribution = pd.Series({c: 0.0 for c in FACTOR_COLS})
    contribution["Mkt-RF"] = 1.0 - idio_share
    return RegimeRiskMetrics(
        regime="test",
        n_obs=500,
        var_95=0.02,
        cvar_95=0.03,
        annualized_vol=vol,
        annualized_mean_return=0.08,
        r_squared=1.0 - idio_share,
        factor_betas=factor_betas,
        factor_risk_contribution_pct=contribution,
        idiosyncratic_variance_share=idio_share,
        idiosyncratic_annualized_vol=vol * np.sqrt(idio_share),
        factor_driven_annualized_vol=vol * np.sqrt(1.0 - idio_share),
    )


def test_classify_significance_below_alpha_is_significant():
    result = classify_significance(0.01)
    assert result.tag == "significant"


def test_classify_significance_above_alpha_is_not_significant():
    result = classify_significance(0.5)
    assert result.tag == "not_significant"


def test_classify_significance_handles_nan():
    result = classify_significance(float("nan"))
    assert result.tag == "unknown"


def test_describe_dominant_factor_picks_largest_contributor():
    contribution = pd.Series({"Mkt-RF": 0.6, "SMB": 0.05, "HML": 0.02, "RMW": 0.0, "CMA": 0.0, "Mom": 0.01})
    result = describe_dominant_factor(contribution)
    assert "Mkt-RF" in result.headline


def test_describe_calm_vs_stressed_flags_weakening_diversification():
    calm = _make_metrics(idio_share=0.10, market_beta=1.04)
    stressed = _make_metrics(idio_share=0.02, market_beta=1.04)
    result = describe_calm_vs_stressed(calm, stressed)
    assert result.tag == "diversification_weakens"


def test_describe_calm_vs_stressed_flags_holding_diversification():
    calm = _make_metrics(idio_share=0.10, market_beta=1.04)
    stressed = _make_metrics(idio_share=0.11, market_beta=1.04)
    result = describe_calm_vs_stressed(calm, stressed)
    assert result.tag == "diversification_holds"


def test_describe_fixed_beta_cross_check_robust_when_close():
    cross_check = {
        "reference_regime": "calm",
        "target_regime": "stressed",
        "refit_r_squared": 0.90,
        "fixed_beta_r_squared": 0.88,
    }
    result = describe_fixed_beta_cross_check(cross_check)
    assert result.tag == "robust"


def test_describe_fixed_beta_cross_check_refit_sensitive_when_far_apart():
    cross_check = {
        "reference_regime": "calm",
        "target_regime": "stressed",
        "refit_r_squared": 0.90,
        "fixed_beta_r_squared": 0.40,
    }
    result = describe_fixed_beta_cross_check(cross_check)
    assert result.tag == "refit_sensitive"


def _make_drift_result(flagged: bool, still_elevated: bool) -> DriftScoreResult:
    dates = pd.date_range("2018-01-01", periods=10, freq="B")
    values = [0.01] * 10
    if flagged:
        values[-3:] = [0.5, 0.5, 0.5 if still_elevated else 0.01]
    drift = pd.Series(values, index=dates)
    threshold = 0.2
    flagged_series = drift > threshold
    events = pd.DataFrame()
    if flagged_series.any():
        flagged_dates = drift[flagged_series]
        events = pd.DataFrame(
            [
                {
                    "start": flagged_dates.index.min(),
                    "end": flagged_dates.index.max(),
                    "peak_date": flagged_dates.idxmax(),
                    "peak_drift": float(flagged_dates.max()),
                }
            ]
        )
    return DriftScoreResult(
        drift=drift,
        baseline_vector=pd.Series({c: 0.1 for c in FACTOR_COLS}),
        metric="cosine",
        threshold=threshold,
        threshold_method="gaussian",
        flagged_events=events,
        baseline_skew=0.1,
    )


def test_describe_style_drift_stable_when_never_flagged():
    result = describe_style_drift(_make_drift_result(flagged=False, still_elevated=False))
    assert result.tag == "stable"


def test_describe_style_drift_sustained_when_still_elevated_near_end():
    result = describe_style_drift(_make_drift_result(flagged=True, still_elevated=True))
    assert result.tag == "sustained_drift"


def test_describe_style_drift_transient_when_reverted():
    result = describe_style_drift(_make_drift_result(flagged=True, still_elevated=False))
    assert result.tag == "transient_drift"


def test_describe_style_drift_with_latest_names_top_movers():
    drift_result = _make_drift_result(flagged=True, still_elevated=True)
    latest = pd.Series({c: 0.1 for c in FACTOR_COLS})
    latest["HML"] = -0.9  # large move away from baseline (0.1)
    result = describe_style_drift_with_latest(drift_result, latest)
    assert "HML" in result.detail


def _make_crowding_result(latest_similarity: float, n_used: int, n_total: int) -> CrowdingScoreResult:
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    crowding = pd.Series([latest_similarity] * 5, index=dates)
    n_peers_used = pd.Series([n_used] * 5, index=dates)
    return CrowdingScoreResult(
        crowding=crowding,
        n_peers_used=n_peers_used,
        peer_tickers=[f"T{i}" for i in range(n_total)],
        skipped_tickers=[],
    )


def test_describe_crowding_flags_high_overlap():
    result = describe_crowding(_make_crowding_result(0.95, n_used=4, n_total=4))
    assert result.tag == "crowded"


def test_describe_crowding_flags_differentiated_group():
    result = describe_crowding(_make_crowding_result(0.3, n_used=4, n_total=4))
    assert result.tag == "differentiated"


def test_describe_portfolio_concentration_diversified():
    result = describe_portfolio_concentration(n_holdings=5, effective_n=4.5, avg_pairwise_similarity=0.2)
    assert result.tag == "diversified"


def test_describe_portfolio_concentration_concentrated():
    result = describe_portfolio_concentration(n_holdings=5, effective_n=1.2, avg_pairwise_similarity=0.95)
    assert result.tag == "concentrated"


def test_describe_portfolio_concentration_single_holding():
    result = describe_portfolio_concentration(n_holdings=1, effective_n=1.0, avg_pairwise_similarity=1.0)
    assert result.tag == "single_holding"
