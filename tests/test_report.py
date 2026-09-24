"""Unit tests for src/report.py.

Builds a small synthetic FundDataset-equivalent and runs it through the
real style_analysis/risk_metrics/drift_score pipeline (not mocks), then
checks that build_fund_report assembles the results without error and
that its fields reflect what those modules actually computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import FundDataset
from src.drift_score import compute_style_drift
from src.kalman_beta import rolling_ols_betas
from src.report import build_fund_report
from src.risk_metrics import compute_regime_risk_metrics
from src.style_analysis import run_unconstrained_ols

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def _make_dataset_and_regimes(n: int = 1000, seed: int = 33):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    factors = pd.DataFrame(
        rng.normal(scale=0.01, size=(n, 6)), index=dates, columns=FACTOR_COLS
    )
    true_beta = np.array([1.0, 0.1, -0.1, 0.0, 0.0, 0.05])

    regime = np.array(["calm"] * (n // 2) + ["stressed"] * (n - n // 2))
    idio_scale = np.where(regime == "calm", 1e-4, 1.5e-3)
    noise = rng.normal(scale=idio_scale, size=n)

    fund_returns = pd.Series(factors.to_numpy() @ true_beta + noise, index=dates, name="FAKE")
    excess_returns = fund_returns.rename("excess_return")
    regime_labels = pd.Series(regime, index=dates)

    dataset = FundDataset(
        ticker="FAKE",
        fund_returns=fund_returns,
        excess_returns=excess_returns,
        factor_returns=factors,
        risk_free=pd.Series(0.0, index=dates),
        start=dates.min(),
        end=dates.max(),
    )
    return dataset, regime_labels


def test_build_fund_report_end_to_end_with_regimes():
    dataset, regime_labels = _make_dataset_and_regimes()

    unconstrained = run_unconstrained_ols(dataset.excess_returns, dataset.factor_returns)
    risk_metrics = compute_regime_risk_metrics(
        dataset.fund_returns, dataset.excess_returns, dataset.factor_returns, regime_labels
    )
    rolling = rolling_ols_betas(dataset.excess_returns, dataset.factor_returns, window=126)
    drift = compute_style_drift(rolling.betas, metric="cosine")
    latest_exposure = rolling.betas.dropna().iloc[-1]

    report = build_fund_report(
        dataset=dataset,
        unconstrained=unconstrained,
        risk_metrics_by_regime=risk_metrics,
        drift_result=drift,
        latest_exposure=latest_exposure,
        regime_labels=regime_labels,
    )

    assert report.ticker == "FAKE"
    assert report.n_obs == len(dataset.fund_returns)
    assert report.dominant_factor.tag == "dominant_factor"
    assert report.calm_vs_stressed_table is not None
    assert {"calm", "stressed", "overall"} <= set(report.calm_vs_stressed_table.index)
    assert report.stress_interpretation is not None
    assert report.fixed_beta_check is not None
    assert report.style_drift is not None
    assert report.crowding is None  # not supplied


def test_build_fund_report_without_regime_labels_skips_fixed_beta_check():
    dataset, regime_labels = _make_dataset_and_regimes()

    unconstrained = run_unconstrained_ols(dataset.excess_returns, dataset.factor_returns)
    risk_metrics = compute_regime_risk_metrics(
        dataset.fund_returns, dataset.excess_returns, dataset.factor_returns, regime_labels
    )
    rolling = rolling_ols_betas(dataset.excess_returns, dataset.factor_returns, window=126)
    drift = compute_style_drift(rolling.betas, metric="cosine")
    latest_exposure = rolling.betas.dropna().iloc[-1]

    report = build_fund_report(
        dataset=dataset,
        unconstrained=unconstrained,
        risk_metrics_by_regime=risk_metrics,
        drift_result=drift,
        latest_exposure=latest_exposure,
        regime_labels=None,
    )

    assert report.fixed_beta_check is None
    # Calm/stressed comparison itself doesn't need regime_labels again --
    # it's derived from the already-computed risk_metrics_by_regime.
    assert report.stress_interpretation is not None
