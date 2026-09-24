"""Fund X-Ray Summary report: the one-screen synthesis of every module in src/.

`build_fund_report` takes results the caller has *already computed*
(style analysis, regime risk metrics, drift) -- it never fetches data or
fits a model itself, so it stays fast, Streamlit-independent, and free
of duplicate network/model-fitting cost when the caller (typically
`app.py`, which already caches each of these calls) assembles a report.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data_loader import FundDataset
from src.drift_score import CrowdingScoreResult, DriftScoreResult
from src.interpretation import (
    Interpretation,
    describe_calm_vs_stressed,
    describe_crowding,
    describe_dominant_factor,
    describe_fixed_beta_cross_check,
    describe_style_drift_with_latest,
)
from src.risk_metrics import (
    OVERALL_LABEL,
    RegimeRiskMetrics,
    compute_fixed_beta_cross_check,
)
from src.style_analysis import StyleAnalysisResult

CALM_LABEL = "calm"
STRESSED_LABEL = "stressed"


@dataclass
class FundReport:
    ticker: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    n_obs: int

    unconstrained_weights: pd.Series
    dominant_factor: Interpretation

    calm_vs_stressed_table: pd.DataFrame | None  # None if calm/stressed regimes unavailable
    stress_interpretation: Interpretation | None
    fixed_beta_check: Interpretation | None

    style_drift: Interpretation
    crowding: Interpretation | None


def _regime_comparison_rows(risk_metrics_by_regime: dict[str, RegimeRiskMetrics]) -> pd.DataFrame:
    rows = {}
    for label in (CALM_LABEL, STRESSED_LABEL, OVERALL_LABEL):
        m = risk_metrics_by_regime.get(label)
        if m is None:
            continue
        rows[label] = {
            "n_obs": m.n_obs,
            "annualized_vol": m.annualized_vol,
            "VaR_95": m.var_95,
            "CVaR_95": m.cvar_95,
            "R_squared": m.r_squared,
            "idiosyncratic_variance_share": m.idiosyncratic_variance_share,
            "idiosyncratic_annualized_vol": m.idiosyncratic_annualized_vol,
            "factor_driven_annualized_vol": m.factor_driven_annualized_vol,
            f"beta_{_dominant_label(m)}": m.factor_betas.get(_dominant_label(m)),
        }
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def _dominant_label(m: RegimeRiskMetrics) -> str:
    return str(m.factor_risk_contribution_pct.abs().idxmax())


def build_fund_report(
    dataset: FundDataset,
    unconstrained: StyleAnalysisResult,
    risk_metrics_by_regime: dict[str, RegimeRiskMetrics],
    drift_result: DriftScoreResult,
    latest_exposure: pd.Series,
    regime_labels: pd.Series | None = None,
    crowding_result: CrowdingScoreResult | None = None,
) -> FundReport:
    """Assemble a `FundReport` from results the caller already computed.

    `latest_exposure` is the most recent row of the rolling/Kalman beta
    DataFrame used to compute `drift_result` -- passed explicitly so this
    function doesn't need to reach back into `kalman_beta.py` itself.
    `regime_labels` is the same series used to produce
    `risk_metrics_by_regime` (`regime_detection.fit_regime_hmm(...).regime_labels`);
    it's needed again here (rather than reusable from `risk_metrics_by_regime`)
    because the fixed-beta cross-check needs the raw per-day labels, not
    just the already-aggregated per-regime metrics.
    """
    overall = risk_metrics_by_regime.get(OVERALL_LABEL)
    dominant_factor = (
        describe_dominant_factor(overall.factor_risk_contribution_pct)
        if overall is not None
        else Interpretation(tag="unknown", headline="No risk metrics available")
    )

    calm = risk_metrics_by_regime.get(CALM_LABEL)
    stressed = risk_metrics_by_regime.get(STRESSED_LABEL)
    calm_vs_stressed_table: pd.DataFrame | None = None
    stress_interpretation: Interpretation | None = None
    fixed_beta_check: Interpretation | None = None

    if calm is not None and stressed is not None:
        calm_vs_stressed_table = _regime_comparison_rows(risk_metrics_by_regime)
        stress_interpretation = describe_calm_vs_stressed(calm, stressed)
        if regime_labels is not None:
            try:
                cross_check = compute_fixed_beta_cross_check(
                    dataset.fund_returns,
                    dataset.excess_returns,
                    dataset.factor_returns,
                    regime_labels,
                    reference_regime=CALM_LABEL,
                    target_regime=STRESSED_LABEL,
                )
                fixed_beta_check = describe_fixed_beta_cross_check(cross_check)
            except ValueError:
                fixed_beta_check = None

    style_drift = describe_style_drift_with_latest(drift_result, latest_exposure)
    crowding = describe_crowding(crowding_result) if crowding_result is not None else None

    return FundReport(
        ticker=dataset.ticker,
        period_start=dataset.start,
        period_end=dataset.end,
        n_obs=len(dataset.fund_returns),
        unconstrained_weights=unconstrained.weights,
        dominant_factor=dominant_factor,
        calm_vs_stressed_table=calm_vs_stressed_table,
        stress_interpretation=stress_interpretation,
        fixed_beta_check=fixed_beta_check,
        style_drift=style_drift,
        crowding=crowding,
    )
