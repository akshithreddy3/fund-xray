"""Plain-English interpretation layer.

Every other module in `src/` answers a quantitative question (a beta, a
p-value, a variance share) but returns numbers, not sentences -- by
design, so the statistical core stays independently testable against
synthetic ground truth. This module is the one place that turns those
numbers into the kind of plain-English read a non-quant investor can
act on, so that narrative text isn't scattered as ad-hoc strings across
`app.py` (or duplicated per Streamlit page).

Every function here is pure (no Streamlit, no network, no randomness)
and returns an `Interpretation`: a short, stable `tag` a caller (or a
test) can branch/assert on, plus `headline`/`detail` prose. Keeping the
tag separate from the wording means the classification logic is
testable without pinning down exact sentences that would make tests
brittle to copy edits.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config
from src.drift_score import CrowdingScoreResult, DriftScoreResult
from src.risk_metrics import RegimeRiskMetrics


@dataclass
class Interpretation:
    tag: str
    headline: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Static style analysis
# ---------------------------------------------------------------------------


def classify_significance(
    p_value: float, alpha: float = config.SIGNIFICANCE_ALPHA
) -> Interpretation:
    """Whether an unconstrained regression beta is distinguishable from zero.

    Uses the Newey-West HAC p-value already computed by
    `style_analysis.run_unconstrained_ols` -- this function only
    classifies a number that already exists, it doesn't compute anything
    new.
    """
    if pd.isna(p_value):
        return Interpretation(tag="unknown", headline="Significance not available")
    if p_value < alpha:
        return Interpretation(
            tag="significant",
            headline="Statistically distinguishable from zero",
            detail=f"p={p_value:.3f} < {alpha:g}",
        )
    return Interpretation(
        tag="not_significant",
        headline="Not statistically distinguishable from zero",
        detail=f"p={p_value:.3f} >= {alpha:g} -- could plausibly be noise at conventional levels",
    )


def describe_dominant_factor(factor_risk_contribution_pct: pd.Series) -> Interpretation:
    """Which single factor drives the largest share of this fund's risk.

    Reads the Euler variance decomposition already computed by
    `risk_metrics.compute_regime_risk_metrics` -- the decomposition sums
    exactly to the factor-driven share of total variance (see
    `risk_metrics.RegimeRiskMetrics`), so "dominant factor" here means
    "largest individual contributor to that sum," not a separate model.
    """
    if factor_risk_contribution_pct.empty:
        return Interpretation(tag="unknown", headline="No factor exposure available")

    dominant = factor_risk_contribution_pct.abs().idxmax()
    share = float(factor_risk_contribution_pct[dominant])
    total_factor_share = float(factor_risk_contribution_pct.sum())

    return Interpretation(
        tag="dominant_factor",
        headline=f"{dominant} drives {share:.0%} of this fund's total risk",
        detail=(
            f"Across all six factors, {total_factor_share:.0%} of this fund's return "
            f"variation is explained by known market factors; the rest is fund-specific "
            f"(manager decisions, individual holdings) rather than shared with the broader "
            f"market."
        ),
    )


# ---------------------------------------------------------------------------
# Regime-conditional risk
# ---------------------------------------------------------------------------

_BETA_STABLE_THRESHOLD = 0.15  # abs. change in Mkt-RF beta considered "essentially unchanged"
_MARKET_FACTOR = "Mkt-RF"


def describe_calm_vs_stressed(
    calm: RegimeRiskMetrics, stressed: RegimeRiskMetrics
) -> Interpretation:
    """The headline "does diversification weaken in stress?" read.

    Statistically grounded in exactly what `risk_metrics.py` computes:
    idiosyncratic variance share is `1 - (factor-driven share)`, and the
    Euler decomposition is unit-tested to sum to R^2, so a share that
    "collapses" is a real property of the fitted decomposition, not a
    display artifact. This function only translates that number into
    English and classifies its magnitude -- it does not re-derive it.
    """
    idio_delta = stressed.idiosyncratic_variance_share - calm.idiosyncratic_variance_share
    market_beta_delta = float(
        stressed.factor_betas.get(_MARKET_FACTOR, float("nan"))
        - calm.factor_betas.get(_MARKET_FACTOR, float("nan"))
    )
    beta_stable = abs(market_beta_delta) < _BETA_STABLE_THRESHOLD

    if idio_delta > -0.02:
        tag = "diversification_holds"
        headline = "This fund's diversification benefit holds up in stressed markets."
        detail = (
            f"Idiosyncratic (fund-specific) risk was {calm.idiosyncratic_variance_share:.1%} "
            f"of total risk in calm markets and {stressed.idiosyncratic_variance_share:.1%} "
            f"in stressed markets -- a fund-specific risk buffer that doesn't meaningfully "
            f"shrink under stress."
        )
    else:
        tag = "diversification_weakens"
        beta_clause = (
            "while its market exposure barely changes"
            if beta_stable
            else "as its market exposure itself shifts too"
        )
        headline = (
            "During stressed markets, almost all of this fund's return variation becomes "
            "explained by common market factors -- the fund-specific component that provided "
            f"some diversification in normal conditions largely disappears, {beta_clause}."
        )
        detail = (
            f"Idiosyncratic (fund-specific) risk share: {calm.idiosyncratic_variance_share:.1%} "
            f"in calm markets -> {stressed.idiosyncratic_variance_share:.1%} in stressed "
            f"markets. In absolute terms, idiosyncratic volatility moved from "
            f"{calm.idiosyncratic_annualized_vol:.1%} to {stressed.idiosyncratic_annualized_vol:.1%} "
            f"(annualized), while market-factor-driven volatility moved from "
            f"{calm.factor_driven_annualized_vol:.1%} to {stressed.factor_driven_annualized_vol:.1%}. "
            f"Mkt-RF beta: {calm.factor_betas.get(_MARKET_FACTOR, float('nan')):.2f} -> "
            f"{stressed.factor_betas.get(_MARKET_FACTOR, float('nan')):.2f}."
        )

    return Interpretation(tag=tag, headline=headline, detail=detail)


def describe_fixed_beta_cross_check(cross_check: dict[str, float | str]) -> Interpretation:
    """Whether the calm-vs-stressed R^2 difference survives without refitting.

    Reads the output of `risk_metrics.compute_fixed_beta_cross_check`: if
    the target regime's R^2 using a fixed (not refit) reference-regime
    beta is still meaningfully different from that reference regime's
    own R^2, the regime-conditional finding isn't just an artifact of
    independently refitting each regime's coefficients.
    """
    refit = float(cross_check["refit_r_squared"])
    fixed = float(cross_check["fixed_beta_r_squared"])
    target = cross_check["target_regime"]
    reference = cross_check["reference_regime"]

    gap = abs(refit - fixed)
    if gap < 0.05:
        tag = "robust"
        headline = (
            f"The '{target}'-regime risk pattern holds even using '{reference}'-regime "
            f"exposure with no refitting -- it isn't an artifact of giving the model extra "
            f"freedom to refit."
        )
    else:
        tag = "refit_sensitive"
        headline = (
            f"Some of the '{target}'-regime R^2 change depends on refitting exposure "
            f"specifically to that regime's data (R^2 using fixed '{reference}'-regime "
            f"exposure: {fixed:.1%} vs. {target}-regime refit: {refit:.1%})."
        )
    return Interpretation(
        tag=tag,
        headline=headline,
        detail=f"refit R^2={refit:.1%}, fixed-beta R^2={fixed:.1%} (gap={gap:.1%})",
    )


# ---------------------------------------------------------------------------
# Style drift
# ---------------------------------------------------------------------------


def describe_style_drift(drift_result: DriftScoreResult) -> Interpretation:
    """Has this fund's style meaningfully -- and lastingly -- changed?

    Classifies the already-computed `flagged_events` (see
    `drift_score.compute_style_drift`) rather than re-deriving drift:
    "stable" (never flagged), "transient" (flagged episodes that end
    well before the end of the series), or "sustained" (the most recent
    flagged episode runs to, or nearly to, the end of the series --
    still different from baseline, not necessarily still moving).
    """
    events = drift_result.flagged_events
    if events.empty:
        return Interpretation(
            tag="stable",
            headline="This fund's factor exposure has stayed within its own historical baseline.",
            detail="No drift episode exceeded the baseline-calibrated threshold.",
        )

    last_event = events.iloc[-1]
    series_end = drift_result.drift.index.max()
    still_elevated = drift_result.drift.iloc[-1] > drift_result.threshold
    near_end = (series_end - last_event["end"]).days <= 60

    if still_elevated and near_end:
        tag = "sustained_drift"
        headline = (
            "This fund's style has moved away from its own baseline and has not reverted -- "
            "read this as a lasting change in how the fund invests, not a temporary swing."
        )
    else:
        tag = "transient_drift"
        headline = (
            "This fund's style drifted away from its own baseline at times, but has since "
            "reverted."
        )
    return Interpretation(tag=tag, headline=headline)


def describe_style_drift_with_latest(
    drift_result: DriftScoreResult, latest_exposure: pd.Series, n: int = 2
) -> Interpretation:
    """Same as `describe_style_drift`, plus the specific factors that moved most.

    Takes the fund's most recent exposure vector explicitly (callers
    already have it -- e.g. the last row of a rolling/Kalman beta
    DataFrame) so the biggest movers can be named without this module
    reaching back into `kalman_beta.py` itself.
    """
    base = describe_style_drift(drift_result)
    diff = (latest_exposure.reindex(drift_result.baseline_vector.index) - drift_result.baseline_vector).dropna()
    if diff.empty:
        return base

    movers = diff.abs().sort_values(ascending=False).head(n).index.tolist()
    mover_strs = [
        f"{factor} ({drift_result.baseline_vector[factor]:+.2f} -> {latest_exposure[factor]:+.2f})"
        for factor in movers
    ]
    detail = f"Largest moves since baseline: {', '.join(mover_strs)}."
    return Interpretation(tag=base.tag, headline=base.headline, detail=detail)


# ---------------------------------------------------------------------------
# Crowding / diversification
# ---------------------------------------------------------------------------

_CROWDING_HIGH_THRESHOLD = 0.85


def describe_crowding(crowding_result: CrowdingScoreResult) -> Interpretation:
    """Are this fund's peers making largely the same bet?

    Reads the average pairwise cosine similarity already computed by
    `drift_score.compute_crowding_score` (1.0 = identical exposure
    shape). Uses the most recent date with data, and reports how many
    peers actually contributed that day (`n_peers_used`), since a score
    from 2 peers is weaker evidence than one from the full group.
    """
    if crowding_result.crowding.empty:
        return Interpretation(tag="unknown", headline="No crowding data available")

    latest = float(crowding_result.crowding.iloc[-1])
    n_used = int(crowding_result.n_peers_used.iloc[-1])
    n_total = len(crowding_result.peer_tickers)

    if latest >= _CROWDING_HIGH_THRESHOLD:
        tag = "crowded"
        headline = (
            f"This fund's peer group is highly overlapping ({latest:.0%} average pairwise "
            f"similarity) -- these funds are largely making the same factor bet, not offering "
            f"independent diversification from one another."
        )
    else:
        tag = "differentiated"
        headline = (
            f"This fund's peer group shows meaningfully different factor exposures "
            f"({latest:.0%} average pairwise similarity) -- less overlap than a tightly "
            f"crowded group."
        )
    detail = f"Based on {n_used} of {n_total} requested peers with valid data on the latest date."
    return Interpretation(tag=tag, headline=headline, detail=detail)


# ---------------------------------------------------------------------------
# Portfolio-level (src/portfolio.py)
# ---------------------------------------------------------------------------


def describe_portfolio_concentration(
    n_holdings: int, effective_n: float, avg_pairwise_similarity: float
) -> Interpretation:
    """Do N holdings actually provide N independent sources of diversification?

    `effective_n` (see `portfolio.compute_portfolio_diversification`) is
    a weight-and-similarity-adjusted count of how many effectively
    distinct factor bets the portfolio holds -- 1/(average pairwise
    similarity)-style shrinkage of the raw holding count, not a new
    statistical model.
    """
    if n_holdings <= 1:
        return Interpretation(
            tag="single_holding",
            headline="A single holding can't be evaluated for cross-fund diversification.",
        )

    ratio = effective_n / n_holdings
    if ratio >= 0.75:
        tag = "diversified"
        headline = (
            f"These {n_holdings} holdings behave like roughly {effective_n:.1f} independent "
            f"factor bets -- meaningful diversification across the group."
        )
    elif ratio >= 0.4:
        tag = "partially_overlapping"
        headline = (
            f"These {n_holdings} holdings behave like roughly {effective_n:.1f} independent "
            f"factor bets -- noticeable overlap, less diversification than the holding count "
            f"alone suggests."
        )
    else:
        tag = "concentrated"
        headline = (
            f"These {n_holdings} holdings behave like roughly {effective_n:.1f} independent "
            f"factor bets -- despite owning {n_holdings} funds, this portfolio is largely "
            f"exposed to one underlying source of risk."
        )
    detail = f"Average pairwise factor-exposure similarity across holdings: {avg_pairwise_similarity:.0%}."
    return Interpretation(tag=tag, headline=headline, detail=detail)
