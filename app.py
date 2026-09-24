"""Fund X-Ray -- Streamlit dashboard.

Wires together every module in `src/` behind a single sidebar (ticker,
date range, peer group, rolling window, regime count) into six
question-oriented tabs: Overview, What Drives This Fund?, Stress
Behavior, Style Drift, Diversification, and Detailed Quant Analysis
(where every raw estimator output -- rolling/Kalman beta paths,
standard errors, VIF, HMM validation/stability -- still lives, for a
technical reader). All network calls and model fits are wrapped in
`st.cache_data`/`st.cache_resource` so Streamlit's rerun-the-whole-
script-on-every-interaction model doesn't re-fetch or re-fit on every
widget change -- only when an actual input changes.

No local-only file paths: `DataLoader`'s parquet cache lives under
`data/cache/` (relative to the repo, created on demand), which works
unchanged on Streamlit Community Cloud's ephemeral filesystem. This
app needs no API keys / `st.secrets` -- every data source (yfinance,
the Kenneth French library) is free and unauthenticated.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.data_loader import DataLoader, FundDataset
from src.drift_score import (
    CrowdingScoreResult,
    build_peer_exposure_vectors,
    compute_crowding_score,
    compute_style_drift,
)
from src.interpretation import (
    classify_significance,
    describe_crowding,
    describe_dominant_factor,
    describe_fixed_beta_cross_check,
    describe_style_drift_with_latest,
)
from src.kalman_beta import TimeVaryingBetaResult, kalman_filter_betas, rolling_ols_betas
from src.regime_detection import (
    RegimeDetectionResult,
    build_regime_features,
    compare_hmm_to_rule_based,
    fit_regime_hmm,
    regime_episodes,
    regime_stability_across_seeds,
    rule_based_regime_labels,
    validate_against_known_stress_periods,
)
from src.report import build_fund_report
from src.risk_metrics import (
    OVERALL_LABEL,
    compute_fixed_beta_cross_check,
    compute_regime_risk_metrics,
    regime_comparison_table,
)
from src.style_analysis import (
    compare_style_results,
    compute_factor_vif,
    run_constrained_style_analysis,
    run_unconstrained_ols,
)
from src.viz_theme import BASELINE, ESTIMATOR_COLORS, FACTOR_COLORS, REGIME_COLORS, apply_theme

st.set_page_config(page_title="Fund X-Ray", page_icon="\U0001f4ca", layout="wide")

DISCLAIMER = (
    "Fund X-Ray is a risk-transparency tool. It reverse-engineers factor exposures and risk "
    "behavior from public returns -- it does not predict returns, and nothing here is "
    "investment advice or a recommendation to buy, hold, or sell any fund."
)


# ---------------------------------------------------------------------------
# Cached data / model layer
# ---------------------------------------------------------------------------
@st.cache_resource
def get_loader() -> DataLoader:
    return DataLoader()


@st.cache_data(show_spinner="Fetching fund and factor data...")
def load_fund_dataset(ticker: str, start: date, end: date) -> FundDataset:
    return get_loader().build_fund_dataset(ticker, start, end)


@st.cache_data(show_spinner="Fetching market regime inputs (SPY, VIX)...")
def load_market_regime_inputs(start: date, end: date) -> tuple[pd.Series, pd.Series]:
    loader = get_loader()
    prices = loader.get_prices(config.REGIME_MARKET_PROXY, start, end)
    spy_returns = loader.to_log_returns(prices)[config.REGIME_MARKET_PROXY]
    vix = loader.get_vix(start, end)
    return spy_returns, vix


@st.cache_data(show_spinner="Fitting regime-detection HMM...")
def compute_regimes(spy_returns: pd.Series, vix: pd.Series, n_regimes: int) -> RegimeDetectionResult:
    return fit_regime_hmm(spy_returns, vix=vix, n_regimes=n_regimes)


@st.cache_data(show_spinner="Checking regime-split stability across random seeds...")
def compute_regime_stability(spy_returns: pd.Series, vix: pd.Series, n_regimes: int) -> float:
    return regime_stability_across_seeds(spy_returns, vix, n_regimes=n_regimes)


@st.cache_data(show_spinner="Fitting rolling-window and Kalman-filter betas...")
def compute_time_varying_betas(
    excess_returns: pd.Series, factor_returns: pd.DataFrame, window: int
) -> tuple[TimeVaryingBetaResult, TimeVaryingBetaResult]:
    rolling = rolling_ols_betas(excess_returns, factor_returns, window=window)
    kalman = kalman_filter_betas(excess_returns, factor_returns)
    return rolling, kalman


@st.cache_data(show_spinner="Building peer exposure vectors for the crowding score...")
def load_crowding_score(
    peer_tickers: tuple[str, ...], start: date, end: date, window: int
) -> tuple[CrowdingScoreResult | None, list[str]]:
    loader = get_loader()
    peer_betas, skipped = build_peer_exposure_vectors(
        loader, list(peer_tickers), start, end, window=window
    )
    if len(peer_betas) < config.MIN_PEERS_FOR_CROWDING:
        return None, skipped
    return compute_crowding_score(peer_betas, skipped_tickers=skipped), skipped


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Fund X-Ray")
st.sidebar.caption("Returns-based style analysis & regime-conditional risk attribution")

ticker = st.sidebar.text_input("Fund / ETF ticker", value=config.DEFAULT_FUND_TICKER).strip().upper()

date_col1, date_col2 = st.sidebar.columns(2)
start_date = date_col1.date_input("Start date", value=config.DEFAULT_START_DATE)
end_date = date_col2.date_input("End date", value=config.DEFAULT_END_DATE)

rolling_window = st.sidebar.select_slider(
    "Rolling window (trading days)",
    options=sorted(config.ROLLING_WINDOWS),
    value=min(config.ROLLING_WINDOWS),
)
n_regimes = st.sidebar.radio(
    "Number of regimes", config.HMM_N_REGIMES_OPTIONS, index=0, horizontal=True
)

st.sidebar.divider()
st.sidebar.subheader("Peer group (Diversification)")
peer_group_text = st.sidebar.text_area(
    "Comma-separated tickers",
    value=", ".join(config.DEFAULT_PEER_GROUP),
    height=70,
)
peer_group = [t.strip().upper() for t in peer_group_text.split(",") if t.strip()]

st.sidebar.divider()
st.sidebar.caption(
    "Data: yfinance (prices), Kenneth French data library (Fama-French factors). "
    "Cached locally as parquet; no API keys required."
)
st.sidebar.caption(DISCLAIMER)

if not ticker:
    st.info("Enter a fund or ETF ticker in the sidebar to begin.")
    st.stop()
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

st.title(f"Fund X-Ray: {ticker}")

try:
    dataset = load_fund_dataset(ticker, start_date, end_date)
except ValueError as exc:
    st.error(f"Couldn't load data for '{ticker}': {exc}")
    st.stop()

st.caption(
    f"{len(dataset.fund_returns):,} trading days, "
    f"{dataset.start.date()} to {dataset.end.date()}."
)

# ---------------------------------------------------------------------------
# Core computations shared across tabs (cheap, pure -- not cache_data'd,
# same convention as the original static-analysis calls: only network
# calls and iterative model fits (HMM, Kalman) are cached above).
# ---------------------------------------------------------------------------
constrained = run_constrained_style_analysis(dataset.excess_returns, dataset.factor_returns)
unconstrained = run_unconstrained_ols(dataset.excess_returns, dataset.factor_returns)
comparison = compare_style_results(constrained, unconstrained)
vif = compute_factor_vif(dataset.factor_returns)

rolling_result, kalman_result = compute_time_varying_betas(
    dataset.excess_returns, dataset.factor_returns, rolling_window
)

spy_returns, vix = load_market_regime_inputs(start_date, end_date)
regime_features, vol_source = build_regime_features(spy_returns, vix)
regime_result = compute_regimes(spy_returns, vix, n_regimes)

risk_metrics_by_regime = compute_regime_risk_metrics(
    dataset.fund_returns, dataset.excess_returns, dataset.factor_returns, regime_result.regime_labels
)

default_drift_result = compute_style_drift(rolling_result.betas, metric="cosine")
latest_exposure = rolling_result.betas.dropna().iloc[-1]

fund_report = build_fund_report(
    dataset=dataset,
    unconstrained=unconstrained,
    risk_metrics_by_regime=risk_metrics_by_regime,
    drift_result=default_drift_result,
    latest_exposure=latest_exposure,
    regime_labels=regime_result.regime_labels,
)

tab_overview, tab_drivers, tab_stress, tab_drift, tab_diversification, tab_detail = st.tabs(
    [
        "Overview",
        "What Drives This Fund?",
        "Stress Behavior",
        "Style Drift",
        "Diversification",
        "Detailed Quant Analysis",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1: Overview -- the Fund X-Ray Summary report
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader(f"{ticker}: {fund_report.n_obs:,} trading days, {fund_report.period_start.date()} to {fund_report.period_end.date()}")
    st.info(DISCLAIMER, icon="ℹ️")

    st.markdown("#### What drives this fund")
    st.markdown(f"**{fund_report.dominant_factor.headline}**")
    st.caption(fund_report.dominant_factor.detail)

    if fund_report.calm_vs_stressed_table is not None:
        st.markdown("#### Normal vs. stressed markets")
        st.markdown(f"**{fund_report.stress_interpretation.headline}**")
        st.caption(fund_report.stress_interpretation.detail)
        st.dataframe(
            fund_report.calm_vs_stressed_table.style.format(
                {
                    "n_obs": "{:.0f}",
                    "annualized_vol": "{:.1%}",
                    "VaR_95": "{:.2%}",
                    "CVaR_95": "{:.2%}",
                    "R_squared": "{:.1%}",
                    "idiosyncratic_variance_share": "{:.1%}",
                    "idiosyncratic_annualized_vol": "{:.1%}",
                    "factor_driven_annualized_vol": "{:.1%}",
                    **{c: "{:.3f}" for c in fund_report.calm_vs_stressed_table.columns if c.startswith("beta_")},
                }
            ),
            use_container_width=True,
        )
        if fund_report.fixed_beta_check is not None:
            st.caption(f"Robustness check: {fund_report.fixed_beta_check.headline}")
    else:
        st.caption("Not enough distinct regimes detected in this date range for a calm-vs-stressed comparison.")

    st.markdown("#### Style stability")
    st.markdown(f"**{fund_report.style_drift.headline}**")
    if fund_report.style_drift.detail:
        st.caption(fund_report.style_drift.detail)

    st.caption(
        "See the 'Diversification' tab for how this fund compares to a peer group, and "
        "'Detailed Quant Analysis' for the full statistical output behind every number above."
    )

# ---------------------------------------------------------------------------
# Tab 2: What drives this fund
# ---------------------------------------------------------------------------
with tab_drivers:
    st.subheader("What drives this fund's risk?")

    overall_metrics = risk_metrics_by_regime[OVERALL_LABEL]
    dominant = describe_dominant_factor(overall_metrics.factor_risk_contribution_pct)
    st.markdown(f"**{dominant.headline}**")
    st.caption(dominant.detail)

    contribution = overall_metrics.factor_risk_contribution_pct.copy()
    contribution["Idiosyncratic (fund-specific)"] = overall_metrics.idiosyncratic_variance_share
    contribution = contribution.sort_values()
    bar_colors = [FACTOR_COLORS.get(f, BASELINE) for f in contribution.index]

    fig = go.Figure(
        go.Bar(
            x=contribution.values,
            y=contribution.index,
            orientation="h",
            marker_color=bar_colors,
        )
    )
    fig.update_layout(
        title="Share of total risk by source (whole-sample)",
        xaxis_title="Share of total variance",
        xaxis_tickformat=".0%",
    )
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Style: constrained vs. unconstrained")
    st.caption(
        "Two estimates of the same style regression, compared: a bounded (Sharpe 1992) fit and "
        "an unconstrained fit with honest uncertainty. Where they disagree is informative -- see "
        "LIMITATIONS.md for why the bounded estimate is a stylized tilt, not a holdings breakdown."
    )

    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            name="Constrained",
            x=comparison.index,
            y=comparison["constrained_weight"],
            marker_color=ESTIMATOR_COLORS["constrained"],
        )
    )
    fig2.add_trace(
        go.Bar(
            name="Unconstrained",
            x=comparison.index,
            y=comparison["unconstrained_beta"],
            marker_color=ESTIMATOR_COLORS["unconstrained"],
            error_y=dict(type="data", array=1.96 * unconstrained.standard_errors.reindex(comparison.index)),
        )
    )
    fig2.update_layout(
        barmode="group",
        title="Factor weight / beta by estimator (unconstrained bars show a 95% CI)",
        yaxis_title="Weight / beta",
    )
    apply_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    significance_col = unconstrained.p_values.reindex(comparison.index).apply(
        lambda p: classify_significance(p).tag
    )
    display_table = comparison.copy()
    display_table["significance"] = significance_col
    display_table["VIF"] = vif.reindex(comparison.index)
    st.dataframe(
        display_table.style.format(
            {"constrained_weight": "{:.3f}", "unconstrained_beta": "{:.3f}", "difference": "{:+.3f}", "VIF": "{:.1f}"}
        ),
        use_container_width=True,
    )
    st.caption(
        "`significance`: whether the unconstrained beta is statistically distinguishable from "
        f"zero at p<{config.SIGNIFICANCE_ALPHA:g} (Newey-West HAC standard errors). `VIF` above "
        f"{config.VIF_HIGH_THRESHOLD:g} flags a factor whose individual beta is unreliable due "
        "to collinearity with the other factors -- the joint fit may still be fine even when an "
        "individual coefficient isn't."
    )

# ---------------------------------------------------------------------------
# Tab 3: Stress behavior
# ---------------------------------------------------------------------------
with tab_stress:
    st.subheader("How does this fund behave when markets are stressed?")

    if fund_report.stress_interpretation is not None:
        st.markdown(f"**{fund_report.stress_interpretation.headline}**")
        st.caption(fund_report.stress_interpretation.detail)
    else:
        st.info("This date range didn't produce distinct calm/stressed regimes to compare.")

    cumulative_return = np.exp(dataset.fund_returns.cumsum()) * 100
    episodes = regime_episodes(regime_result.regime_labels)

    fig = go.Figure()
    for _, ep in episodes.iterrows():
        fig.add_vrect(
            x0=ep["start"],
            x1=ep["end"],
            fillcolor=REGIME_COLORS.get(ep["label"], "#c3c2b7"),
            opacity=0.15,
            line_width=0,
        )
    fig.add_trace(
        go.Scatter(
            x=cumulative_return.index,
            y=cumulative_return,
            name=ticker,
            mode="lines",
            line=dict(color=FACTOR_COLORS["Mkt-RF"], width=2),
        )
    )
    fig.update_layout(
        title=f"{ticker} cumulative return (indexed to 100), shaded by market regime ({n_regimes}-regime, independent of {ticker})",
        yaxis_title="Cumulative return (indexed to 100)",
    )
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    legend_line = "  ".join(
        f"<span style='color:{color}'>■</span> {label}"
        for label, color in REGIME_COLORS.items()
        if label in regime_result.regime_labels.unique()
    )
    st.markdown(legend_line, unsafe_allow_html=True)

    st.markdown("#### Calm vs. stressed, side by side")
    table = regime_comparison_table(risk_metrics_by_regime)
    st.dataframe(
        table.style.format(
            {
                "n_obs": "{:.0f}",
                "annualized_vol": "{:.1%}",
                "annualized_mean_return": "{:.1%}",
                "VaR_95": "{:.2%}",
                "CVaR_95": "{:.2%}",
                "R_squared": "{:.1%}",
                "idiosyncratic_variance_share": "{:.1%}",
                "idiosyncratic_annualized_vol": "{:.1%}",
                "factor_driven_annualized_vol": "{:.1%}",
                **{c: "{:.3f}" for c in table.columns if c.startswith("beta_")},
            }
        ),
        use_container_width=True,
    )
    st.caption(
        "`idiosyncratic_annualized_vol` / `factor_driven_annualized_vol` split the fund's total "
        "volatility in absolute terms, not just as a share -- useful for telling apart 'my "
        "fund-specific risk shrank' from 'market-driven risk grew faster than my fund-specific "
        "risk stayed the same.'"
    )

    if "calm" in risk_metrics_by_regime and "stressed" in risk_metrics_by_regime:
        cross_check = compute_fixed_beta_cross_check(
            dataset.fund_returns,
            dataset.excess_returns,
            dataset.factor_returns,
            regime_result.regime_labels,
            reference_regime="calm",
            target_regime="stressed",
        )
        cross_check_interp = describe_fixed_beta_cross_check(cross_check)
        with st.expander("Robustness check: is this an artifact of refitting per regime?"):
            st.markdown(f"**{cross_check_interp.headline}**")
            st.caption(cross_check_interp.detail)
            st.caption(
                "Each regime's exposure is independently refit on that regime's own data, which "
                "gives the model extra freedom that can inflate an R^2 difference on its own. "
                "This check applies the calm regime's exposure, UN-REFIT, to the stressed "
                "regime's data, to see whether the finding survives without that extra freedom."
            )

# ---------------------------------------------------------------------------
# Tab 4: Style drift
# ---------------------------------------------------------------------------
with tab_drift:
    st.subheader("Is this fund's style changing over time?")

    estimator_choice = st.radio(
        "Exposure source", ["Rolling OLS", "Kalman filter"], index=0, horizontal=True
    )
    metric_choice = st.radio("Distance metric", ["cosine", "euclidean"], index=0, horizontal=True)
    threshold_method = st.radio(
        "Drift threshold",
        ["gaussian (mean + 2σ)", "percentile (95th)"],
        index=0,
        horizontal=True,
    )
    threshold_method_key = "gaussian" if threshold_method.startswith("gaussian") else "percentile"

    betas_for_drift = rolling_result.betas if estimator_choice == "Rolling OLS" else kalman_result.betas
    drift_result = compute_style_drift(
        betas_for_drift, metric=metric_choice, threshold_method=threshold_method_key
    )
    latest_for_drift = betas_for_drift.dropna().iloc[-1]
    drift_interp = describe_style_drift_with_latest(drift_result, latest_for_drift)

    st.markdown(f"**{drift_interp.headline}**")
    if drift_interp.detail:
        st.caption(drift_interp.detail)
    if abs(drift_result.baseline_skew) > 1.0:
        st.caption(
            f"Note: the baseline-period drift distribution is notably skewed "
            f"(skew={drift_result.baseline_skew:.2f}) -- consider the percentile threshold "
            "above as a distribution-free alternative to the default gaussian one."
        )

    fig = go.Figure()
    for _, ev in drift_result.flagged_events.iterrows():
        fig.add_vrect(x0=ev["start"], x1=ev["end"], fillcolor=REGIME_COLORS["stressed"], opacity=0.12, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=drift_result.drift.index,
            y=drift_result.drift,
            name="Drift",
            mode="lines",
            line=dict(color=FACTOR_COLORS["Mkt-RF"], width=2),
        )
    )
    fig.add_hline(
        y=drift_result.threshold,
        line_dash="dash",
        line_color=REGIME_COLORS["stressed"],
        annotation_text=f"threshold ({threshold_method_key})",
    )
    fig.update_layout(
        title=f"Distance from {config.STYLE_DRIFT_BASELINE_MONTHS}-month baseline exposure ({metric_choice})",
        yaxis_title="Drift score",
    )
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Baseline exposure vector")
    st.dataframe(drift_result.baseline_vector.rename("weight").to_frame().T, use_container_width=True)

    st.markdown("#### Flagged drift episodes")
    if drift_result.flagged_events.empty:
        st.caption("No drift episodes exceeded the threshold.")
    else:
        st.dataframe(drift_result.flagged_events, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5: Diversification (Crowding Score)
# ---------------------------------------------------------------------------
with tab_diversification:
    st.subheader("Is this fund's peer group actually diversifying?")

    if len(peer_group) < config.MIN_PEERS_FOR_CROWDING:
        st.info(f"Add at least {config.MIN_PEERS_FOR_CROWDING} peer tickers in the sidebar.")
    else:
        crowding_result, skipped = load_crowding_score(
            tuple(peer_group), start_date, end_date, rolling_window
        )
        if skipped:
            st.warning(f"Skipped peer ticker(s) (couldn't fetch data): {', '.join(skipped)}")

        if crowding_result is None:
            st.error(
                f"Fewer than {config.MIN_PEERS_FOR_CROWDING} peers had usable data -- "
                "can't compute a crowding score."
            )
        else:
            crowding_interp = describe_crowding(crowding_result)
            st.markdown(f"**{crowding_interp.headline}**")
            st.caption(crowding_interp.detail)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=crowding_result.crowding.index,
                    y=crowding_result.crowding,
                    name="Crowding",
                    mode="lines",
                    line=dict(color=FACTOR_COLORS["Mkt-RF"], width=2),
                )
            )
            fig.update_layout(
                title=f"Average pairwise cosine similarity across {', '.join(crowding_result.peer_tickers)}",
                yaxis_title="Avg. pairwise cosine similarity",
                yaxis_range=[0, 1],
            )
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                f"Peers used: {', '.join(crowding_result.peer_tickers)}. "
                f"Peers/date ranges from {crowding_result.n_peers_used.min()} to "
                f"{crowding_result.n_peers_used.max()} (of {len(crowding_result.peer_tickers)} "
                "requested) depending on data availability that day."
            )
            st.caption(
                "Want to check several holdings at once, weighted as a portfolio? See the "
                "'Portfolio X-Ray' page in the sidebar navigation."
            )

# ---------------------------------------------------------------------------
# Tab 6: Detailed quant analysis
# ---------------------------------------------------------------------------
with tab_detail:
    st.subheader("Detailed quantitative analysis")
    st.caption(
        "Full statistical output behind every summary above: time-varying beta paths, raw "
        "regression diagnostics, and regime-model validation/robustness checks."
    )

    with st.expander("Time-varying exposure: rolling OLS vs. Kalman filter", expanded=True):
        factor_choice = st.selectbox("Factor", config.FACTOR_COLUMNS, index=0)
        show_rolling = st.checkbox(f"Show rolling ({rolling_window}-day)", value=True)
        show_kalman = st.checkbox("Show Kalman filter", value=True)

        fig = go.Figure()
        if show_rolling:
            fig.add_trace(
                go.Scatter(
                    x=rolling_result.betas.index,
                    y=rolling_result.betas[factor_choice],
                    name=f"Rolling ({rolling_window}d)",
                    mode="lines",
                    line=dict(color=ESTIMATOR_COLORS["rolling"], width=2),
                )
            )
        if show_kalman:
            fig.add_trace(
                go.Scatter(
                    x=kalman_result.betas.index,
                    y=kalman_result.betas[factor_choice],
                    name="Kalman filter",
                    mode="lines",
                    line=dict(color=ESTIMATOR_COLORS["kalman"], width=2),
                )
            )
        fig.update_layout(title=f"{factor_choice} beta over time", yaxis_title="Beta")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Rolling OLS drops its first {rolling_result.n_burn_in} observations (burn-in for a "
            f"full window); the Kalman filter drops its first {kalman_result.n_burn_in} "
            "(warm-up used only to seed the filter's initial state). Neither burn-in period is "
            "backfilled. See README for the delta=0.02 calibration story."
        )

    with st.expander("Regime model: validation, stability, and a rule-based baseline"):
        st.markdown("**Validation against known historical stress windows**")
        st.dataframe(validate_against_known_stress_periods(regime_result), use_container_width=True)
        st.caption(
            "Percent of each known historical stress window's trading days labeled 'stressed' "
            "by this model -- computed only as a post-fit sanity check, never used to fit the HMM."
        )

        st.markdown("**Regime posterior probabilities (most recent 10 days)**")
        st.dataframe(regime_result.regime_probabilities.tail(10), use_container_width=True)

        st.markdown("**Stability across random initializations**")
        stability = compute_regime_stability(spy_returns, vix, n_regimes)
        st.metric("Day-label agreement across 5 seeds", f"{stability:.0%}")
        st.caption(
            "How often two independent HMM refits (different random initializations) assign the "
            "same day the same label. Low agreement would mean the calm/stressed split depends "
            "on the EM algorithm's starting point, not just the data."
        )

        st.markdown("**Rule-based baseline (no model fitting)**")
        rule_labels = rule_based_regime_labels(regime_features["vol_signal"], threshold_percentile=80.0)
        agreement = compare_hmm_to_rule_based(regime_result.regime_labels, rule_labels)
        st.metric("HMM vs. simple percentile-threshold rule agreement", f"{agreement:.0%}")
        st.caption(
            f"'Stressed' = {vol_source} above its own trailing 80th percentile, with no model "
            "fitting at all. High agreement means the HMM's calm/stressed story isn't an "
            "artifact of that specific model choice."
        )

    with st.expander("Full style regression output (standard errors, t-stats, p-values, VIF)"):
        detail_table = pd.DataFrame(
            {
                "constrained_weight": constrained.weights,
                "unconstrained_beta": unconstrained.weights,
                "standard_error": unconstrained.standard_errors,
                "t_stat": unconstrained.t_stats,
                "p_value": unconstrained.p_values,
                "VIF": vif,
                "at_boundary": comparison["at_boundary"],
            }
        )
        st.dataframe(
            detail_table.style.format(
                {
                    "constrained_weight": "{:.3f}",
                    "unconstrained_beta": "{:.3f}",
                    "standard_error": "{:.4f}",
                    "t_stat": "{:.2f}",
                    "p_value": "{:.3f}",
                    "VIF": "{:.1f}",
                }
            ),
            use_container_width=True,
        )
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Constrained R²", f"{constrained.r_squared:.1%}")
        metric_col2.metric("Unconstrained R²", f"{unconstrained.r_squared:.1%}")
        metric_col3.metric(
            "Annualized alpha (unconstrained)",
            f"{unconstrained.alpha * config.TRADING_DAYS_PER_YEAR:+.2%}",
        )
        st.caption(
            "`at_boundary` flags factors where the constrained weight sits at (or near) 0 or 1 -- "
            "the clearest sign the simplex constraint, not the data, is driving that coefficient. "
            "See LIMITATIONS.md for why Sharpe's simplex constraint is a stylized adaptation here, "
            "not a literal asset-allocation read."
        )
