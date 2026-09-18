"""Fund X-Ray -- Streamlit dashboard.

Wires together every module in `src/` behind a single sidebar (ticker,
date range, peer group, rolling window, regime count). All network
calls and model fits are wrapped in `st.cache_data`/`st.cache_resource`
so Streamlit's rerun-the-whole-script-on-every-interaction model
doesn't re-fetch or re-fit on every widget change -- only when an
actual input changes.

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
from src.data_loader import DataLoader
from src.drift_score import (
    build_peer_exposure_vectors,
    compute_crowding_score,
    compute_style_drift,
)
from src.kalman_beta import kalman_filter_betas, rolling_ols_betas
from src.regime_detection import (
    fit_regime_hmm,
    regime_episodes,
    validate_against_known_stress_periods,
)
from src.risk_metrics import compute_regime_risk_metrics, regime_comparison_table
from src.style_analysis import (
    compare_style_results,
    run_constrained_style_analysis,
    run_unconstrained_ols,
)
from src.viz_theme import ESTIMATOR_COLORS, FACTOR_COLORS, REGIME_COLORS, apply_theme

st.set_page_config(page_title="Fund X-Ray", page_icon="\U0001f4ca", layout="wide")


# ---------------------------------------------------------------------------
# Cached data / model layer
# ---------------------------------------------------------------------------
@st.cache_resource
def get_loader() -> DataLoader:
    return DataLoader()


@st.cache_data(show_spinner="Fetching fund and factor data...")
def load_fund_dataset(ticker: str, start: date, end: date):
    return get_loader().build_fund_dataset(ticker, start, end)


@st.cache_data(show_spinner="Fetching market regime inputs (SPY, VIX)...")
def load_market_regime_inputs(start: date, end: date):
    loader = get_loader()
    prices = loader.get_prices(config.REGIME_MARKET_PROXY, start, end)
    spy_returns = loader.to_log_returns(prices)[config.REGIME_MARKET_PROXY]
    vix = loader.get_vix(start, end)
    return spy_returns, vix


@st.cache_data(show_spinner="Fitting regime-detection HMM...")
def compute_regimes(spy_returns: pd.Series, vix: pd.Series, n_regimes: int):
    return fit_regime_hmm(spy_returns, vix=vix, n_regimes=n_regimes)


@st.cache_data(show_spinner="Fitting rolling-window and Kalman-filter betas...")
def compute_time_varying_betas(excess_returns: pd.Series, factor_returns: pd.DataFrame, window: int):
    rolling = rolling_ols_betas(excess_returns, factor_returns, window=window)
    kalman = kalman_filter_betas(excess_returns, factor_returns)
    return rolling, kalman


@st.cache_data(show_spinner="Building peer exposure vectors for the crowding score...")
def load_crowding_score(peer_tickers: tuple[str, ...], start: date, end: date, window: int):
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
st.sidebar.subheader("Peer group (Crowding Score)")
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

tab_static, tab_dynamic, tab_regime, tab_drift, tab_crowding = st.tabs(
    ["Static Exposures", "Time-Varying Exposures", "Regime View", "Style Drift", "Crowding Score"]
)

# ---------------------------------------------------------------------------
# Tab 1: Static exposures
# ---------------------------------------------------------------------------
with tab_static:
    st.subheader("Constrained (Sharpe 1992) vs. unconstrained factor exposures")

    constrained = run_constrained_style_analysis(dataset.excess_returns, dataset.factor_returns)
    unconstrained = run_unconstrained_ols(dataset.excess_returns, dataset.factor_returns)
    comparison = compare_style_results(constrained, unconstrained)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Constrained R²", f"{constrained.r_squared:.1%}")
    metric_col2.metric("Unconstrained R²", f"{unconstrained.r_squared:.1%}")
    metric_col3.metric(
        "Annualized alpha (unconstrained)",
        f"{unconstrained.alpha * config.TRADING_DAYS_PER_YEAR:+.2%}",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Constrained",
            x=comparison.index,
            y=comparison["constrained_weight"],
            marker_color=ESTIMATOR_COLORS["constrained"],
        )
    )
    fig.add_trace(
        go.Bar(
            name="Unconstrained",
            x=comparison.index,
            y=comparison["unconstrained_beta"],
            marker_color=ESTIMATOR_COLORS["unconstrained"],
        )
    )
    fig.update_layout(barmode="group", title="Factor weight / beta by estimator", yaxis_title="Weight / beta")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        comparison.style.format(
            {"constrained_weight": "{:.3f}", "unconstrained_beta": "{:.3f}", "difference": "{:+.3f}"}
        ),
        use_container_width=True,
    )
    st.caption(
        "`at_boundary` flags factors where the constrained weight sits at (or near) 0 or 1 -- "
        "the clearest sign the simplex constraint, not the data, is driving that coefficient. "
        "See LIMITATIONS.md for why Sharpe's simplex constraint is a stylized adaptation here, "
        "not a literal asset-allocation read."
    )

# ---------------------------------------------------------------------------
# Tab 2: Time-varying exposures
# ---------------------------------------------------------------------------
with tab_dynamic:
    st.subheader("Rolling-window OLS vs. Kalman filter")

    rolling_result, kalman_result = compute_time_varying_betas(
        dataset.excess_returns, dataset.factor_returns, rolling_window
    )

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
        "backfilled. See README for the delta=0.02 calibration story and a documented example "
        "of the Kalman filter reacting faster than the rolling window around the 2020 COVID crash."
    )

# ---------------------------------------------------------------------------
# Tab 3: Regime view
# ---------------------------------------------------------------------------
with tab_regime:
    st.subheader(f"{n_regimes}-regime market view (independent of {ticker})")

    spy_returns, vix = load_market_regime_inputs(start_date, end_date)
    regime_result = compute_regimes(spy_returns, vix, n_regimes)

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
        title=f"{ticker} cumulative return (indexed to 100), shaded by market regime",
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

    st.markdown("#### Regime-conditional risk metrics")
    risk_metrics = compute_regime_risk_metrics(
        dataset.fund_returns, dataset.excess_returns, dataset.factor_returns, regime_result.regime_labels
    )
    table = regime_comparison_table(risk_metrics)
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
                **{c: "{:.3f}" for c in table.columns if c.startswith("beta_")},
            }
        ),
        use_container_width=True,
    )

    with st.expander("Validation: does this line up with known stress periods?"):
        st.dataframe(validate_against_known_stress_periods(regime_result), use_container_width=True)
        st.caption(
            "Percent of each known historical stress window's trading days labeled 'stressed' "
            "by this model -- computed only as a post-fit sanity check, never used to fit the HMM."
        )

# ---------------------------------------------------------------------------
# Tab 4: Style drift
# ---------------------------------------------------------------------------
with tab_drift:
    st.subheader("Style Drift Score")

    estimator_choice = st.radio(
        "Exposure source", ["Rolling OLS", "Kalman filter"], index=0, horizontal=True
    )
    metric_choice = st.radio("Distance metric", ["cosine", "euclidean"], index=0, horizontal=True)

    betas_for_drift = rolling_result.betas if estimator_choice == "Rolling OLS" else kalman_result.betas
    drift_result = compute_style_drift(betas_for_drift, metric=metric_choice)

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
        annotation_text="threshold (baseline mean + 2 std)",
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
    st.caption(
        "A long, uninterrupted flagged episode means the fund's exposure has permanently "
        "diverged from its own baseline period, not that it's an ongoing anomaly expected to "
        "revert -- see LIMITATIONS.md."
    )

# ---------------------------------------------------------------------------
# Tab 5: Crowding score
# ---------------------------------------------------------------------------
with tab_crowding:
    st.subheader("Crowding Score (peer group positioning overlap)")

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
