"""Fund vs. Fund -- side-by-side comparison of 2-3 tickers.

No new modeling: reuses the same single-fund pipeline as the main page
(`build_fund_dataset` -> `run_unconstrained_ols` -> regime detection ->
`compute_regime_risk_metrics` -> `compute_style_drift` ->
`report.build_fund_report`) for each ticker and lays the results out
side by side.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import DataLoader, FundDataset
from src.drift_score import compute_style_drift
from src.kalman_beta import rolling_ols_betas
from src.regime_detection import RegimeDetectionResult, fit_regime_hmm
from src.report import FundReport, build_fund_report
from src.risk_metrics import compute_regime_risk_metrics
from src.style_analysis import run_unconstrained_ols

st.set_page_config(page_title="Fund X-Ray - Fund vs. Fund", page_icon="\U0001f4ca", layout="wide")

DISCLAIMER = (
    "This is a risk-transparency comparison, not investment advice. It does not predict returns "
    "and does not identify a 'best' fund."
)


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


def build_report_for_ticker(
    ticker: str, start: date, end: date, regime_labels: pd.Series, rolling_window: int
) -> FundReport | str:
    """Returns a FundReport, or an error message string if the ticker couldn't be loaded."""
    try:
        dataset = load_fund_dataset(ticker, start, end)
    except ValueError as exc:
        return f"Couldn't load '{ticker}': {exc}"

    unconstrained = run_unconstrained_ols(dataset.excess_returns, dataset.factor_returns)
    risk_metrics = compute_regime_risk_metrics(
        dataset.fund_returns, dataset.excess_returns, dataset.factor_returns, regime_labels
    )
    rolling = rolling_ols_betas(dataset.excess_returns, dataset.factor_returns, window=rolling_window)
    drift = compute_style_drift(rolling.betas, metric="cosine")
    latest_exposure = rolling.betas.dropna().iloc[-1]

    return build_fund_report(
        dataset=dataset,
        unconstrained=unconstrained,
        risk_metrics_by_regime=risk_metrics,
        drift_result=drift,
        latest_exposure=latest_exposure,
        regime_labels=regime_labels,
    )


st.sidebar.title("Fund vs. Fund")
st.sidebar.caption("Compare 2-3 tickers side by side.")

date_col1, date_col2 = st.sidebar.columns(2)
start_date = date_col1.date_input("Start date", value=config.DEFAULT_START_DATE, key="compare_start")
end_date = date_col2.date_input("End date", value=config.DEFAULT_END_DATE, key="compare_end")
rolling_window = st.sidebar.select_slider(
    "Rolling window (trading days)", options=sorted(config.ROLLING_WINDOWS), value=min(config.ROLLING_WINDOWS)
)
n_regimes = st.sidebar.radio("Number of regimes", config.HMM_N_REGIMES_OPTIONS, index=0, horizontal=True)
st.sidebar.caption(DISCLAIMER)

st.title("Fund vs. Fund")

ticker_input = st.text_input(
    "Tickers to compare (comma-separated, 2-3)", value="QQQ, VUG, SPY"
)
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()][:3]

if len(tickers) < 2:
    st.info("Enter at least 2 tickers to compare.")
    st.stop()
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

spy_returns, vix = load_market_regime_inputs(start_date, end_date)
regime_result = compute_regimes(spy_returns, vix, n_regimes)

reports: dict[str, FundReport] = {}
for ticker in tickers:
    result = build_report_for_ticker(
        ticker, start_date, end_date, regime_result.regime_labels, rolling_window
    )
    if isinstance(result, str):
        st.error(result)
    else:
        reports[ticker] = result

if len(reports) < 2:
    st.stop()

columns = st.columns(len(reports))
for col, (ticker, report) in zip(columns, reports.items(), strict=True):
    with col:
        st.markdown(f"### {ticker}")
        st.caption(f"{report.n_obs:,} trading days, {report.period_start.date()} to {report.period_end.date()}")

        st.markdown("**What drives it**")
        st.caption(report.dominant_factor.headline)

        if report.calm_vs_stressed_table is not None:
            row_calm = report.calm_vs_stressed_table.loc["calm"]
            row_stressed = report.calm_vs_stressed_table.loc["stressed"]
            st.markdown("**Stress behavior**")
            st.metric("VaR 95% (stressed)", f"{row_stressed['VaR_95']:.2%}", delta=f"{row_stressed['VaR_95'] - row_calm['VaR_95']:+.2%} vs. calm")
            st.metric("Idiosyncratic share (stressed)", f"{row_stressed['idiosyncratic_variance_share']:.1%}")
        else:
            st.caption("No distinct calm/stressed comparison available for this range.")

        st.markdown("**Style drift**")
        st.caption(report.style_drift.headline)

st.markdown("#### Side-by-side table")
summary_rows = {}
for ticker, report in reports.items():
    row = {"dominant_factor": report.dominant_factor.headline, "style_drift": report.style_drift.tag}
    if report.calm_vs_stressed_table is not None:
        stressed = report.calm_vs_stressed_table.loc["stressed"]
        row.update(
            {
                "stressed_VaR_95": stressed["VaR_95"],
                "stressed_idiosyncratic_share": stressed["idiosyncratic_variance_share"],
            }
        )
    summary_rows[ticker] = row

summary_table = pd.DataFrame(summary_rows).T
st.dataframe(
    summary_table.style.format(
        {"stressed_VaR_95": "{:.2%}", "stressed_idiosyncratic_share": "{:.1%}"}, na_rep="n/a"
    ),
    use_container_width=True,
)
