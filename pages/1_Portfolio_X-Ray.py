"""Portfolio X-Ray -- multi-fund exposure aggregation.

Answers a question the single-fund page can't: "although I own several
funds, am I getting several sources of diversification, or largely one?"
Built on `src/portfolio.py`, which reuses `DataLoader` and the existing
unconstrained style regression per holding -- no new modeling, just a
weight-weighted aggregation across holdings.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.data_loader import DataLoader
from src.interpretation import describe_portfolio_concentration
from src.portfolio import (
    PortfolioHolding,
    build_portfolio_betas,
    build_portfolio_datasets,
    compute_blended_exposure,
    compute_portfolio_diversification,
    normalize_weights,
)
from src.viz_theme import BASELINE, FACTOR_COLORS, apply_theme

st.set_page_config(page_title="Fund X-Ray - Portfolio", page_icon="\U0001f4ca", layout="wide")

DISCLAIMER = (
    "This is a risk-transparency view, not investment advice. It does not predict returns and "
    "does not recommend any allocation."
)


@st.cache_resource
def get_loader() -> DataLoader:
    return DataLoader()


@st.cache_data(show_spinner="Fetching each holding's data and fitting factor exposures...")
def load_portfolio(
    tickers: tuple[str, ...],
    weights: tuple[float, ...],
    start,
    end,
) -> tuple[dict[str, pd.Series], list[str], dict[str, int]]:
    loader = get_loader()
    holdings = [PortfolioHolding(t, w) for t, w in zip(tickers, weights, strict=True)]
    datasets, skipped = build_portfolio_datasets(loader, holdings, start, end)
    betas = build_portfolio_betas(datasets)
    n_obs = {ticker: len(ds.fund_returns) for ticker, ds in datasets.items()}
    return betas, skipped, n_obs


st.sidebar.title("Portfolio X-Ray")
st.sidebar.caption("Are your holdings giving you independent bets, or the same one repeated?")

date_col1, date_col2 = st.sidebar.columns(2)
start_date = date_col1.date_input(
    "Start date",
    value=config.DEFAULT_START_DATE,
    min_value=config.MIN_SELECTABLE_DATE,
    max_value=date.today(),
    key="portfolio_start",
)
end_date = date_col2.date_input(
    "End date",
    value=config.DEFAULT_END_DATE,
    min_value=config.MIN_SELECTABLE_DATE,
    max_value=date.today(),
    key="portfolio_end",
)

st.sidebar.caption(DISCLAIMER)

st.title("Portfolio X-Ray")
st.caption("Enter your holdings and their weights (any consistent unit -- dollars, shares, %).")

default_holdings = pd.DataFrame(
    {"ticker": ["FCNTX", "AGTHX", "ANCFX", "VUG"], "weight": [25.0, 25.0, 25.0, 25.0]}
)
holdings_df = st.data_editor(
    default_holdings,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", required=True),
        "weight": st.column_config.NumberColumn("Weight", required=True, min_value=0.0),
    },
)

holdings_df = holdings_df.dropna(subset=["ticker", "weight"])
holdings_df["ticker"] = holdings_df["ticker"].str.strip().str.upper()
holdings_df = holdings_df[(holdings_df["ticker"] != "") & (holdings_df["weight"] > 0)]

if len(holdings_df) < 2:
    st.info("Add at least 2 holdings with positive weight to see portfolio-level diversification.")
    st.stop()
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()
if (end_date - start_date).days < 1095:
    st.sidebar.warning(
        "Short date range: rolling/Kalman burn-in and regime detection need several "
        "years of data, and the range should include at least one market downturn. "
        "Results may be unreliable."
    )

tickers = tuple(holdings_df["ticker"])
weights_raw = tuple(float(w) for w in holdings_df["weight"])

betas_by_ticker, skipped, n_obs = load_portfolio(tickers, weights_raw, start_date, end_date)

if skipped:
    st.warning(f"Skipped holding(s) (couldn't fetch data): {', '.join(skipped)}")

if len(betas_by_ticker) < 2:
    st.error("Fewer than 2 holdings had usable data -- can't compute portfolio-level diversification.")
    st.stop()

holdings = [PortfolioHolding(t, w) for t, w in zip(tickers, weights_raw, strict=True)]
weights = normalize_weights(holdings, available_tickers=set(betas_by_ticker))

blended = compute_blended_exposure(betas_by_ticker, weights)
diversification = compute_portfolio_diversification(betas_by_ticker, weights)
interp = describe_portfolio_concentration(
    diversification.n_holdings, diversification.effective_n, diversification.avg_pairwise_similarity
)

st.markdown(f"### {interp.headline}")
st.caption(interp.detail)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Holdings", diversification.n_holdings)
metric_col2.metric("Effective independent bets", f"{diversification.effective_n:.1f}")
metric_col3.metric("Avg. pairwise similarity", f"{diversification.avg_pairwise_similarity:.0%}")

st.markdown("#### Blended portfolio factor exposure")
st.caption(
    "Weighted sum of each holding's own unconstrained factor exposure -- what the portfolio "
    "looks like as if it were a single fund."
)
fig = go.Figure(
    go.Bar(
        x=blended.index,
        y=blended.values,
        marker_color=[FACTOR_COLORS.get(f, BASELINE) for f in blended.index],
    )
)
fig.update_layout(title="Blended factor exposure", yaxis_title="Weighted beta")
apply_theme(fig)
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Pairwise exposure similarity")
st.caption(
    "Cosine similarity of factor exposure between every pair of holdings -- 100% means two "
    "holdings tilt toward the same factors in the same proportions (redundant), 0% means "
    "unrelated tilts (genuinely diversifying)."
)
similarity = diversification.pairwise_similarity
heatmap = go.Figure(
    go.Heatmap(
        z=similarity.to_numpy(),
        x=similarity.columns,
        y=similarity.index,
        zmin=0,
        zmax=1,
        colorscale=px.colors.sequential.Blues,
        colorbar=dict(title="Similarity", tickformat=".0%"),
    )
)
heatmap.update_layout(title="Pairwise factor-exposure similarity")
apply_theme(heatmap)
st.plotly_chart(heatmap, use_container_width=True)

st.markdown("#### Holdings")
holdings_table = pd.DataFrame(
    {
        "weight (normalized)": pd.Series(weights),
        "trading days": pd.Series(n_obs),
    }
)
st.dataframe(holdings_table.style.format({"weight (normalized)": "{:.1%}"}), use_container_width=True)

st.caption(
    "Effective independent bets is a weight-and-similarity-adjusted heuristic, not a precise "
    "statistical estimate -- see LIMITATIONS.md. Want the full single-fund breakdown (stress "
    "behavior, style drift) for one of these holdings? Use the main Fund X-Ray page."
)
