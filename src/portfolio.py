"""Portfolio X-Ray: multi-fund exposure aggregation.

Answers a question a single-fund view can't: "although I own N funds, am
I actually getting N sources of diversification, or are they largely
exposing me to the same underlying factors?" Built entirely on top of
`DataLoader` and the existing unconstrained style regression -- no new
estimation, just a weighted aggregation across holdings, plus the same
pairwise-cosine-similarity machinery `drift_score.py` already uses for
the single-fund Crowding Score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from itertools import combinations

import numpy as np
import pandas as pd

from src.data_loader import DataLoader, FundDataset
from src.drift_score import cosine_distance
from src.style_analysis import run_unconstrained_ols

logger = logging.getLogger(__name__)


@dataclass
class PortfolioHolding:
    ticker: str
    weight: float  # raw, unnormalized weight (e.g. dollar amount or percent)


@dataclass
class PortfolioDiversification:
    n_holdings: int
    effective_n: float  # weight-and-similarity-adjusted count of "independent" bets
    avg_pairwise_similarity: float
    pairwise_similarity: pd.DataFrame  # ticker x ticker cosine similarity matrix


def build_portfolio_datasets(
    loader: DataLoader,
    holdings: list[PortfolioHolding],
    start: date | str,
    end: date | str,
) -> tuple[dict[str, FundDataset], list[str]]:
    """Fetch each holding's dataset, skipping (and logging) any that fail.

    Mirrors `drift_score.build_peer_exposure_vectors`'s skip-and-log
    pattern: a bad/delisted ticker shouldn't fail the whole portfolio
    view.
    """
    datasets: dict[str, FundDataset] = {}
    skipped: list[str] = []
    for holding in holdings:
        try:
            datasets[holding.ticker] = loader.build_fund_dataset(holding.ticker, start, end)
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping portfolio holding '%s': %s", holding.ticker, exc)
            skipped.append(holding.ticker)
    return datasets, skipped


def normalize_weights(
    holdings: list[PortfolioHolding], available_tickers: set[str]
) -> dict[str, float]:
    """Raw holding weights -> weights summing to 1, restricted to holdings with usable data.

    Raw weights (dollar amounts, percentages, arbitrary units) don't need
    to already sum to 1 -- only their relative size matters for the
    blended exposure and diversification calculations below.
    """
    usable = [h for h in holdings if h.ticker in available_tickers]
    total = sum(h.weight for h in usable)
    if total <= 0:
        raise ValueError("Total portfolio weight of holdings with usable data must be positive.")
    return {h.ticker: h.weight / total for h in usable}


def compute_blended_exposure(
    betas_by_ticker: dict[str, pd.Series], weights: dict[str, float]
) -> pd.Series:
    """Weighted linear combination of each holding's factor exposure.

    Valid because a portfolio's return is the weighted sum of its
    holdings' returns, so (to a first-order, linear-factor-model
    approximation) the portfolio's factor exposure is the same
    weight-weighted sum of each holding's own exposure -- no new
    estimation, just linearity of the factor model already fit per
    holding by `style_analysis.run_unconstrained_ols`.
    """
    tickers = list(betas_by_ticker)
    factor_cols = betas_by_ticker[tickers[0]].index
    blended = pd.Series(0.0, index=factor_cols)
    for ticker in tickers:
        blended = blended + weights[ticker] * betas_by_ticker[ticker].reindex(factor_cols).fillna(0.0)
    return blended


def compute_portfolio_diversification(
    betas_by_ticker: dict[str, pd.Series], weights: dict[str, float]
) -> PortfolioDiversification:
    """How many effectively distinct factor bets do these holdings represent?

    `effective_n` shrinks the raw holding count by how similar (cosine
    similarity of factor exposure) the holdings are to each other,
    weighted by portfolio weight -- two holdings with near-identical
    exposure act like one bet no matter how the raw count reads. This is
    a diversification-style heuristic (weight-weighted average pairwise
    similarity collapsing toward 1/N for N truly independent, equally
    weighted bets), not a new statistical model.
    """
    tickers = list(betas_by_ticker)
    n = len(tickers)
    similarity = pd.DataFrame(1.0, index=tickers, columns=tickers)
    for a, b in combinations(tickers, 2):
        sim = 1.0 - cosine_distance(betas_by_ticker[a].to_numpy(), betas_by_ticker[b].to_numpy())
        similarity.loc[a, b] = sim
        similarity.loc[b, a] = sim

    if n < 2:
        return PortfolioDiversification(
            n_holdings=n, effective_n=float(n), avg_pairwise_similarity=1.0, pairwise_similarity=similarity
        )

    w = np.array([weights[t] for t in tickers])
    # Weight-weighted average pairwise similarity (excluding the diagonal).
    weight_outer = np.outer(w, w)
    np.fill_diagonal(weight_outer, 0.0)
    weighted_similarity_sum = float((similarity.to_numpy() * weight_outer).sum())
    weight_pair_sum = float(weight_outer.sum())
    avg_similarity = weighted_similarity_sum / weight_pair_sum if weight_pair_sum > 0 else float("nan")

    # Effective N: N truly independent, equally-weighted holdings has
    # avg pairwise similarity ~0 and effective_n == N; N identical
    # holdings has avg pairwise similarity == 1 and effective_n == 1.
    # Linear interpolation between those two anchors is the simplest
    # monotonic mapping satisfying both, and is treated as a heuristic,
    # not a precise estimate (see LIMITATIONS.md).
    effective_n = 1.0 + (n - 1) * (1.0 - avg_similarity)

    return PortfolioDiversification(
        n_holdings=n,
        effective_n=float(effective_n),
        avg_pairwise_similarity=float(avg_similarity),
        pairwise_similarity=similarity,
    )


def build_portfolio_betas(
    datasets: dict[str, FundDataset],
) -> dict[str, pd.Series]:
    """Unconstrained factor exposure (betas only, no alpha) per holding."""
    betas: dict[str, pd.Series] = {}
    for ticker, dataset in datasets.items():
        fit = run_unconstrained_ols(dataset.excess_returns, dataset.factor_returns)
        betas[ticker] = fit.weights
    return betas
