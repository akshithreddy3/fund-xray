"""Style Drift Score: how far has a fund's exposure moved from its own baseline?

Takes the time-varying exposure vectors from Phase 3 (rolling-OLS or
Kalman betas, factor columns only -- `const`/alpha is never part of a
"style" vector) and reduces each date's vector to a single number: its
distance from a reference ("baseline") exposure vector. The baseline is
either:

- the fund's own exposure, averaged over its first `n_months` of
  history (the default -- "has this fund moved away from how it used
  to invest?"), or
- a user-supplied vector (e.g. a stated mandate's target weights --
  "has this fund moved away from what it says it does?").

Two distance metrics are offered because they answer subtly different
questions:

- **Cosine distance** (`1 - cosine_similarity`) measures a change in
  *shape* -- which factors dominate -- independent of overall
  magnitude. A fund whose betas all scale up by 20% but keep the same
  relative mix shows zero cosine drift.
- **Euclidean distance** measures a change in *magnitude* too. The
  same 20%-scale-up fund would show nonzero Euclidean drift.

A drift event is flagged when the score exceeds
`baseline_period_mean + threshold_std * baseline_period_std`, i.e. a
threshold calibrated to how much this specific fund's exposure moved
around *during its own baseline period* -- not a single universal
cutoff, which would treat a fund with a naturally noisy baseline the
same as one with a very stable one.

This module also computes the **Crowding Score** (Phase 7, stretch
goal): for a peer group of tickers, each peer's own time-varying
exposure vector (same Phase-3 rolling-OLS machinery, one fit per
ticker) is compared pairwise via cosine similarity, averaged across
all pairs at each date. A high, rising average pairwise similarity
means the peer group's managers are converging on the same factor
tilts -- "crowded" positioning that, if it unwinds, tends to unwind for
everyone in the group at once (a risk no single fund's own numbers can
reveal).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from itertools import combinations

import numpy as np
import pandas as pd

from src import config
from src.data_loader import DataLoader
from src.kalman_beta import rolling_ols_betas

logger = logging.getLogger(__name__)

DistanceMetric = str  # "cosine" | "euclidean"


@dataclass
class DriftScoreResult:
    drift: pd.Series  # index=date, value=distance from baseline
    baseline_vector: pd.Series  # index=factor name
    metric: DistanceMetric
    threshold: float
    flagged_events: pd.DataFrame  # columns: start, end, peak_date, peak_drift


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return np.nan
    return float(1.0 - (u @ v) / denom)


def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(u - v))


_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "cosine": cosine_distance,
    "euclidean": euclidean_distance,
}


def _factor_columns(betas: pd.DataFrame) -> list[str]:
    return [c for c in betas.columns if c != "const"]


def compute_baseline_vector(
    betas: pd.DataFrame, n_months: int = config.STYLE_DRIFT_BASELINE_MONTHS
) -> pd.Series:
    """Mean factor-exposure vector over the fund's first `n_months` of history."""
    factor_cols = _factor_columns(betas)
    exposures = betas[factor_cols].dropna(how="any")
    if exposures.empty:
        raise ValueError("No non-NaN exposure vectors available to build a baseline.")

    cutoff = exposures.index[0] + pd.DateOffset(months=n_months)
    baseline_window = exposures.loc[: cutoff - pd.Timedelta(days=1)]
    if baseline_window.empty:
        raise ValueError(
            f"No observations within the first {n_months} months to build a baseline "
            f"(series starts {exposures.index[0].date()})."
        )
    if len(baseline_window) < 20:
        logger.warning(
            "Only %d observations in the %d-month baseline window -- baseline vector "
            "may be noisy.",
            len(baseline_window),
            n_months,
        )
    return baseline_window.mean()


def _summarize_flagged_events(drift: pd.Series, flagged: pd.Series) -> pd.DataFrame:
    """Collapse a boolean flagged-day series into contiguous episodes."""
    if not flagged.any():
        return pd.DataFrame(columns=["start", "end", "peak_date", "peak_drift"])

    group_id = (flagged != flagged.shift(fill_value=False)).cumsum()
    events = []
    for _, group in drift[flagged].groupby(group_id[flagged]):
        events.append(
            {
                "start": group.index.min(),
                "end": group.index.max(),
                "peak_date": group.idxmax(),
                "peak_drift": float(group.max()),
            }
        )
    return pd.DataFrame(events)


def compute_style_drift(
    betas: pd.DataFrame,
    baseline_vector: pd.Series | None = None,
    metric: DistanceMetric = "cosine",
    n_months: int = config.STYLE_DRIFT_BASELINE_MONTHS,
    threshold_std: float = config.STYLE_DRIFT_THRESHOLD_STD,
) -> DriftScoreResult:
    """Distance-from-baseline time series plus a data-driven drift threshold.

    If `baseline_vector` is omitted, it's computed from the fund's own
    first `n_months` (a "has this fund moved from how it used to
    invest?" question). Pass an explicit `baseline_vector` (e.g. a
    stated-mandate weight vector) to instead ask "has it moved from
    what it says it does?".
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {list(_METRICS)}, got '{metric}'")
    dist_fn = _METRICS[metric]

    factor_cols = _factor_columns(betas)
    exposures = betas[factor_cols].dropna(how="any")
    n_dropped = len(betas) - len(exposures)
    if n_dropped:
        logger.info(
            "Dropped %d rows with missing exposure data before computing drift "
            "(e.g. rolling/Kalman burn-in period).",
            n_dropped,
        )

    if baseline_vector is None:
        baseline_vector = compute_baseline_vector(betas, n_months=n_months)
    else:
        baseline_vector = baseline_vector.reindex(factor_cols)
        if baseline_vector.isna().any():
            raise ValueError(
                f"Supplied baseline_vector is missing values for factors: "
                f"{baseline_vector[baseline_vector.isna()].index.tolist()}"
            )

    baseline_arr = baseline_vector.to_numpy()
    drift = exposures.apply(lambda row: dist_fn(row.to_numpy(), baseline_arr), axis=1)
    drift.name = "drift"

    baseline_cutoff = exposures.index[0] + pd.DateOffset(months=n_months)
    baseline_period_drift = drift.loc[: baseline_cutoff - pd.Timedelta(days=1)]
    threshold = float(
        baseline_period_drift.mean() + threshold_std * baseline_period_drift.std()
    )

    flagged = drift > threshold
    flagged_events = _summarize_flagged_events(drift, flagged)

    logger.info(
        "Style drift (%s): threshold=%.4f (baseline mean=%.4f, std=%.4f); "
        "%d flagged episode(s).",
        metric,
        threshold,
        baseline_period_drift.mean(),
        baseline_period_drift.std(),
        len(flagged_events),
    )

    return DriftScoreResult(
        drift=drift,
        baseline_vector=baseline_vector,
        metric=metric,
        threshold=threshold,
        flagged_events=flagged_events,
    )


# ---------------------------------------------------------------------------
# Crowding score (Phase 7, stretch goal)
# ---------------------------------------------------------------------------


@dataclass
class CrowdingScoreResult:
    crowding: pd.Series  # index=date, value=average pairwise cosine similarity
    n_peers_used: pd.Series  # index=date, value=how many peers had valid data that day
    peer_tickers: list[str]  # peers actually used (after dropping fetch/data failures)
    skipped_tickers: list[str]  # peers requested but excluded, and why (logged)


def average_pairwise_cosine_similarity(vectors: dict[str, np.ndarray]) -> float:
    """Mean cosine similarity (not distance) over every unordered peer pair.

    Similarity, not distance, is the natural unit for "crowding": 1.0
    means every peer's exposure vector points the same direction
    (maximally crowded), 0 means orthogonal/unrelated tilts.
    """
    pairs = list(combinations(vectors.keys(), 2))
    similarities = [1.0 - cosine_distance(vectors[a], vectors[b]) for a, b in pairs]
    return float(np.mean(similarities))


def build_peer_exposure_vectors(
    loader: DataLoader,
    peer_tickers: list[str],
    start: date | str,
    end: date | str,
    window: int = min(config.ROLLING_WINDOWS),
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Fetch each peer's dataset and fit rolling-OLS betas (Phase 3 machinery).

    A peer that fails to fetch (bad/delisted ticker, no overlapping factor
    history, etc.) is skipped -- logged explicitly, not silently dropped --
    rather than failing the whole crowding calculation.
    """
    peer_betas: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for ticker in peer_tickers:
        try:
            dataset = loader.build_fund_dataset(ticker, start, end)
            result = rolling_ols_betas(dataset.excess_returns, dataset.factor_returns, window=window)
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping peer '%s' from crowding score: %s", ticker, exc)
            skipped.append(ticker)
            continue
        peer_betas[ticker] = result.betas
    return peer_betas, skipped


def compute_crowding_score(
    peer_betas: dict[str, pd.DataFrame],
    skipped_tickers: list[str] | None = None,
) -> CrowdingScoreResult:
    """Average pairwise cosine similarity of peer exposure vectors, per date.

    Peers are allowed to have ragged histories (different inception
    dates, different rolling-window burn-ins): each date's score is
    computed from whichever peers have a valid (non-NaN) exposure
    vector that day, provided at least `config.MIN_PEERS_FOR_CROWDING`
    do. `n_peers_used` reports how many contributed to each date so a
    score based on 2 peers isn't mistaken for one based on the whole
    group.
    """
    if len(peer_betas) < config.MIN_PEERS_FOR_CROWDING:
        raise ValueError(
            f"Need at least {config.MIN_PEERS_FOR_CROWDING} peers with data, "
            f"got {len(peer_betas)}."
        )

    cleaned: dict[str, pd.DataFrame] = {}
    factor_cols: list[str] | None = None
    for ticker, betas in peer_betas.items():
        cols = _factor_columns(betas)
        if factor_cols is None:
            factor_cols = cols
        elif cols != factor_cols:
            raise ValueError(f"Factor columns mismatch for peer '{ticker}': {cols} vs {factor_cols}")
        cleaned[ticker] = betas[cols]

    all_dates = sorted(set().union(*(df.dropna(how="any").index for df in cleaned.values())))

    crowding_by_date: dict[pd.Timestamp, float] = {}
    n_peers_by_date: dict[pd.Timestamp, int] = {}
    for dt in all_dates:
        vectors = {
            ticker: df.loc[dt].to_numpy()
            for ticker, df in cleaned.items()
            if dt in df.index and df.loc[dt].notna().all()
        }
        if len(vectors) >= config.MIN_PEERS_FOR_CROWDING:
            crowding_by_date[dt] = average_pairwise_cosine_similarity(vectors)
            n_peers_by_date[dt] = len(vectors)

    if not crowding_by_date:
        raise ValueError(
            "No date has at least "
            f"{config.MIN_PEERS_FOR_CROWDING} peers with overlapping, valid exposure data."
        )

    crowding = pd.Series(crowding_by_date, name="crowding").sort_index()
    n_peers_used = pd.Series(n_peers_by_date, name="n_peers_used").sort_index()

    logger.info(
        "Crowding score computed over %d dates for %d peers (median peers/date: %.0f).",
        len(crowding),
        len(cleaned),
        n_peers_used.median(),
    )

    return CrowdingScoreResult(
        crowding=crowding,
        n_peers_used=n_peers_used,
        peer_tickers=list(cleaned.keys()),
        skipped_tickers=list(skipped_tickers) if skipped_tickers else [],
    )
