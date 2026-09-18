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
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config

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
