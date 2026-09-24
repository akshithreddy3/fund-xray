"""Market regime detection via a Gaussian Hidden Markov Model.

Fits an HMM on **market-wide** signals only (SPY returns, and VIX level
or realized SPY volatility if VIX isn't supplied) -- never on the target
fund's own returns. This independence is deliberate: if "stressed" were
partly defined by the fund's own bad days, every downstream
regime-conditional risk number would be circular (the fund would look
risky in "stress" almost by construction). Regimes here describe the
market environment; risk_metrics.py then asks how a given
fund's returns *behave conditional on* that independently-defined
environment.

Model: a Gaussian HMM (`hmmlearn.hmm.GaussianHMM`) with `n_regimes`
hidden states, each emitting a multivariate Gaussian over
`[market_return, volatility_signal]`. Two standard HMM assumptions are
worth stating explicitly since they don't obviously hold for markets:

- **Markov property**: tomorrow's regime depends on today's regime and
  nothing earlier. Real regime persistence (a stress period tends to
  last weeks, not one Markov step) is captured only through the fitted
  transition matrix's self-transition probabilities, not through any
  richer memory.
- **Gaussian, state-conditional-i.i.d. emissions**: within a regime,
  returns are assumed Gaussian (no fat tails) and independent across
  days given the state. Daily equity returns are well known to have
  fatter tails than a Gaussian, so extreme single-day moves can look
  like "regime noise" to this model rather than being modeled directly.

`hmmlearn` assigns state indices arbitrarily (state 0 isn't
necessarily "calm"), so `fit_regime_hmm` relabels states by the
ascending mean of the volatility feature after fitting.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src import config

logger = logging.getLogger(__name__)


@dataclass
class RegimeDetectionResult:
    n_regimes: int
    regime_labels: pd.Series  # index=date, values=str label e.g. "calm"/"stressed"
    regime_probabilities: pd.DataFrame  # index=date, columns=label names, smoothed posteriors
    state_means: pd.DataFrame  # index=label name, columns=feature name, in ORIGINAL units
    feature_columns: list[str]
    vol_source: str  # "vix" | "realized_vol"
    log_likelihood: float
    converged: bool
    model: GaussianHMM


def build_regime_features(
    market_returns: pd.Series,
    vix: pd.Series | None = None,
    realized_vol_window: int = config.REALIZED_VOL_WINDOW,
) -> tuple[pd.DataFrame, str]:
    """Assemble the [return, volatility_signal] feature matrix for the HMM.

    Prefers VIX (a forward-looking, market-implied vol measure) when
    available; falls back to trailing realized volatility of the market
    return series otherwise. Either way, rows are dropped (and logged)
    rather than filled where a feature is missing.
    """
    if vix is not None:
        features = pd.concat({"return": market_returns, "vol_signal": vix}, axis=1, sort=False)
        vol_source = "vix"
    else:
        realized_vol = market_returns.rolling(realized_vol_window).std() * np.sqrt(
            config.TRADING_DAYS_PER_YEAR
        )
        features = pd.concat(
            {"return": market_returns, "vol_signal": realized_vol}, axis=1, sort=False
        )
        vol_source = "realized_vol"

    n_before = len(features)
    features = features.dropna(how="any")
    n_dropped = n_before - len(features)
    if n_dropped:
        logger.warning(
            "Dropped %d rows with missing regime features (source=%s).", n_dropped, vol_source
        )

    return features, vol_source


def fit_regime_hmm(
    market_returns: pd.Series,
    vix: pd.Series | None = None,
    n_regimes: int = config.HMM_N_REGIMES_DEFAULT,
    random_state: int = config.HMM_RANDOM_STATE,
) -> RegimeDetectionResult:
    if n_regimes not in config.HMM_REGIME_LABELS:
        raise ValueError(
            f"n_regimes={n_regimes} has no configured label set; "
            f"add one to config.HMM_REGIME_LABELS or use one of {list(config.HMM_REGIME_LABELS)}."
        )

    features, vol_source = build_regime_features(market_returns, vix)
    feature_columns = list(features.columns)

    # Standardize before fitting: the two features live on very
    # different scales (returns ~0.01, VIX ~10-80), and an
    # unstandardized full-covariance fit is numerically fragile.
    means = features.mean()
    stds = features.std()
    X_scaled = ((features - means) / stds).to_numpy()

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type=config.HMM_COVARIANCE_TYPE,
        n_iter=config.HMM_N_ITER,
        random_state=random_state,
    )
    model.fit(X_scaled)

    if not model.monitor_.converged:
        logger.warning(
            "HMM did not converge within %d iterations (final log-likelihood change "
            "still above tol). Regime labels may be unstable.",
            config.HMM_N_ITER,
        )

    raw_states = model.predict(X_scaled)
    raw_probs = model.predict_proba(X_scaled)

    # Relabel: order raw state indices by ascending mean of the
    # volatility feature so label 0 is always "calmest".
    vol_idx = feature_columns.index("vol_signal")
    order = np.argsort(model.means_[:, vol_idx])
    labels_ordered = config.HMM_REGIME_LABELS[n_regimes]
    raw_to_label = {raw_state: labels_ordered[rank] for rank, raw_state in enumerate(order)}

    regime_labels = pd.Series(
        [raw_to_label[s] for s in raw_states], index=features.index, name="regime"
    )
    regime_probabilities = pd.DataFrame(
        raw_probs[:, order], index=features.index, columns=labels_ordered
    )

    # State means back in original (unstandardized) units, for interpretability.
    means_original = model.means_[order] * stds.to_numpy() + means.to_numpy()
    state_means = pd.DataFrame(means_original, index=labels_ordered, columns=feature_columns)

    logger.info(
        "Fit %d-regime HMM on %d observations (vol_source=%s). Regime frequencies: %s",
        n_regimes,
        len(features),
        vol_source,
        regime_labels.value_counts().to_dict(),
    )

    return RegimeDetectionResult(
        n_regimes=n_regimes,
        regime_labels=regime_labels,
        regime_probabilities=regime_probabilities,
        state_means=state_means,
        feature_columns=feature_columns,
        vol_source=vol_source,
        log_likelihood=float(model.score(X_scaled)),
        converged=bool(model.monitor_.converged),
        model=model,
    )


def validate_against_known_stress_periods(
    result: RegimeDetectionResult,
    stress_periods: dict[str, tuple] = config.KNOWN_STRESS_PERIODS,
    stress_label: str = "stressed",
) -> pd.DataFrame:
    """Sanity-check regime labels against known historical stress windows.

    For each named period, reports what fraction of its trading days the
    model assigned the most-stressed label. This is a validation-only
    step -- the periods are never used to fit or otherwise influence the
    model, which would be circular.
    """
    rows = []
    for name, (start, end) in stress_periods.items():
        window = result.regime_labels.loc[str(start) : str(end)]
        if window.empty:
            rows.append(
                {
                    "period": name,
                    "start": start,
                    "end": end,
                    "n_days": 0,
                    "pct_stressed": np.nan,
                }
            )
            continue
        pct_stressed = float((window == stress_label).mean())
        rows.append(
            {
                "period": name,
                "start": start,
                "end": end,
                "n_days": len(window),
                "pct_stressed": pct_stressed,
            }
        )
    return pd.DataFrame(rows).set_index("period")


def regime_stability_across_seeds(
    market_returns: pd.Series,
    vix: pd.Series | None,
    n_regimes: int = config.HMM_N_REGIMES_DEFAULT,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
) -> float:
    """Average pairwise day-label agreement across independent HMM refits.

    The single fixed `HMM_RANDOM_STATE` used elsewhere gives a
    reproducible regime split, but reproducible isn't the same as
    *stable* -- the EM algorithm can converge to different local optima
    from different random initializations. Regime labels are already
    relabeled by ascending volatility-feature mean (see `fit_regime_hmm`),
    so agreement can be checked directly by label name across seeds
    without a separate label-matching step. Low agreement means the
    calm/stressed split is sensitive to initialization, not a robust
    property of the data.
    """
    fits = [
        fit_regime_hmm(market_returns, vix=vix, n_regimes=n_regimes, random_state=seed)
        for seed in seeds
    ]
    label_series = [f.regime_labels for f in fits]

    agreements = []
    for a, b in combinations(label_series, 2):
        aligned_a, aligned_b = a.align(b, join="inner")
        if len(aligned_a) == 0:
            continue
        agreements.append(float((aligned_a == aligned_b).mean()))

    return float(np.mean(agreements)) if agreements else float("nan")


def rule_based_regime_labels(
    vol_signal: pd.Series,
    threshold_percentile: float = 80.0,
    calm_label: str = "calm",
    stress_label: str = "stressed",
) -> pd.Series:
    """Transparent, non-model-based regime baseline: 'stressed' whenever
    the volatility signal (VIX level, or realized vol -- whichever
    `build_regime_features` produced) sits above its own trailing
    `threshold_percentile`.

    Exists to check the HMM's calm/stressed split isn't an artifact of
    that specific model choice: if a one-line percentile rule tells a
    similar story, the regime-conditional findings are more defensible;
    if it disagrees sharply, that's worth surfacing too (see
    `compare_hmm_to_rule_based`).
    """
    signal = vol_signal.dropna()
    threshold = float(np.percentile(signal, threshold_percentile))
    labels = np.where(vol_signal >= threshold, stress_label, calm_label)
    return pd.Series(labels, index=vol_signal.index, name="regime_rule_based")


def compare_hmm_to_rule_based(
    hmm_labels: pd.Series, rule_labels: pd.Series, calm_label: str = "calm"
) -> float:
    """Agreement fraction between the HMM split and the rule-based baseline.

    Any non-calm HMM state (e.g. both 'elevated' and 'stressed' in a
    3-regime model) is collapsed to 'stressed' so a 2- or 3-regime HMM
    is directly comparable to the binary rule-based labels.
    """
    aligned_hmm, aligned_rule = hmm_labels.align(rule_labels, join="inner")
    if len(aligned_hmm) == 0:
        return float("nan")
    hmm_binary = aligned_hmm.where(aligned_hmm == calm_label, other="stressed")
    return float((hmm_binary == aligned_rule).mean())


def regime_episodes(regime_labels: pd.Series) -> pd.DataFrame:
    """Collapse a per-day regime label series into contiguous episodes.

    Used for regime-shaded charts (shading one rectangle per episode
    reads far better than one mark per day) and for at-a-glance
    inspection of how persistent each regime actually was.
    """
    group_id = (regime_labels != regime_labels.shift()).cumsum()
    episodes = [
        {
            "label": group.iloc[0],
            "start": group.index.min(),
            "end": group.index.max(),
            "n_days": len(group),
        }
        for _, group in regime_labels.groupby(group_id)
    ]
    return pd.DataFrame(episodes)
