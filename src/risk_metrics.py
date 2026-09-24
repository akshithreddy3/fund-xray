"""Regime-conditional risk attribution.

Answers the question a single, whole-sample risk number can't: what
does this fund's risk actually look like *during* a given market
regime (see `regime_detection.py`), as opposed to on average across all regimes blended
together? For each regime (plus "overall", as a baseline), this module
recomputes:

- **VaR(95%) / CVaR(95%)** of the fund's own returns, via historical
  (empirical-quantile) simulation rather than a parametric (Gaussian)
  formula. Daily equity-like returns are well known to have fatter
  tails than a normal distribution, and a stress regime by definition
  concentrates the extreme days, so assuming normality here would
  understate exactly the tail risk this analysis exists to surface.
  The tradeoff is that historical VaR/CVaR is noisier with few
  observations -- a real concern for short regimes (see
  `MIN_REGIME_OBS_WARNING`).
- **Factor exposures and R²** via the same unconstrained OLS used in
  `style_analysis.py`, refit on only the regime's observations.
- **Factor vs. idiosyncratic variance share**, via an Euler
  (marginal-contribution-weighted) decomposition of the fitted factor
  variance `beta' * Sigma_f * beta`, so each factor's contribution to
  total variance is reported individually and sums exactly to the
  factor-driven share (with the residual making up the rest).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.style_analysis import run_unconstrained_ols

logger = logging.getLogger(__name__)

OVERALL_LABEL = "overall"


@dataclass
class RegimeRiskMetrics:
    regime: str
    n_obs: int
    var_95: float
    cvar_95: float
    annualized_vol: float
    annualized_mean_return: float
    r_squared: float
    factor_betas: pd.Series
    factor_risk_contribution_pct: pd.Series  # share of TOTAL variance, sums to ~r_squared
    idiosyncratic_variance_share: float  # ~= 1 - r_squared
    idiosyncratic_annualized_vol: float  # absolute idiosyncratic vol, not just its share
    factor_driven_annualized_vol: float  # absolute factor-driven vol, not just its share


def historical_var_cvar(
    returns: pd.Series, confidence: float = config.VAR_CONFIDENCE
) -> tuple[float, float]:
    """Historical (empirical-quantile) VaR and CVaR, returned as positive loss magnitudes.

    VaR is the loss such that `1 - confidence` of observed returns were
    worse than it; CVaR (expected shortfall) is the average loss among
    exactly those worst observations.
    """
    tail_quantile = 1.0 - confidence
    threshold = float(np.quantile(returns, tail_quantile))
    tail_losses = returns[returns <= threshold]
    var = -threshold
    cvar = -float(tail_losses.mean()) if len(tail_losses) else var
    return var, cvar


def _compute_metrics_for_subset(
    label: str,
    fund_returns: pd.Series,
    excess_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> RegimeRiskMetrics:
    n_obs = len(fund_returns)
    if n_obs < config.MIN_REGIME_OBS_WARNING:
        logger.warning(
            "Regime '%s' has only %d observations (< %d) -- exposure and VaR/CVaR "
            "estimates for this regime are statistically fragile.",
            label,
            n_obs,
            config.MIN_REGIME_OBS_WARNING,
        )

    var_95, cvar_95 = historical_var_cvar(fund_returns)

    ols = run_unconstrained_ols(excess_returns, factor_returns)
    betas = ols.weights

    factor_cov = factor_returns.cov()  # sample covariance, ddof=1, matches Var(fund) below
    total_variance = float(excess_returns.var(ddof=1))

    # Euler decomposition of the fitted factor variance beta' Sigma_f beta:
    # contribution_j = beta_j * (Sigma_f @ beta)_j, which sums exactly to
    # beta' Sigma_f beta by construction (Euler's homogeneous-function identity).
    sigma_beta = factor_cov.to_numpy() @ betas.to_numpy()
    contribution = betas.to_numpy() * sigma_beta
    factor_risk_contribution_pct = pd.Series(
        contribution / total_variance if total_variance > 0 else np.full_like(contribution, np.nan),
        index=betas.index,
    )

    idiosyncratic_variance_share = 1.0 - float(factor_risk_contribution_pct.sum())
    annualized_vol = float(fund_returns.std(ddof=1) * np.sqrt(config.TRADING_DAYS_PER_YEAR))

    return RegimeRiskMetrics(
        regime=label,
        n_obs=n_obs,
        var_95=var_95,
        cvar_95=cvar_95,
        annualized_vol=annualized_vol,
        annualized_mean_return=float(fund_returns.mean() * config.TRADING_DAYS_PER_YEAR),
        r_squared=ols.r_squared,
        factor_betas=betas,
        factor_risk_contribution_pct=factor_risk_contribution_pct,
        idiosyncratic_variance_share=idiosyncratic_variance_share,
        # Absolute vol split, not just the share: two funds with the same
        # idiosyncratic *share* can have very different idiosyncratic vol
        # in absolute terms if their total vol differs. sqrt() is valid
        # here because variance is additive (factor-driven + idiosyncratic
        # = total) and share is a unit-less ratio of that same variance.
        idiosyncratic_annualized_vol=annualized_vol * np.sqrt(max(idiosyncratic_variance_share, 0.0)),
        factor_driven_annualized_vol=annualized_vol
        * np.sqrt(max(1.0 - idiosyncratic_variance_share, 0.0)),
    )


def split_by_regime(
    fund_returns: pd.Series,
    excess_returns: pd.Series,
    factor_returns: pd.DataFrame,
    regime_labels: pd.Series,
) -> dict[str, dict[str, pd.Series | pd.DataFrame]]:
    """Align fund/excess/factor returns with regime labels and split by label.

    Returns ``{"overall": {...}, "calm": {...}, "stressed": {...}, ...}``,
    each value a dict with "fund", "excess", "factors" keys. Shared by
    `compute_regime_risk_metrics` and `compute_fixed_beta_cross_check` so
    both use identical alignment/dropna logic instead of duplicating it.

    `regime_labels` typically comes from a model fit on independent
    market data (`regime_detection.py`) over a different date range than
    the fund's own history, so dates are inner-joined here; any
    non-overlap is logged rather than silently dropped.
    """
    combined = pd.concat(
        {"fund": fund_returns, "excess": excess_returns, "regime": regime_labels},
        axis=1,
        sort=False,
    ).join(factor_returns, how="inner")

    n_before = max(len(fund_returns), len(regime_labels))
    combined = combined.dropna(how="any")
    n_dropped = n_before - len(combined)
    if n_dropped:
        logger.warning(
            "Dropped %d rows with no overlapping fund/regime/factor data while "
            "splitting by regime.",
            n_dropped,
        )
    if combined.empty:
        raise ValueError("No overlapping dates between fund returns and regime labels.")

    subsets: dict[str, dict[str, pd.Series | pd.DataFrame]] = {
        OVERALL_LABEL: {
            "fund": combined["fund"],
            "excess": combined["excess"],
            "factors": combined[config.FACTOR_COLUMNS],
        }
    }
    for label, subset in combined.groupby("regime", observed=True):
        subsets[label] = {
            "fund": subset["fund"],
            "excess": subset["excess"],
            "factors": subset[config.FACTOR_COLUMNS],
        }
    return subsets


def compute_regime_risk_metrics(
    fund_returns: pd.Series,
    excess_returns: pd.Series,
    factor_returns: pd.DataFrame,
    regime_labels: pd.Series,
) -> dict[str, RegimeRiskMetrics]:
    """Risk metrics for "overall" plus every regime label present in the data."""
    subsets = split_by_regime(fund_returns, excess_returns, factor_returns, regime_labels)
    return {
        label: _compute_metrics_for_subset(label, s["fund"], s["excess"], s["factors"])
        for label, s in subsets.items()
    }


def evaluate_fixed_beta_r_squared(
    excess_returns: pd.Series,
    factor_returns: pd.DataFrame,
    alpha: float,
    betas: pd.Series,
) -> float:
    """R^2 of a return series against a FIXED (not refit) alpha/betas.

    Used to check whether a regime-conditional R^2 difference survives
    without giving the model the extra freedom to refit its coefficients
    on that regime's own data -- a regime-refit R^2 can rise partly
    just because refitting has more freedom relative to a smaller
    sample, not only because the fund's behavior genuinely changed
    between regimes.
    """
    y, X = excess_returns.align(factor_returns[betas.index], join="inner", axis=0)
    fitted = alpha + X.to_numpy() @ betas.to_numpy()
    residuals = y.to_numpy() - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y.to_numpy() - y.to_numpy().mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def compute_fixed_beta_cross_check(
    fund_returns: pd.Series,
    excess_returns: pd.Series,
    factor_returns: pd.DataFrame,
    regime_labels: pd.Series,
    reference_regime: str,
    target_regime: str,
) -> dict[str, float | str]:
    """Compare a regime's own refit R^2 against the R^2 from applying
    another regime's fixed exposure, un-refit.

    If both tell a similar story (e.g. both show low R^2 in "calm" and
    high R^2 in "stressed"), the regime-conditional finding is not just
    an artifact of independently refitting each regime's coefficients --
    it survives even under a fixed reference exposure.
    """
    subsets = split_by_regime(fund_returns, excess_returns, factor_returns, regime_labels)
    if reference_regime not in subsets or target_regime not in subsets:
        raise ValueError(
            f"Both '{reference_regime}' and '{target_regime}' must be present regimes; "
            f"got {list(subsets)}."
        )

    reference = subsets[reference_regime]
    reference_fit = run_unconstrained_ols(reference["excess"], reference["factors"])

    target = subsets[target_regime]
    target_refit = run_unconstrained_ols(target["excess"], target["factors"])
    fixed_r_squared = evaluate_fixed_beta_r_squared(
        target["excess"], target["factors"], reference_fit.alpha, reference_fit.weights
    )

    return {
        "reference_regime": reference_regime,
        "target_regime": target_regime,
        "refit_r_squared": target_refit.r_squared,
        "fixed_beta_r_squared": fixed_r_squared,
    }


def regime_comparison_table(metrics: dict[str, RegimeRiskMetrics]) -> pd.DataFrame:
    """One row per regime (+ overall), side by side, for the dashboard/README."""
    rows = {}
    for label, m in metrics.items():
        rows[label] = {
            "n_obs": m.n_obs,
            "annualized_vol": m.annualized_vol,
            "annualized_mean_return": m.annualized_mean_return,
            "VaR_95": m.var_95,
            "CVaR_95": m.cvar_95,
            "R_squared": m.r_squared,
            "idiosyncratic_variance_share": m.idiosyncratic_variance_share,
            "idiosyncratic_annualized_vol": m.idiosyncratic_annualized_vol,
            "factor_driven_annualized_vol": m.factor_driven_annualized_vol,
            **{f"beta_{f}": m.factor_betas[f] for f in m.factor_betas.index},
        }
    return pd.DataFrame(rows).T
