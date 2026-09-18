"""Static (single-window) returns-based style analysis.

Implements two estimators of the same linear factor model

    excess_return_t = sum_j beta_j * factor_j_t + e_t

and reports both so their disagreement is itself informative:

1. **Constrained ("Sharpe") style regression** -- Sharpe (1992),
   "Asset Allocation: Management Style and Performance Measurement
   Analysis". The factor loadings are restricted to a simplex
   (``beta_j >= 0``, ``sum_j beta_j == 1``) and there is no intercept.
   Solved with ``scipy.optimize.minimize`` (SLSQP) because inequality
   and equality constraints on the coefficients aren't expressible as
   an OLS normal-equations problem.

   Caveat worth stating plainly: Sharpe's simplex constraint is
   designed for a spanning set of fully-invested, long-only asset
   class indices (stocks/bonds/cash), where "weights summing to 1" has
   a literal portfolio-allocation meaning. Here the "factors" are the
   Fama-French/Carhart long-short, zero-net-investment factors
   (Mkt-RF, SMB, HML, RMW, CMA, Mom), for which the simplex constraint
   has no such literal interpretation -- it is used as a
   regularizer that forces an interpretable, bounded exposure profile
   rather than as a claim about implied asset allocation. Read the
   constrained weights as *relative factor tilts*, not as a holdings
   breakdown. This is documented again in LIMITATIONS.md.

2. **Unconstrained Fama-French OLS** -- the standard multi-factor
   regression (Fama & French, 1993/2015; Carhart, 1997), estimated
   with an intercept (alpha) and Newey-West (1987, 1994) HAC standard
   errors to correct for the serial correlation and heteroskedasticity
   that daily overlapping-window return data typically exhibit.

Comparing the two surfaces exactly what the constraint costs: a
constrained weight pinned at its 0 or 1 boundary that the unconstrained
regression estimates far from that boundary indicates the simplex
constraint is binding and distorting the fitted exposures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize


@dataclass
class StyleAnalysisResult:
    """Common result shape for both estimators, for easy comparison."""

    method: str  # "constrained" | "unconstrained"
    weights: pd.Series  # factor name -> beta
    alpha: float | None  # annualized-NOT; raw per-period intercept (None if not modeled)
    r_squared: float
    residuals: pd.Series
    n_obs: int
    standard_errors: pd.Series | None = None
    t_stats: pd.Series | None = None
    p_values: pd.Series | None = None
    nw_lags: int | None = None


def newey_west_lag(n_obs: int) -> int:
    """Automatic HAC lag selection (Newey & West, 1994, rule of thumb).

    L = floor(4 * (T / 100)^(2/9)). This grows slowly with sample size
    and is the standard default rather than a hand-picked constant.
    """
    return max(1, int(np.floor(4 * (n_obs / 100) ** (2 / 9))))


def run_constrained_style_analysis(
    excess_returns: pd.Series, factor_returns: pd.DataFrame
) -> StyleAnalysisResult:
    """Sharpe (1992) constrained style regression.

    minimize sum_t (y_t - X_t . w)^2
    subject to  w_j >= 0,  sum_j w_j = 1,  no intercept.
    """
    y, X = excess_returns.align(factor_returns, join="inner", axis=0)
    y_arr = y.to_numpy()
    X_arr = X.to_numpy()
    k = X_arr.shape[1]

    def objective(w: np.ndarray) -> float:
        resid = y_arr - X_arr @ w
        return float(resid @ resid)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * k
    w0 = np.full(k, 1.0 / k)

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Constrained style analysis failed to converge: {result.message}")

    weights = pd.Series(result.x, index=X.columns)
    fitted = X_arr @ result.x
    residuals = pd.Series(y_arr - fitted, index=y.index)

    ss_res = float(residuals.pow(2).sum())
    ss_tot = float(((y_arr - y_arr.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return StyleAnalysisResult(
        method="constrained",
        weights=weights,
        alpha=None,
        r_squared=r_squared,
        residuals=residuals,
        n_obs=len(y),
    )


def run_unconstrained_ols(
    excess_returns: pd.Series,
    factor_returns: pd.DataFrame,
    nw_lags: int | None = None,
) -> StyleAnalysisResult:
    """Unconstrained Fama-French OLS with Newey-West (HAC) standard errors."""
    y, X = excess_returns.align(factor_returns, join="inner", axis=0)
    X_with_const = sm.add_constant(X, has_constant="add")

    lags = nw_lags if nw_lags is not None else newey_west_lag(len(y))
    fit = sm.OLS(y, X_with_const).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    alpha = float(fit.params["const"])
    weights = fit.params.drop("const")
    residuals = fit.resid

    return StyleAnalysisResult(
        method="unconstrained",
        weights=weights,
        alpha=alpha,
        r_squared=float(fit.rsquared),
        residuals=residuals,
        n_obs=int(fit.nobs),
        standard_errors=fit.bse.drop("const"),
        t_stats=fit.tvalues.drop("const"),
        p_values=fit.pvalues.drop("const"),
        nw_lags=lags,
    )


def compare_style_results(
    constrained: StyleAnalysisResult, unconstrained: StyleAnalysisResult
) -> pd.DataFrame:
    """Side-by-side comparison table of constrained vs. unconstrained betas.

    ``at_boundary`` flags factors where the constrained weight sits at
    (or within floating-point tolerance of) 0 or 1 -- the clearest sign
    the simplex constraint, not the data, is driving that coefficient.
    """
    comparison = pd.DataFrame(
        {
            "constrained_weight": constrained.weights,
            "unconstrained_beta": unconstrained.weights,
        }
    )
    comparison["difference"] = comparison["constrained_weight"] - comparison["unconstrained_beta"]
    # SLSQP's convergence tolerance is on the objective, not the
    # parameters, so a weight pinned "at" a boundary typically lands
    # within ~1e-4 of it rather than exactly on it.
    comparison["at_boundary"] = comparison["constrained_weight"].apply(
        lambda w: bool(np.isclose(w, 0.0, atol=1e-3) or np.isclose(w, 1.0, atol=1e-3))
    )
    return comparison
