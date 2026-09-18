# Limitations

Each entry below reflects a real caveat that surfaced while implementing the corresponding
module, not a boilerplate disclaimer.

## Returns-based analysis, generally

- All conclusions here are inferred from a fund's *return series*, never its actual holdings.
  Anything that changes intra-period and mean-reverts by the time returns are observed —
  temporary hedges, an option overlay, a leverage change that's unwound before period-end,
  short-term tactical trades — is invisible to every model in this project.
- Every estimate assumes the fund's exposures are reasonably stable *within* the estimation
  window. A fund that changes its factor exposure mid-window will show up as a blend of the two
  regimes, not either one cleanly (this is exactly the motivation for the time-varying exposure
  approach below, though even that has a reaction lag).

## Style analysis (`src/style_analysis.py`)

- **The Sharpe (1992) simplex constraint (`beta >= 0`, `sum(beta) == 1`) is a stylized
  adaptation, not a textbook application.** Sharpe's method was designed for a spanning set of
  fully-invested, long-only asset-class indices, where "weights sum to 1" literally describes a
  portfolio allocation. The Fama-French/Carhart factors used here (Mkt-RF, SMB, HML, RMW, CMA,
  Mom) are long-short, zero-net-investment portfolios, so the constrained weights should be read
  as *relative factor tilts* forced into an interpretable, bounded shape — not as an implied
  holdings breakdown.
- **The constraint can hide real, economically meaningful exposures.** Concretely: on FMAGX
  (2015–2024) the unconstrained regression finds a real negative HML loading (growth tilt) and
  negative SMB loading (large-cap tilt); the constrained model, unable to go negative, zeroes
  both out and compensates by over-loading Mkt-RF. Never treat the constrained result alone as
  ground truth — always check it against the unconstrained comparison table.
- Newey-West HAC standard errors correct for serial correlation/heteroskedasticity in the
  *unconstrained* regression's inference, but the constrained model reports no standard errors
  at all (there's no closed-form sampling distribution for a constrained-optimization estimator
  without bootstrapping, which this project doesn't implement) — its weights should be treated
  as point estimates only, not tested for statistical significance.

## Time-varying exposure (`src/kalman_beta.py`)

- **The Kalman filter assumes constant observation (idiosyncratic) variance R for the whole
  sample**, estimated once from the warm-up window. A real fund's residual volatility is not
  actually constant — it's typically higher during stress regimes (see the regime detection and
  regime-conditional risk sections below) — so the filter's uncertainty band, and to a lesser
  extent its gain, is mis-specified precisely during
  the periods most interesting to a risk analysis. A more complete model would let R vary too
  (e.g. via a separate volatility filter or regime-conditional R), which this project doesn't
  implement.
- **`delta` (and therefore the filter's responsiveness) is set by one empirical sweep against
  one event** (FMAGX's momentum beta around the 2020 COVID crash), not estimated by maximum
  likelihood. A formal treatment would estimate `delta` (or Q directly) via MLE over the
  state-space model's likelihood, which is possible with
  `statsmodels.tsa.statespace.MLEModel` but adds meaningful implementation complexity; the
  hand-picked default here is a reasonable, documented starting point, not a fitted parameter.
  Users comparing a different fund should re-check the default against a known event for that
  fund's return history rather than assuming `delta=0.02` transfers unchanged.
- **The initial state (mean and covariance) is seeded from a single static OLS fit on the first
  `warmup_window` observations**, and that warm-up period itself gets no beta estimate (logged,
  not backfilled). A fund with an unusual first few months (e.g. a fund's actual inception
  during an unusually volatile period) will get a less representative warm start than one with
  a typical early period.
- Rolling-window OLS and the Kalman filter are compared on the same beta series, but only
  informally (visual/point-in-time lag comparison in the README, not a formal statistical test
  of which estimator is "better" out-of-sample).

## Regime detection (`src/regime_detection.py`)

- **The HMM assumes Gaussian, state-conditional-i.i.d. emissions.** Daily equity returns have
  fatter tails than a Gaussian, so a single extreme day can be absorbed as "regime noise"
  rather than triggering (or ending) a regime transition, and the model has no way to represent
  that mismatch.
- **The Markov property means regime persistence is entirely a function of the fitted
  transition matrix's self-transition probabilities** — there's no explicit "minimum regime
  duration" or richer memory of how long a regime has already lasted. In practice this can
  produce brief, single-day flickers between labels around a regime boundary, which a human
  reading the same chart might smooth over.
- **Regime *count* is a modeling choice, not something estimated from the data.** The 2- vs.
  3-regime comparison above shows this concretely: the 2022 selloff is called "stressed" 99% of
  the time in a 2-state model but splits into distinct "elevated"/"stressed" phases in a
  3-state model — neither is more "correct"; they're different resolutions of the same
  underlying volatility path. Selecting `n_regimes` by, e.g., BIC would give a formal answer
  but was not implemented here in favor of letting the user choose and compare (Streamlit
  sidebar `n_regimes` control).
- **Regimes are fit once over the full available history and are not truly point-in-time.**
  Because `model.fit()` sees the whole sample at once, a day early in the sample is labeled
  using information about volatility regimes that hadn't happened yet from that day's
  perspective. This is standard practice for *retrospective* regime characterization (which is
  what this project does), but it means the regime labels should not be read as what a
  point-in-time analyst could have known on that date, and the pipeline as built is not
  suitable for real-time regime nowcasting without re-fitting only on data available up to
  each date.

## Regime-conditional risk metrics (`src/risk_metrics.py`)

- **Historical VaR/CVaR get noisier as a regime gets shorter.** The 95% confidence level means
  the tail estimate is driven by roughly the worst 5% of observations in that regime; a
  100-observation regime has only ~5 points defining its CVaR. `MIN_REGIME_OBS_WARNING` (20)
  logs a warning below that count, but the numbers are still reported -- treat a short regime's
  VaR/CVaR as directionally informative, not precise.
  - The one-time SPY/VIX regime fit used for the README's FMAGX example split ~1,759
    observations into two multi-hundred-day regimes, well above this threshold, but a
    3-regime split or a shorter fund history could easily produce a thin "stressed" bucket.
- **Regime exposures are refit independently per regime, with no shrinkage toward the
  overall-sample estimate.** A short, noisy regime's betas can look more extreme than they
  "really" are for exactly the small-sample reason above; a Bayesian or shrinkage estimator
  would pull short-regime betas back toward the whole-sample fit, which this project doesn't do.
- **The Euler variance decomposition is exact only in-sample for the same regime's own fitted
  betas and factor covariance.** It correctly attributes *this* regime's variance to *this*
  regime's factor loadings, but is not a forecast of how much each factor will contribute to
  variance going forward if the regime persists or if betas drift within it.

## Style Drift Score (`src/drift_score.py`)

- **A fixed baseline can't distinguish a permanent style evolution from a fund that will
  eventually revert.** With the default first-12-months baseline, a real, permanent shift in a
  fund's factor exposure (a genuine change in strategy or management) shows up identically to a
  fund the drift score just hasn't seen return to normal yet — the flagged-episode structure
  works well for short blips, but on FMAGX's real history it produces one continuous flagged
  episode spanning multiple years once the exposure permanently diverges. Read a long flagged
  episode as "still different from the baseline period," not as an ongoing anomaly the fund is
  expected to correct.
- **The baseline window itself can be unrepresentative.** If a fund's first `n_months` happen to
  fall in an unusual period for that fund (e.g. a manager transition, or a fund's early history
  before it reached scale), every later date's drift is measured against that atypical starting
  point. A user-supplied mandate vector sidesteps this by anchoring to a stated target instead
  of the fund's own possibly-unusual early history.
- **Cosine and Euclidean distance can disagree**, and this module doesn't tell you which is
  "right" for a given question — a fund that scales all its exposures up proportionally (e.g.
  adds leverage without changing its factor mix) shows large Euclidean drift but near-zero
  cosine drift. Check both if the two questions ("has the shape changed?" vs. "has the
  magnitude changed?") both matter for the fund in question.
- **The drift score inherits whichever estimator (rolling-OLS or Kalman) produced its input
  betas**, including that estimator's own burn-in period (dropped, not backfilled, consistent
  with the rest of this project) and, for the Kalman filter, its constant-observation-variance
  assumption (documented above).

## Crowding Score (`src/drift_score.py`)

- **The peer group is a hand-picked, hardcoded list (`config.DEFAULT_PEER_GROUP`), not
  discovered from the data.** Whether the crowding score is meaningful depends entirely on
  whether the chosen peers actually compete for similar capital/positioning; the default
  four-ticker large-cap-growth group is a plausible example, not a validated category
  definition, and a user pointing this at a mismatched peer group (e.g. mixing large-cap growth
  with small-cap value) will get a low similarity score that reflects the mismatch, not real
  "un-crowding."
- **Rolling-OLS-only, not Kalman.** `build_peer_exposure_vectors` always uses the Phase-3
  rolling-window estimator, not the Kalman filter, purely to keep the multi-ticker fetch-and-fit
  loop simple; nothing prevents swapping in Kalman betas per peer, but as built the crowding
  score inherits rolling OLS's lag/ghosting characteristics (documented above) for every peer.
- **A single average number hides dispersion.** Two peer groups with the same average pairwise
  similarity can have very different structures -- e.g. three peers nearly identical plus one
  outlier, vs. four peers all moderately similar to each other. The averaged crowding score
  doesn't distinguish these; a full pairwise similarity matrix (not currently exposed as a
  separate output) would be needed to see that structure.
- **A peer with too little overlapping history contributes nothing rather than being
  down-weighted.** `n_peers_used` reports this explicitly, but the crowding score itself treats
  a date backed by 2 peers identically in kind to one backed by 6 -- it doesn't reflect the
  lower reliability of a thinner sample.
