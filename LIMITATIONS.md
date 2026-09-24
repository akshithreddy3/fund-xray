# Limitations

This isn't a boilerplate disclaimers page. Every point below is a real modeling tradeoff that
surfaced while building the corresponding module, and each one affects how the numbers in
[README.md](README.md) should be read.

## What returns-based analysis can never see

Every conclusion in this project comes from a fund's *return series*, never its actual
holdings. That's the whole premise — it's also the ceiling. Anything that changes within a
period and unwinds before the return is observed — a temporary hedge, an option overlay, a
leverage change closed out before period-end, short-term tactical trades — simply isn't there
in the data. No amount of clever modeling recovers it, because it left no trace in the signal
being analyzed.

A related assumption runs through every estimator here: that a fund's exposures are reasonably
stable *within* whatever window is being fit. A fund that genuinely changes its factor exposure
mid-window shows up as a blend of the two states, not as either one cleanly — which is exactly
the motivation for the time-varying estimators below, though even those only track the change
with a lag, not instantly.

## Style analysis (`src/style_analysis.py`)

The Sharpe (1992) simplex constraint — weights non-negative, summing to one — is a stylized
adaptation here, not a textbook application. Sharpe designed it for a spanning set of
fully-invested, long-only asset-class indices, where "weights sum to one" is literally a
portfolio allocation. The Fama-French/Carhart factors used in this project (Mkt-RF, SMB, HML,
RMW, CMA, Mom) are long-short, zero-net-investment portfolios, so that literal reading doesn't
carry over. Read the constrained weights as bounded, interpretable factor *tilts* — not as an
implied holdings breakdown.

That constraint can also hide exposures that are real and economically meaningful. On FMAGX
(2015–2024) the unconstrained regression finds a genuine negative HML loading (a growth tilt)
and negative SMB loading (a large-cap tilt); the constrained model, unable to go negative,
zeroes both out and compensates by over-loading Mkt-RF instead. Don't treat the constrained
result alone as ground truth — it's only informative next to the unconstrained comparison.

And on inference: Newey-West HAC standard errors correct for serial correlation in the
*unconstrained* regression, but the constrained model reports no standard errors at all —
there's no closed-form sampling distribution for a constrained-optimization estimator without
bootstrapping, which isn't implemented here. Treat its weights as point estimates, not as
something you can run a significance test against.

**The VIF (multicollinearity) check has its own rule-of-thumb, not a formal decision rule.**
`compute_factor_vif` flags a factor above the conventional VIF > 10 threshold, but 10 is a
convention borrowed from applied-econometrics practice, not something derived from this
project's own data. A factor just under that line isn't meaningfully more reliable than one
just over it, and a high-VIF factor's beta can still be part of a perfectly good *joint* fit —
VIF only says the *individual* coefficient is underdetermined by collinearity, not that
anything is wrong with the regression as a whole.

## Time-varying exposure (`src/kalman_beta.py`)

The Kalman filter assumes constant observation (idiosyncratic) variance for the entire sample,
estimated once from the warm-up window. A real fund's residual volatility isn't actually
constant — it's typically higher during stress regimes, which is the whole subject of the
regime-conditional sections of this project — so the filter's uncertainty band, and to a lesser
extent its gain, is mis-specified precisely during the periods most worth getting right. A more
complete model would let observation variance vary too (a separate volatility filter, or a
regime-conditional variance), which this project doesn't implement.

`delta` — and therefore how responsive the filter is — was set by one empirical sweep against
one event (FMAGX's momentum beta through the 2020 COVID crash), not estimated by maximum
likelihood. A formal treatment would fit `delta` (or the process-noise covariance directly) via
MLE over the state-space model's likelihood — `statsmodels.tsa.statespace.MLEModel` makes this
possible, but it adds real implementation complexity. The hand-picked default here is a
reasonable, documented starting point, not a fitted parameter, and anyone applying this to a
different fund should re-check it against a known event in that fund's own history rather than
assume `delta=0.02` transfers unchanged.

The filter's initial state is seeded from a single static OLS fit on the first `warmup_window`
observations, and that warm-up period itself gets no beta estimate at all — it's logged, not
backfilled. A fund whose early history happens to be unusually volatile (an inception during a
turbulent stretch, for instance) gets a less representative warm start than one with an
ordinary early period.

One more thing worth being honest about: the rolling-OLS and Kalman estimates are compared only
informally — a visual, point-in-time lag comparison, not a formal statistical test of which
estimator performs better out-of-sample.

## Regime detection (`src/regime_detection.py`)

The HMM assumes Gaussian, state-conditional i.i.d. emissions. Daily equity returns are known to
have fatter tails than that, so a single extreme day can get absorbed as regime noise instead of
triggering — or ending — a transition, and the model has no mechanism to represent that
mismatch.

Because the Markov property makes regime persistence entirely a function of the fitted
transition matrix's self-transition probabilities, there's no explicit notion of a minimum
regime duration, or any memory of how long a regime has already lasted. In practice this can
produce brief, single-day flickers between labels right at a regime boundary — the kind of
thing a human eyeballing the same chart would probably smooth over without thinking about it.

Regime *count* is a modeling choice here, not something estimated from the data, and the 2- vs.
3-regime comparison in the README shows exactly what that choice costs: the 2022 selloff reads
as "stressed" 99% of the time under a 2-state model but splits into distinct
"elevated"/"stressed" phases under a 3-state model. Neither is more correct — they're different
resolutions of the same underlying volatility path. Selecting `n_regimes` by something like BIC
would give a more principled answer; this project instead exposes it as a sidebar control and
lets the user compare directly.

And the regimes themselves are fit once over the entire available history, so they aren't truly
point-in-time — because `model.fit()` sees the whole sample at once, a day early in the series
gets labeled with the benefit of volatility information that hadn't happened yet as of that
date. That's standard for *retrospective* regime characterization, which is what this project
does, but it means the labels shouldn't be read as what a point-in-time analyst could have known
on that date, and this pipeline isn't set up for real-time regime nowcasting without refitting
on only the data available up to each date.

**The multi-seed stability check and the rule-based baseline are both robustness evidence, not
proof.** `regime_stability_across_seeds` refits across only 5 seeds — a reassuring agreement
number from 5 seeds doesn't rule out disagreement from a 6th, and the check only detects
sensitivity to *initialization*, not sensitivity to `n_regimes`, `covariance_type`, or the
feature set itself (return + VIX/realized vol). `rule_based_regime_labels`'s own 80th-percentile
threshold is itself an arbitrary, undefended choice — a different percentile would shift how
much of the series reads "stressed" and therefore shift its agreement with the HMM. High
agreement between the two is evidence the calm/stressed split isn't a fragile artifact of one
specific model; it is not evidence that 80% (or the HMM's own hyperparameters) is the "right"
line to draw.

## Regime-conditional risk metrics (`src/risk_metrics.py`)

Historical VaR/CVaR get noisier the shorter a regime is — the 95% confidence level means the
tail estimate is driven by roughly the worst 5% of that regime's observations, so a
100-observation regime has only about five points defining its CVaR. `MIN_REGIME_OBS_WARNING`
(20) logs a warning below that count, but the number still gets reported; treat a short
regime's VaR/CVaR as directionally informative, not precise. The one-time SPY/VIX regime fit
behind the README's FMAGX numbers split roughly 1,759 observations into two multi-hundred-day
regimes, comfortably above that threshold — but a 3-regime split, or a shorter fund history,
could easily land in a thin, noisier "stressed" bucket.

Regime exposures are refit independently per regime with no shrinkage back toward the
whole-sample estimate. A short, noisy regime's betas can look more extreme than they "really"
are for the same small-sample reason above; a Bayesian or shrinkage approach would pull them
back toward the overall fit, and this project doesn't do that.

The Euler variance decomposition is exact only in-sample, for that regime's own fitted betas and
factor covariance — it correctly attributes the regime's *observed* variance to its factor
loadings, but it isn't a forecast of how much each factor will contribute going forward if the
regime persists or betas keep drifting within it.

**The fixed-beta cross-check is a robustness check on *one* confound, not a proof of the
underlying finding.** `compute_fixed_beta_cross_check` addresses a specific, real concern —
that a regime-refit R² change could partly be an artifact of giving the model extra freedom to
fit a smaller subset of data — by applying a reference regime's own fitted exposure, un-refit,
to the target regime. But the "reference" exposure is itself just one regime's own point
estimate, with its own sampling noise; the check doesn't account for uncertainty in *that*
estimate, and it says nothing about the other confounds discussed elsewhere in this file (thin
regimes, no shrinkage, Gaussian-emission mismatch). A small refit-vs-fixed gap is reassuring; it
is not equivalent to a formal out-of-sample validation.

## Style drift score (`src/drift_score.py`)

A fixed baseline can't tell the difference between a permanent style change and a fund that just
hasn't reverted yet. With the default first-12-months baseline, a genuine, lasting shift in a
fund's strategy looks identical to a fund the drift score simply hasn't caught coming back to
normal. On FMAGX's real history this produces one continuous flagged episode spanning multiple
years, once the exposure diverges for good — the flagged-episode structure handles short blips
well, but a multi-year flag should be read as "still different from the baseline period," not
as an ongoing anomaly the fund is expected to correct.

The baseline window itself can be unrepresentative. If a fund's first several months happen to
fall in an atypical period for that fund — a manager transition, or an early period before it
reached scale — every later date's drift gets measured against that atypical starting point. A
user-supplied mandate vector sidesteps this by anchoring to a stated target instead of the
fund's own possibly-unusual early history.

Cosine and Euclidean distance can also disagree, and the module doesn't adjudicate which one is
"right" for a given question — a fund that scales all its exposures up proportionally (adds
leverage without changing its factor mix, say) shows large Euclidean drift but near-zero cosine
drift. If both "has the shape changed?" and "has the magnitude changed?" matter for the fund in
question, check both.

Finally, the drift score inherits whatever estimator — rolling OLS or Kalman — produced its
input betas, including that estimator's own burn-in period (dropped, not backfilled, consistent
with the rest of this project) and, for the Kalman filter, its constant-observation-variance
assumption documented above.

**The percentile threshold alternative trades one assumption for a different weakness.**
`threshold_method="percentile"` avoids assuming the baseline-period drift distribution is
gaussian, but a percentile estimate is itself noisy with few observations — a 6-month baseline
window is roughly 125 trading days, so its 95th percentile is set by only the handful of most
extreme days in that window, which can be unstable in exactly the way `baseline_skew` is meant
to flag. Neither threshold method is uniformly better; `baseline_skew` is reported precisely so
a user can judge which assumption is less wrong for a given fund's baseline period, rather than
trusting the default blindly.

## Crowding score (`src/drift_score.py`)

The peer group is a hand-picked, hardcoded list (`config.DEFAULT_PEER_GROUP`), not discovered
from the data. Whether the crowding score means anything at all depends entirely on whether the
chosen peers actually compete for similar capital and positioning; the default four-ticker
large-cap-growth group is a plausible example, not a validated category definition. Point this
at a mismatched peer group — large-cap growth mixed with small-cap value, say — and you'll get a
low similarity score that reflects the mismatch, not genuine "un-crowding."

`build_peer_exposure_vectors` always uses the rolling-window estimator, never the Kalman filter,
purely to keep the multi-ticker fetch-and-fit loop simple. Nothing prevents swapping in Kalman
betas per peer, but as built, the crowding score inherits rolling OLS's lag and ghosting
characteristics (documented above) for every peer in the group.

A single averaged number also hides dispersion. Two peer groups with identical average pairwise
similarity can have very different structures underneath — three peers nearly identical plus one
outlier, versus four peers all moderately similar to each other. The averaged score can't tell
these apart; seeing that structure would require exposing the full pairwise similarity matrix,
which isn't currently a separate output.

And a peer with too little overlapping history contributes nothing rather than being
down-weighted. `n_peers_used` reports this explicitly, but the crowding score itself treats a
date backed by two peers the same in kind as one backed by six — it doesn't reflect the lower
reliability of the thinner sample.

## Portfolio X-Ray (`src/portfolio.py`)

**Blended exposure assumes each holding's own factor exposure is stationary and correctly
estimated over the full requested window.** `compute_blended_exposure` is a weighted sum of
each holding's *own* unconstrained regression beta — which is exactly right algebraically (a
portfolio's return is the weighted sum of its holdings' returns, so its factor exposure is that
same weighted sum), but it inherits every limitation already documented above for the
unconstrained regression itself, once per holding. It also doesn't account for rebalancing: the
blend is computed from each holding's whole-period beta at the *stated* weights, not from an
actually-rebalanced-through-time portfolio return series, so it describes "a portfolio held at
these weights, analyzed holding-by-holding," not a simulated backtest of the portfolio itself.

**"Effective independent bets" is a heuristic, not a rigorous diversification estimator.**
`compute_portfolio_diversification`'s `effective_n` linearly interpolates between two anchor
cases (all holdings identical -> 1; all holdings pairwise-orthogonal -> N) using weight-weighted
average pairwise cosine similarity. This is a reasonable, monotonic, easy-to-explain number, but
it is not a formal measure like an eigenvalue-based effective-bets count from the portfolio's
actual return covariance matrix (which would also capture idiosyncratic correlation between
holdings' residuals, not just their factor exposure shape). Two portfolios with the same average
pairwise similarity can have different underlying structures (see the analogous point already
made about the single-fund Crowding Score above) that this single number can't distinguish.

**Raw weights are silently renormalized to sum to 1 across only the holdings with usable
data.** `normalize_weights` drops any ticker that failed to fetch and renormalizes the remainder
— which is the only sensible behavior, but it means a portfolio where a large holding fails to
fetch will have its *remaining* holdings' effective weights inflated well past what the user
actually entered. The UI surfaces which tickers were skipped, but doesn't re-confirm the
renormalized weights back to the user before computing the blend.

## Interpretation layer (`src/interpretation.py`)

**Every classification threshold in this layer is a judgment call, not a statistically derived
cutoff.** The "diversification weakens" vs. "holds" split (idiosyncratic-share delta more
negative than -2 percentage points), the significance threshold (p < 0.05), the VIF threshold
(> 10), and the crowding threshold (average pairwise similarity >= 85%) are all conventional,
round-number defaults, chosen for defensibility and consistency rather than fit to this
project's own data. A fund whose numbers sit close to one of these lines can flip categories
from a small data revision or a slightly different date range, even though the underlying
numbers barely moved. The `tag` a function returns is a discrete summary of a continuous
number; the continuous number (always included in each `Interpretation.detail`, and in the
underlying tables) is the more reliable thing to read closely.
