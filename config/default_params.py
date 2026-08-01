"""Default model hyperparameters and inference configuration for PMCMC.

Defines four dataclasses consumed throughout the codebase:

  * ``ModelParams``        — statistical model parameters (regime variances,
                             insider logistic coefficients, observation noise).
  * ``PhiPrior``           — the single prior spec the VEM M-step optimizes
                             against, shared with the Laplace layer and PSIS.
  * ``OnlineScorerConfig`` — forgetting / learning-rate schedule for the
                             streaming scorer (``src/inference/online_scorer.py``).
  * ``InferenceConfig``    — particle filter / iPMCMC tuning knobs.

The module-level ``PRODUCTION`` preset is the reference configuration for
overnight runs; individual scripts may override specific fields via
``dataclasses.replace``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln


@dataclass
class ModelParams:
    """Parameters of the regime-switching insider-detection SSM (§5).

    NaN defaults for the four variance parameters are intentional — they must
    be set either via ``warm_start`` (recommended) or explicitly before running
    inference. Running the sampler with NaN params will raise immediately on
    the first Kalman update.

    Two *model modes* share every field but the insider predictor's level term
    (`z_logit_level`), selected by the `anonymous` flag:

      * **wallet** (default, Polymarket): the level is the per-wallet
        ``logit(theta_w[w])``, shrunk by the Beta(a, b) hierarchy;
      * **anonymous** (Kalshi): the public trade feed carries no per-account
        identifier, so there is no ``theta_w`` to anchor on and the level is the
        single per-market intercept `alpha`, estimated in the IRLS M-step.
    """

    # Regime-switched process variances
    sigma2_0: float = float("nan")  # calm regime; set via warm_start
    sigma2_1: float = float("nan")  # news regime; set via warm_start

    # Volatility regime transition probabilities (decision #10)
    q_01: float = 0.05  # calm → news
    q_10: float = 0.50  # news → calm

    # Insider indicator logistic coefficients (decision #10)
    beta_S: float = 0.0  # trade size effect
    beta_Z: float = 0.0  # insider persistence

    # Observation noise variances
    tau2_0: float = float("nan")  # uninformed; set via warm_start
    tau2_1: float = float("nan")  # informed; set via warm_start

    # Beta prior on per-wallet insider propensity theta_w (decision #10)
    # Prior mean = a/(a+b) = 1/20 = 5%
    a: float = 1.0
    b: float = 19.0

    # Fixed hyperparameters (decision #12)
    gamma: float = 1.0  # size-informativeness scaling
    s0_2: float = 1.0  # initialization variance for X_{t_0}

    # ---- Model mode (see the class docstring) ----
    # A bool rather than a mode *string* on purpose: `stream_scoring` restores a
    # warm-start artifact by coercing every ModelParams field through `float`,
    # which a string would break, while `float(False)/float(True)` round-trips
    # the flag's truth value intact.
    anonymous: bool = False
    # Per-market insider intercept on the logit scale; used only when
    # `anonymous` is set. Estimated by the IRLS M-step block (it rides on
    # `estimate_betas`, so with beta estimation off it stays at this value).
    # The 0.0 default is a 50% base rate and is deliberately *not* a sensible
    # anonymous starting point — `warm_start(..., anonymous=True)` sets it to
    # the Beta(a, b) prior-mean logit, matching how wallet mode initializes
    # theta_w.
    alpha: float = 0.0

    @classmethod
    def warm_start(cls, Y: np.ndarray, *, anonymous: bool = False) -> ModelParams:
        """Moment-matched initialization from logit-price observations (§10).

        Args:
            Y: Logit-price observations pooled over the dataset.
            anonymous: Select the anonymous (no-wallet) mode, which additionally
                seeds `alpha` at the Beta(a, b) prior-mean logit — the level
                wallet mode starts every `theta_w` at. Keyword-only; the default
                reproduces the wallet-mode initialization bit for bit.

        Returns:
            Parameters with the four variances moment-matched to ``Var[Y]``.
        """
        var_Y = float(np.var(Y))
        params = cls(
            sigma2_0=0.1 * var_Y,
            sigma2_1=var_Y,
            tau2_0=var_Y,
            tau2_1=0.01 * var_Y,
            anonymous=anonymous,
        )
        if anonymous:
            base_rate = params.a / (params.a + params.b)
            params.alpha = math.log(base_rate / (1.0 - base_rate))
        return params

    def z_logit_level(self, logit_theta_w: float | np.ndarray) -> float | np.ndarray:
        """Level term of the insider logistic predictor under the active mode.

        The single mode switch the whole model shares (KTD1): the synthetic
        generator, the ADF filter, the VEM M-step and the online scorer all
        build ``logit(pi_Z) = level + beta_S * x_S~ + beta_Z * x_Z~`` and differ
        only in what ``level`` is. Passing the wallet logits through unchanged
        keeps wallet mode bit-identical; anonymous mode ignores them and returns
        the scalar per-market intercept, which broadcasts against any covariate
        shape.

        Args:
            logit_theta_w: ``logit(theta_w[w])`` for the trade(s) in question.
                Ignored in anonymous mode, where no wallet identity exists.

        Returns:
            `alpha` in anonymous mode, otherwise ``logit_theta_w`` unchanged.
        """
        return self.alpha if self.anonymous else logit_theta_w


@dataclass
class PhiPrior:
    """Single source of truth for the priors the VEM M-step optimizes against.

    Historically the M-step's prior hyperparameters were scattered as magic
    constants inside ``_vem_m_step`` (``variational_em.py``). They are lifted
    here so that inference (the M-step MAP updates), the Laplace layer
    (``src/inference/laplace.py``), and PSIS/SBC (``log p(phi)``) all consume
    *one* definition — changing a hyperparameter is a one-line, everywhere-
    consistent edit (plan ``2026-07-23-002`` KTD3/R8).

    Defaults equal the current effective M-step values, so the sequential
    inference path is behaviour-preserving:
      * process variance ``sigma2_v``: Inverse-Gamma(2, 1), MAP mode
        ``beta / (alpha + 1)`` — identical to the pre-refactor ``IG(2, 1)`` term;
      * transition ``q``: Beta(1, 1) (uniform) — identical pseudo-counts;
      * standardized ``beta_S/beta_Z``: Cauchy(0, 2.5) (Gelman et al. 2008);
      * observation variance ``tau2_z``: a *new* weak Inverse-Gamma pseudo-count
        prior. The M-step MAP is ``(SS + 2*beta) / (N + 2*alpha)``, which reduces
        *exactly* to the pre-refactor prior-free moment-match ``SS / N`` as
        ``(alpha, beta) -> 0``. The tiny defaults keep the point estimate shift
        numerically negligible (even at the small-T fixture) while giving the
        Laplace ``tau2`` block a defined, positive curvature at all sample sizes
        (the sole documented deviation from bit-exactness, plan R8).

    ``sigma2`` and ``tau2`` intentionally use different MAP algebra: ``sigma2``
    keeps the standard Inverse-Gamma mode ``beta / (alpha + 1)`` to reproduce the
    existing behaviour, whereas ``tau2`` uses the pseudo-count form so its weak
    prior does not introduce the standard mode's ``O(1/N)`` shift. ``log_prior``
    below evaluates the proper Inverse-Gamma(alpha, beta) density for both; the
    two share hyperparameters and both vanish as ``(alpha, beta) -> 0``.
    """

    # Process variance sigma2_v ~ Inverse-Gamma(alpha, beta); MAP = beta/(alpha+1).
    sigma2_ig_alpha: float = 2.0
    sigma2_ig_beta: float = 1.0
    # Observation variance tau2_z weak pseudo-count Inverse-Gamma prior; MAP =
    # (SS + 2*beta)/(N + 2*alpha) -> SS/N as (alpha, beta) -> 0. See class doc.
    # Deliberately tiny: the prior exists only to give the Laplace tau2 block a
    # defined-sign curvature (the empty-regime N_z -> 0 case is caught by the
    # Laplace fallback), so it is kept far below the point where its estimate
    # shift would perturb the inference path (<< the 1e-6 regression tolerance).
    tau2_ig_alpha: float = 1e-9
    tau2_ig_beta: float = 1e-9
    # Transition q_01/q_10 ~ Beta(a, b); Beta(1, 1) is the uniform default.
    q_beta_a: float = 1.0
    q_beta_b: float = 1.0
    # Cauchy(0, scale) weakly-informative prior on each *standardized* beta.
    beta_cauchy_scale: float = 2.5
    # Cauchy(0, scale) prior on the anonymous-mode intercept `ModelParams.alpha`
    # (unused in wallet mode, where `theta_w`'s Beta hierarchy carries the
    # level). Gelman et al. (2008) recommend a *wider* scale for an intercept
    # than for slopes because it must absorb the base rate, hence 10 rather than
    # 2.5. Note what this buys and what it costs: `alpha` is fit from one
    # market's trades with far less shrinkage than Beta(1, 19) ever applied to
    # `theta_w`, so a low-trade-count market carries an incidental-parameters
    # -style bias here (plan 2026-07-23-005 KTD2). Measured on the anonymous
    # synthetic generator (24 seeds, planted alpha = logit(0.05)), the
    # intercept's RMSE falls 0.466 -> 0.173 -> 0.113 logit across
    # T = 200/1000/3000 and its bias -0.161 -> -0.043 -> +0.012, the small-T
    # bias being under two Monte Carlo standard errors. Tightening to
    # Cauchy(0, 2.5) improved that by less than one MC standard error and not
    # at all past T = 1000, so the wider scale was kept — it is also the one
    # that stays honest for base rates rarer than 5%, where a 2.5 scale would
    # shrink the level hard toward zero. See
    # `tests/test_variational_em.py::test_anonymous_alpha_bias_shrinks_with_T`.
    # No hierarchical pooling across markets is built.
    alpha_cauchy_scale: float = 10.0

    def sigma2_map(self, ss: float, n: float) -> float:
        """Inverse-Gamma MAP mode for a process variance from its E-step stats.

        Args:
            ss: Weighted sum of squared standardized process residuals for the
                regime (the ``SS_v`` sufficient statistic).
            n: Effective count of trades assigned to the regime (``N_v``).

        Returns:
            The MAP estimate ``(beta + SS/2) / (alpha + N/2 + 1)``.
        """
        return (self.sigma2_ig_beta + ss / 2.0) / (
            self.sigma2_ig_alpha + n / 2.0 + 1.0
        )

    def tau2_map(self, ss: float, n: float) -> float:
        """Pseudo-count Inverse-Gamma MAP for an observation variance.

        The pseudo-count form ``(SS + 2*beta) / (N + 2*alpha)`` reduces to the
        pre-refactor moment-match ``SS / N`` as ``(alpha, beta) -> 0``, so the
        weak default prior perturbs the estimate only negligibly while still
        regularizing the ``N -> 0`` (empty regime) case away from a divide-by-zero.

        Args:
            ss: Weighted sum of squared observation residuals for the regime
                (the ``SS_z`` sufficient statistic, already scaled by ``denom_t``).
            n: Effective count of trades assigned to the regime (``N_z``).

        Returns:
            The MAP estimate ``(SS + 2*beta) / (N + 2*alpha)``.
        """
        return (ss + 2.0 * self.tau2_ig_beta) / (n + 2.0 * self.tau2_ig_alpha)

    def q_map(self, n_switch: float, n_stay: float) -> float:
        """Posterior-mean Beta update for a transition probability.

        Args:
            n_switch: Expected count of regime switches out of the source state
                (``n_01`` for ``q_01``; ``n_10`` for ``q_10``).
            n_stay: Expected count of stays in the source state (``n_00``; ``n_11``).

        Returns:
            The posterior mean ``(a + n_switch) / (a + b + n_switch + n_stay)``.
        """
        return (self.q_beta_a + n_switch) / (
            self.q_beta_a + self.q_beta_b + n_switch + n_stay
        )

    @staticmethod
    def _invgamma_logpdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """Log Inverse-Gamma(alpha, beta) density, elementwise (natural scale)."""
        return (
            alpha * np.log(beta)
            - gammaln(alpha)
            - (alpha + 1.0) * np.log(x)
            - beta / x
        )

    @staticmethod
    def _beta_logpdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Log Beta(a, b) density, elementwise; Beta(1, 1) is identically 0."""
        log_beta_fn = gammaln(a) + gammaln(b) - gammaln(a + b)
        return (a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x) - log_beta_fn

    @staticmethod
    def _cauchy_logpdf(x: np.ndarray, scale: float) -> np.ndarray:
        """Log Cauchy(0, scale) density, elementwise."""
        return -np.log(np.pi * scale) - np.log1p((x / scale) ** 2)

    def log_prior(self, phi: np.ndarray) -> np.ndarray:
        """Log model-prior density ``log p(phi)`` on the *constrained* scale.

        Evaluates the sum of independent block priors at the natural-scale
        parameter vector, in the canonical order
        ``(sigma2_0, sigma2_1, q_01, q_10, beta_S, beta_Z, tau2_0, tau2_1)``.
        Usable as PSIS's ``log p(phi_s)`` term; the Laplace posterior carries the
        change-of-variables Jacobian so both densities live on the same space
        (see ``src/inference/laplace.py``).

        Args:
            phi: Constrained parameter vector(s), shape ``(..., 8)`` in the
                canonical order above. Variances must be positive, ``q`` in
                ``(0, 1)``, betas real.

        Returns:
            ``log p(phi)`` with the trailing length-8 axis reduced, shape
            ``phi.shape[:-1]``.
        """
        phi = np.asarray(phi, dtype=float)
        s_a, s_b = self.sigma2_ig_alpha, self.sigma2_ig_beta
        t_a, t_b = self.tau2_ig_alpha, self.tau2_ig_beta
        q_a, q_b = self.q_beta_a, self.q_beta_b
        c = self.beta_cauchy_scale
        return (
            self._invgamma_logpdf(phi[..., 0], s_a, s_b)
            + self._invgamma_logpdf(phi[..., 1], s_a, s_b)
            + self._beta_logpdf(phi[..., 2], q_a, q_b)
            + self._beta_logpdf(phi[..., 3], q_a, q_b)
            + self._cauchy_logpdf(phi[..., 4], c)
            + self._cauchy_logpdf(phi[..., 5], c)
            + self._invgamma_logpdf(phi[..., 6], t_a, t_b)
            + self._invgamma_logpdf(phi[..., 7], t_a, t_b)
        )


# Cap on OnlineScorerConfig.effective_window. Keeps `forgetting = 1.0` (the
# frozen no-adaptation anchor) from producing an infinite seed weight; any value
# far above a realistic market's trade count works, since at that setting the
# seeded statistics are never consumed.
_MAX_EFFECTIVE_WINDOW = 1e6


@dataclass
class OnlineScorerConfig:
    """Forgetting / learning-rate schedule for `OnlineScorer` (§6, live scoring).

    `src.inference.online_scorer.OnlineScorer` adapts the model parameters with
    Cappé & Moulines (2009) online-EM sufficient-statistic recursions

        S_t = (1 - rho_t) * S_{t-1} + s(trade_t)

    driven by the learning rate ``rho_t`` this config defines. Two schedules:

      * ``"fixed"``: ``rho_t = 1 - forgetting`` — an exponential forgetting
        factor. The statistics summarize an effective window of
        ``1 / (1 - forgetting)`` trades, so the estimator tracks a *drifting*
        parameter forever rather than converging. This is the live-trading
        setting.
      * ``"robbins_monro"``: ``rho_t = (t + rho_t0) ** -rho_alpha`` — a
        decreasing rate satisfying the Robbins-Monro conditions (sum rho = inf,
        sum rho^2 < inf) for ``rho_alpha`` in ``(0.5, 1]`` and any finite
        ``rho_t0``, so the estimator *converges* to the batch fixed point on a
        stationary stream. This is the setting for offline replay/validation
        against batch VEM.

    ``forgetting = 1.0`` (the "fixed" schedule with ``rho_t == 0``) is the
    degenerate no-adaptation case and is load-bearing: combined with
    ``n_refresh = None`` it makes `OnlineScorer` a bare, frozen-parameter
    `ADFFilter`, which is the regression anchor the online path is tested
    against.

    Attributes:
        forgetting: ``lambda`` in ``(0, 1]``. Also sets the pseudo-count weight
            the initial parameters are seeded with (``1 / (1 - lambda)``
            pseudo-trades, capped), so adaptation starts *at* the supplied
            batch fit instead of at the prior.
        n_refresh: Trades between decayed-IRLS refreshes of
            ``beta_S``/``beta_Z``. ``None`` (default) or a non-positive value
            never refreshes them — matching `variational_em`'s
            ``estimate_betas=False`` default, and for the same reason (the ADF
            E-step does not identify ``Z`` on the synthetic generator; see
            ARCHITECTURE.md §6.2).
        rho_schedule: ``"fixed"`` or ``"robbins_monro"``; see above.
        rho_alpha: Exponent of the Robbins-Monro schedule; must lie in
            ``(0.5, 1]``. Ignored by the ``"fixed"`` schedule.
        beta_window: Number of most-recent trades the beta refresh refits on.
            ``None`` uses `effective_window`, so the beta block forgets on the
            same timescale as the variance/transition blocks.
        rho_t0: Offset in the Robbins-Monro rate ``(t + rho_t0) ** -rho_alpha``.
            *Not* a pseudo-trade count: under the decayed-*sum* recursion the
            rate alone fixes the quasi-stationary accumulated mass at roughly
            ``(t + rho_t0) ** rho_alpha``, so the default (50, ``rho_alpha =
            0.6``) starts the schedule as if about 10.5 trades of mass were in
            hand at ``t = 0``, not 50. What ``rho_t0`` actually buys is a gentle
            *first* decay: it caps ``rho_0`` at ``rho_t0 ** -rho_alpha < 1`` so
            trade 0 keeps most of the statistics `OnlineScorer` was seeded with
            (measured: a 50-pseudo-trade seed decays to ~42 by ``t = 500`` at
            the defaults, rather than being erased on trade 1). Must be at least
            2 — at ``rho_t0 = 1`` the rate is exactly ``1.0`` and the decay
            factor ``1 - rho_0`` is ``0``, and just above 1 it is small enough
            (``~6e-5`` at ``rho_t0 = 1.0001``) that the seed is wiped in all but
            name, so the scorer would jump to the prior instead of starting at
            the fit it was handed. Deliberately a constant rather than
            `effective_window`, which degenerates to the
            `_MAX_EFFECTIVE_WINDOW` seed-weight cap at ``forgetting = 1.0`` and
            would freeze adaptation for a million trades. Ignored by the
            ``"fixed"`` schedule.

    Reference: Cappé, O. & Moulines, E. (2009) "On-line
    expectation-maximization algorithm for latent data models", JRSS-B 71(3).
    """

    forgetting: float = 0.98
    n_refresh: int | None = None
    rho_schedule: str = "fixed"
    rho_alpha: float = 0.6
    beta_window: int | None = None
    rho_t0: float = 50.0

    def __post_init__(self) -> None:
        """Reject schedules that would silently produce a non-convergent rate."""
        if not 0.0 < self.forgetting <= 1.0:
            raise ValueError(f"forgetting must be in (0, 1]; got {self.forgetting}")
        if self.rho_schedule not in ("fixed", "robbins_monro"):
            raise ValueError(
                "rho_schedule must be 'fixed' or 'robbins_monro'; got "
                f"{self.rho_schedule!r}"
            )
        # Outside (0.5, 1] the Robbins-Monro conditions fail: alpha <= 0.5
        # leaves sum rho_t^2 divergent (the estimate never settles), alpha > 1
        # leaves sum rho_t finite (the estimate freezes short of the optimum).
        if not 0.5 < self.rho_alpha <= 1.0:
            raise ValueError(f"rho_alpha must be in (0.5, 1]; got {self.rho_alpha}")
        # rho_t0 <= 1 makes rho_0 >= 1, i.e. a first-trade decay factor of zero
        # (or negative): the seeded statistics would be discarded unread. The
        # bound is 2 rather than a bare "> 1" because the interval just above 1
        # is no better in practice - at rho_t0 = 1.0001 the surviving fraction
        # 1 - rho_0 is ~6e-5, which erases the seed as surely as zero does.
        if self.rho_t0 < 2.0:
            raise ValueError(f"rho_t0 must be >= 2; got {self.rho_t0}")
        if self.beta_window is not None and self.beta_window < 2:
            raise ValueError(f"beta_window must be >= 2; got {self.beta_window}")

    @property
    def effective_window(self) -> float:
        """Effective number of trades the decayed statistics summarize.

        The exponential window ``1 / (1 - forgetting)``, capped at
        `_MAX_EFFECTIVE_WINDOW` so ``forgetting = 1.0`` (no forgetting) stays
        finite — it is used only as a seed weight there, since that setting
        disables adaptation entirely.
        """
        return min(1.0 / (1.0 - self.forgetting + 1e-12), _MAX_EFFECTIVE_WINDOW)

    def rho(self, t: int) -> float:
        """Learning rate for the trade at 0-based stream position ``t``.

        Args:
            t: Number of trades already consumed, i.e. the incoming trade's
                0-based index in the stream.

        Returns:
            ``rho_t`` in ``[0, 1)``; exactly ``0.0`` only in the frozen
            ``forgetting = 1.0`` / ``"fixed"`` case. Never exactly ``1.0``,
            which would wipe the accumulated statistics (see `rho_t0`).
        """
        if self.rho_schedule == "fixed":
            return 1.0 - self.forgetting
        return float((t + self.rho_t0) ** -self.rho_alpha)


@dataclass
class InferenceConfig:
    """Particle filter and iPMCMC tuning knobs (§6, decisions #5 and #7).

    ``N``, ``n_iter``, and ``n_burnin`` drive the primary speed/quality
    trade-off. Prefer the named presets from ARCHITECTURE.md §10 (dev /
    half-prod / prod) over setting these by hand.
    """

    # Particle filter (decision #7)
    N: int = 50  # particles per chain (50 dev, 500 final)
    ess_resample_threshold: float = 0.5  # resample when ESS < threshold * N

    # iPMCMC chain configuration (decision #5)
    M: int = 8  # total chains
    P: int = 4  # conditional chains; M - P unconditional

    # MCMC schedule (decision #9)
    n_iter: int = 200  # total iterations (200 dev, 3000 final)
    n_burnin: int = 50  # burn-in to discard (50 dev, 500 final)

    # MH step sizes on natural / log scale (decision #11)
    mh_step_beta_S: float = 0.1
    mh_step_beta_Z: float = 0.1
    # RWMH on logit(theta_w); ~0.5 targets ~30% acceptance per wallet in pilot runs
    mh_step_logit_theta_w: float = 0.5
    mh_step_log_tau2_0: float = 0.3
    mh_step_log_tau2_1: float = 0.3

    # Diagnostics thresholds (decision #13)
    rhat_threshold: float = 1.01
    degeneracy_threshold: float = 0.25  # flag if particle ESS < threshold * N

    # joblib parallelism over K markets; 1 = sequential (reproducible); -1 = all CPUs
    n_jobs: int = 1

    # Reproducibility
    seed: int = 42

    @property
    def n_unconditional(self) -> int:
        """Number of unconditional chains (M - P)."""
        return self.M - self.P


# Ready-to-use production config — swap in for overnight runs
PRODUCTION = InferenceConfig(N=500, n_iter=3000, n_burnin=500)
