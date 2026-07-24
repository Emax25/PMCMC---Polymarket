"""Default model hyperparameters and inference configuration for PMCMC.

Defines two dataclasses consumed throughout the codebase:

  * ``ModelParams``     — statistical model parameters (regime variances,
                          insider logistic coefficients, observation noise).
  * ``InferenceConfig`` — particle filter / iPMCMC tuning knobs.

The module-level ``PRODUCTION`` preset is the reference configuration for
overnight runs; individual scripts may override specific fields via
``dataclasses.replace``.
"""

from __future__ import annotations

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

    @classmethod
    def warm_start(cls, Y: np.ndarray) -> ModelParams:
        """Moment-matched initialization from logit-price observations (§10)."""
        var_Y = float(np.var(Y))
        return cls(
            sigma2_0=0.1 * var_Y,
            sigma2_1=var_Y,
            tau2_0=var_Y,
            tau2_1=0.01 * var_Y,
        )


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
