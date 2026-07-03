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
