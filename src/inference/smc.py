"""Bootstrap Sequential Monte Carlo with Rao-Blackwellized X.

Per decision #2, Phase 4 uses the bootstrap (prior) proposal for (V_i, Z_i):
each particle samples V_i from the regime Markov chain and Z_i from the
wallet/size logistic, then accepts an incremental weight equal to the
predictive density of Y_i (from `kalman_step`). Adaptive systematic
resampling (decision #3) is triggered when ESS < threshold * N.

The particle structure is RBPF: every particle carries its own (mu, sigma2)
Kalman moments alongside the discrete state (decision #1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from config.default_params import InferenceConfig, ModelParams
from src.inference.kalman import kalman_step
from src.utils.transforms import (
    bounded_indicator_probability,
    ess,
    logit,
    sigmoid,
    systematic_resample,
)


@dataclass
class SMCOutput:
    """Container for SMC diagnostics, trajectories, and terminal particles."""

    log_marginal: float  # log p_hat(Y_{1:T} | params, theta_w)
    ess_per_step: np.ndarray  # (T,) ESS after the update at step i
    resample_steps: list[int]  # indices i at which we resampled

    # Filter estimates (weighted by normalized log_W at each step)
    X_filt: np.ndarray  # (T,) E[X_i | Y_{1:i}]
    V_prob_filt: np.ndarray  # (T,) P(V_i = 1 | Y_{1:i})
    Z_prob_filt: np.ndarray  # (T,) P(Z_i = 1 | Y_{1:i})

    # Final particle ensemble
    final_V: np.ndarray  # (N,)
    final_Z: np.ndarray  # (N,)
    final_mu: np.ndarray  # (N,)
    final_sigma2: np.ndarray  # (N,)
    final_log_W: np.ndarray  # (N,) normalized log-weights

    # Full path history (post-update, pre-resample state per step)
    V_hist: np.ndarray  # (T, N) int8
    Z_hist: np.ndarray  # (T, N) int8
    mu_hist: np.ndarray  # (T, N) float
    ancestors: np.ndarray  # (T, N) int32; ancestors[i][n] = parent index at i-1


def bootstrap_smc(
    Y: np.ndarray,
    delta: np.ndarray,
    log_size_ratio: np.ndarray,
    wallet_ids: np.ndarray,
    theta_w: np.ndarray,
    params: ModelParams,
    config: InferenceConfig,
    *,
    rng: np.random.Generator,
) -> SMCOutput:
    """Run bootstrap SMC on a single market.

    Args:
        Y: (T,) logit-price observations.
        delta: (T,) inter-trade times; delta[0] = 0 by convention.
        log_size_ratio: (T,) log(S_i / S_bar).
        wallet_ids: (T,) integer wallet index per trade.
        theta_w: (n_wallets,) per-wallet insider propensities.
        params: ModelParams (must have non-NaN sigma2/tau2; use warm_start).
        config: InferenceConfig; uses N and ess_resample_threshold.
        rng: explicit Generator (§7.1).

    Returns:
        SMCOutput with log marginal estimate, filtering summaries, particle
        history, and final weighted particle ensemble.
    """
    T = len(Y)
    N = config.N

    # State for the "previous" step (incoming particles at step i)
    V_prev = np.zeros(N, dtype=np.int8)
    Z_prev = np.zeros(N, dtype=np.int8)
    mu = np.zeros(N)
    sigma2 = np.full(N, params.s0_2)
    log_W = np.full(N, -np.log(N))

    # Diagnostics + filter estimates
    ess_per_step = np.empty(T)
    X_filt = np.empty(T)
    V_prob_filt = np.empty(T)
    Z_prob_filt = np.empty(T)
    log_marginal = 0.0
    resample_steps: list[int] = []

    # Path storage
    V_hist = np.empty((T, N), dtype=np.int8)
    Z_hist = np.empty((T, N), dtype=np.int8)
    mu_hist = np.empty((T, N))
    ancestors = np.empty((T, N), dtype=np.int32)
    ancestors[0] = np.arange(N)  # no parents at step 0

    # Precomputations
    logit_theta = logit(theta_w)
    denom_q = params.q_01 + params.q_10
    rho_V = params.q_01 / denom_q if denom_q > 0 else 0.5

    for i in range(T):
        # --- Propose V_i and Z_i from the prior ---
        if i == 0:
            V_new = (rng.random(N) < rho_V).astype(np.int8)
            Z_new = np.zeros(N, dtype=np.int8)  # Z_0 := 0
        else:
            flip_prob = np.where(V_prev == 0, params.q_01, params.q_10)
            flips = rng.random(N) < flip_prob
            V_new = np.where(flips, 1 - V_prev, V_prev).astype(np.int8)

            logit_pi_Z = (
                logit_theta[int(wallet_ids[i])]
                + params.beta_S * float(log_size_ratio[i])
                + params.beta_Z * Z_prev.astype(float)
            )
            pi_Z = sigmoid(logit_pi_Z)
            Z_new = (rng.random(N) < pi_Z).astype(np.int8)

        # --- Kalman predict + update (vectorized over particles) ---
        mu_new, sigma2_new, log_lik = kalman_step(
            mu,
            sigma2,
            float(Y[i]),
            V_new,
            Z_new,
            float(delta[i]),
            float(log_size_ratio[i]),
            params,
        )

        # --- Weight update + incremental marginal-likelihood contribution ---
        log_W_unnorm = log_W + log_lik
        log_Z_step = float(logsumexp(log_W_unnorm))
        log_marginal += log_Z_step
        log_W = log_W_unnorm - log_Z_step
        W = np.exp(log_W)
        W /= W.sum()  # guard against ~1e-16 logsumexp roundoff

        # --- Record history and filter summaries ---
        V_hist[i] = V_new
        Z_hist[i] = Z_new
        mu_hist[i] = mu_new
        X_filt[i] = float(W @ mu_new)
        # Clip the indicator probabilities against ~1e-16 roundoff in W·V
        V_prob_filt[i] = bounded_indicator_probability(W, V_new)
        Z_prob_filt[i] = bounded_indicator_probability(W, Z_new)
        ess_per_step[i] = ess(log_W)

        # --- Adaptive resample (affects step i+1's incoming particles) ---
        if ess_per_step[i] < config.ess_resample_threshold * N:
            a = systematic_resample(log_W, rng)
            V_prev = V_new[a]
            Z_prev = Z_new[a]
            mu = mu_new[a]
            sigma2 = sigma2_new[a]
            log_W = np.full(N, -np.log(N))
            resample_steps.append(i)
            if i + 1 < T:
                ancestors[i + 1] = a
        else:
            V_prev = V_new
            Z_prev = Z_new
            mu = mu_new
            sigma2 = sigma2_new
            if i + 1 < T:
                ancestors[i + 1] = np.arange(N)

    return SMCOutput(
        log_marginal=log_marginal,
        ess_per_step=ess_per_step,
        resample_steps=resample_steps,
        X_filt=X_filt,
        V_prob_filt=V_prob_filt,
        Z_prob_filt=Z_prob_filt,
        final_V=V_prev,
        final_Z=Z_prev,
        final_mu=mu,
        final_sigma2=sigma2,
        final_log_W=log_W,
        V_hist=V_hist,
        Z_hist=Z_hist,
        mu_hist=mu_hist,
        ancestors=ancestors,
    )


def smooth_paths(out: SMCOutput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace surviving lineages back to compute smoothed estimates.

    For each step t, the smoothed estimate weights each per-step particle by
    the final-time weight of its descendants, after tracing through the
    ancestor table. Note: in vanilla bootstrap PF this degenerates at early
    t (few unique lineages); PG/iPMCMC are the principled fix.

    Returns (X_smooth, V_prob_smooth, Z_prob_smooth) each of shape (T,).

    Args:
        out: Output of ``bootstrap_smc`` containing history and ancestors.

    Returns:
        Tuple ``(X_smooth, V_prob_smooth, Z_prob_smooth)``, each shape ``(T,)``.
    """
    T, N = out.V_hist.shape
    lineage = np.empty((T, N), dtype=np.int32)
    lineage[T - 1] = np.arange(N)
    for t in range(T - 2, -1, -1):
        lineage[t] = out.ancestors[t + 1][lineage[t + 1]]

    W_final = np.exp(out.final_log_W)
    W_final /= W_final.sum()  # guard against float-roundoff from logsumexp
    X_smooth = np.empty(T)
    V_prob_smooth = np.empty(T)
    Z_prob_smooth = np.empty(T)
    for t in range(T):
        idx = lineage[t]
        X_smooth[t] = float(W_final @ out.mu_hist[t][idx])
        V_prob_smooth[t] = float(W_final @ out.V_hist[t][idx].astype(float))
        Z_prob_smooth[t] = float(W_final @ out.Z_hist[t][idx].astype(float))

    return X_smooth, V_prob_smooth, Z_prob_smooth


def sample_path(
    out: SMCOutput,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one (V, Z) trajectory from the SMC posterior.

    Picks a final-time particle index proportional to W_final, then walks the
    ancestor table back to t = 0. This is what Particle Gibbs uses to choose
    the next CSMC reference trajectory.

    Args:
        out: Output of ``bootstrap_smc`` containing history and ancestors.
        rng: Random generator used for final particle-index sampling.

    Returns:
        A sampled ``(V_path, Z_path)`` pair, each with shape ``(T,)``.
    """
    T, N = out.V_hist.shape
    W = np.exp(out.final_log_W)
    W /= W.sum()
    n_star = int(rng.choice(N, p=W))

    lineage = np.empty(T, dtype=np.int32)
    lineage[T - 1] = n_star
    for t in range(T - 2, -1, -1):
        lineage[t] = out.ancestors[t + 1][lineage[t + 1]]

    t_idx = np.arange(T)
    V_path = out.V_hist[t_idx, lineage].astype(np.int8)
    Z_path = out.Z_hist[t_idx, lineage].astype(np.int8)
    return V_path, Z_path
