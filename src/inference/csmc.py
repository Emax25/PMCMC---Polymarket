"""Conditional Sequential Monte Carlo with the locally-optimal proposal.

CSMC is the inner engine of Particle Gibbs: one trajectory (V_ref, Z_ref) is
pinned at a fixed particle slot for the whole pass, while the remaining N-1
particles run a standard SMC update. Drawing a new path from the resulting
ensemble leaves the posterior invariant — that's PG.

Per decision #2, the proposal for (V_i, Z_i) is the locally-optimal Bayes
proposal: at each step we enumerate the four joint values, weight by
p(V_i, Z_i | prev) * p(Y_i | V_i, Z_i, ...), normalize, and sample. This
gives much lower weight variance than the bootstrap prior — and it costs
just four vectorized `kalman_step` calls per time step.

Per decision #4, naive index-pinning is used (slot 0 always holds the
reference); ancestor sampling for the reference is explicitly deferred to
iPMCMC.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from config.default_params import InferenceConfig, ModelParams
from src.inference.kalman import kalman_step
from src.inference.smc import SMCOutput
from src.utils.transforms import ess, log1pexp, logit, systematic_resample

REFERENCE_INDEX = 0


def conditional_smc(
    Y: np.ndarray,
    delta: np.ndarray,
    log_size_ratio: np.ndarray,
    wallet_ids: np.ndarray,
    theta_w: np.ndarray,
    params: ModelParams,
    config: InferenceConfig,
    V_ref: np.ndarray,
    Z_ref: np.ndarray,
    *,
    rng: np.random.Generator,
) -> SMCOutput:
    """Run CSMC with locally-optimal proposal and index-0 reference pinning."""
    T = len(Y)
    N = config.N

    if len(V_ref) != T or len(Z_ref) != T:
        raise ValueError("V_ref and Z_ref must have length T.")
    if int(Z_ref[0]) != 0:
        raise ValueError("Z_ref[0] must be 0 (model convention).")
    V_ref = np.asarray(V_ref, dtype=np.int8)
    Z_ref = np.asarray(Z_ref, dtype=np.int8)
    if not (set(np.unique(V_ref)).issubset({0, 1})
            and set(np.unique(Z_ref)).issubset({0, 1})):
        raise ValueError("Reference trajectories must be binary {0, 1}.")

    # Incoming particle state at step i (state at end of step i-1, post-resample)
    V_prev = np.zeros(N, dtype=np.int8)
    Z_prev = np.zeros(N, dtype=np.int8)
    mu = np.zeros(N)
    sigma2 = np.full(N, params.s0_2)
    log_W = np.full(N, -np.log(N))

    # Diagnostics + history
    ess_per_step = np.empty(T)
    X_filt = np.empty(T)
    V_prob_filt = np.empty(T)
    Z_prob_filt = np.empty(T)
    log_marginal = 0.0
    resample_steps: list[int] = []
    V_hist = np.empty((T, N), dtype=np.int8)
    Z_hist = np.empty((T, N), dtype=np.int8)
    mu_hist = np.empty((T, N))
    ancestors = np.empty((T, N), dtype=np.int32)
    ancestors[0] = np.arange(N)

    logit_theta = logit(theta_w)
    denom_q = params.q_01 + params.q_10
    rho_V = params.q_01 / denom_q if denom_q > 0 else 0.5

    for i in range(T):
        # ---------- log p(V_i = v | V_prev), shape (N, 2) ----------
        if i == 0:
            log_p_V = np.broadcast_to(
                np.array([np.log1p(-rho_V), np.log(rho_V) if rho_V > 0 else -np.inf]),
                (N, 2),
            )
        else:
            row_calm = np.array([np.log1p(-params.q_01),
                                 np.log(params.q_01) if params.q_01 > 0 else -np.inf])
            row_news = np.array([np.log(params.q_10) if params.q_10 > 0 else -np.inf,
                                 np.log1p(-params.q_10)])
            log_p_V = np.where(V_prev[:, None] == 0, row_calm[None, :], row_news[None, :])

        # ---------- log p(Z_i = z | Z_prev, w, S, theta), shape (N, 2) ----------
        if i == 0:
            log_p_Z = np.broadcast_to(np.array([0.0, -np.inf]), (N, 2))
        else:
            logit_pi_Z = (
                logit_theta[int(wallet_ids[i])]
                + params.beta_S * float(log_size_ratio[i])
                + params.beta_Z * Z_prev.astype(float)
            )                                                   # (N,)
            lp = log1pexp(logit_pi_Z)
            log_p_Z = np.stack([-lp, logit_pi_Z - lp], axis=1)  # (N, 2)

        # log_prior_joint[n, 2v+z] = log_p_V[n, v] + log_p_Z[n, z]   shape (N, 4)
        log_prior_joint = (log_p_V[:, :, None] + log_p_Z[:, None, :]).reshape(N, 4)

        # ---------- One vectorized kalman_step per (v, z) combo ----------
        log_lik = np.empty((N, 4))
        mu_combos = np.empty((N, 4))
        sigma2_combos = np.empty((N, 4))
        for v in (0, 1):
            for z in (0, 1):
                k = 2 * v + z
                mu_k, s2_k, ll_k = kalman_step(
                    mu, sigma2, float(Y[i]),
                    np.full(N, v, dtype=np.int8),
                    np.full(N, z, dtype=np.int8),
                    float(delta[i]), float(log_size_ratio[i]),
                    params,
                )
                log_lik[:, k] = ll_k
                mu_combos[:, k] = mu_k
                sigma2_combos[:, k] = s2_k

        # ---------- Locally-optimal proposal + per-particle normalizing constant ----------
        log_joint = log_prior_joint + log_lik                   # (N, 4)
        log_Z_per = logsumexp(log_joint, axis=1)                # (N,)
        q = np.exp(log_joint - log_Z_per[:, None])              # (N, 4)

        # Sample outcome ∈ {0,1,2,3} per particle (categorical from q)
        cum_q = np.cumsum(q, axis=1)
        cum_q[:, -1] = 1.0                                       # float-overshoot guard
        u = rng.random(N)
        outcomes = (cum_q < u[:, None]).sum(axis=1).astype(np.int32)

        # Pin the reference particle
        outcomes[REFERENCE_INDEX] = 2 * int(V_ref[i]) + int(Z_ref[i])

        V_new = (outcomes // 2).astype(np.int8)
        Z_new = (outcomes % 2).astype(np.int8)
        row = np.arange(N)
        mu_new = mu_combos[row, outcomes]
        sigma2_new = sigma2_combos[row, outcomes]

        # ---------- Weight update + marginal-likelihood contribution ----------
        log_W_unnorm = log_W + log_Z_per
        log_Z_step = float(logsumexp(log_W_unnorm))
        log_marginal += log_Z_step
        log_W = log_W_unnorm - log_Z_step
        W = np.exp(log_W)
        W /= W.sum()  # guard against ~1e-16 logsumexp roundoff

        V_hist[i] = V_new
        Z_hist[i] = Z_new
        mu_hist[i] = mu_new
        X_filt[i] = float(W @ mu_new)
        # Clip the indicator probabilities against ~1e-16 roundoff in W·V
        V_prob_filt[i] = min(1.0, max(0.0, float(W @ V_new.astype(float))))
        Z_prob_filt[i] = min(1.0, max(0.0, float(W @ Z_new.astype(float))))
        ess_per_step[i] = ess(log_W)

        # ---------- Adaptive resample, reference slot pinned ----------
        if ess_per_step[i] < config.ess_resample_threshold * N:
            a = systematic_resample(log_W, rng)
            a[REFERENCE_INDEX] = REFERENCE_INDEX
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
        X_filt=X_filt, V_prob_filt=V_prob_filt, Z_prob_filt=Z_prob_filt,
        final_V=V_prev, final_Z=Z_prev,
        final_mu=mu, final_sigma2=sigma2, final_log_W=log_W,
        V_hist=V_hist, Z_hist=Z_hist, mu_hist=mu_hist, ancestors=ancestors,
    )
