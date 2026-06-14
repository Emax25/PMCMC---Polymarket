"""Conjugate Gibbs and Metropolis-Hastings updates for the model parameters.

Decision #6 splits the parameter block into:
  * conjugate Gibbs for sigma2_0, sigma2_1, q_01, q_10, theta_w
  * random-walk MH for beta_S, beta_Z, tau2_0, tau2_1

Decision #8 pools sufficient statistics across markets — every update accepts a
list of `MarketLatents` so single-market inference is just K = 1.

The theta_w update is Beta-Bernoulli treating Z_i | theta_{w_i} as a direct
Bernoulli (exact when beta_S = beta_Z = 0; approximate otherwise — this
matches the README framing).

The tau2 update uses a Jeffreys prior p(tau2) ∝ 1/tau2: on the log-scale this
is uniform, and the log-normal proposal's Jacobian τ*/τ exactly cancels the
prior ratio, leaving the acceptance probability equal to the bare likelihood
ratio.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from config.default_params import InferenceConfig, ModelParams
from src.utils.transforms import log1pexp, logit


@dataclass
class MarketLatents:
    """One market's observations plus the current sampled latent state."""
    Y: np.ndarray
    delta: np.ndarray
    log_size_ratio: np.ndarray
    wallet_ids: np.ndarray
    X: np.ndarray
    V: np.ndarray
    Z: np.ndarray


# ---------------- Conjugate updates ----------------

def update_sigma2(
    markets: list[MarketLatents],
    rng: np.random.Generator,
    *,
    alpha_prior: float = 2.0,
    beta_prior: float = 1.0,
) -> tuple[float, float]:
    """Inverse-Gamma conjugate update for (sigma2_0, sigma2_1).

    Posterior given X, V increments:
        sigma2_v | . ~ Inv-Gamma(alpha_prior + N_v/2, beta_prior + SS_v/2)
    where N_v = #{i >= 1 : V_i = v, Δ_i > 0} and
          SS_v = Σ (X_i - X_{i-1})^2 / Δ_i over those i.

    Steps with Δ_i = 0 (same-second trades on real data) are dropped: the
    model says X_i = X_{i-1} deterministically there, so they carry no
    information about σ² — and the division by zero would otherwise corrupt
    the posterior.
    """
    N_v = np.zeros(2, dtype=int)
    SS_v = np.zeros(2)
    for m in markets:
        dX = np.diff(m.X)                  # X_i - X_{i-1}, i = 1..T-1
        dT = m.delta[1:]                   # Δ_i, i = 1..T-1
        V_i = m.V[1:]                       # regime at the destination step
        valid = dT > 0
        for v in (0, 1):
            mask = (V_i == v) & valid
            N_v[v] += int(mask.sum())
            if mask.any():
                SS_v[v] += float(np.sum(dX[mask] ** 2 / dT[mask]))

    alpha_post = alpha_prior + N_v / 2.0
    beta_post = beta_prior + SS_v / 2.0
    sigma2_0 = float(1.0 / rng.gamma(alpha_post[0], 1.0 / beta_post[0]))
    sigma2_1 = float(1.0 / rng.gamma(alpha_post[1], 1.0 / beta_post[1]))
    return sigma2_0, sigma2_1


def update_q(
    markets: list[MarketLatents],
    rng: np.random.Generator,
    *,
    a_prior: float = 1.0,
    b_prior: float = 1.0,
) -> tuple[float, float]:
    """Beta conjugate update for (q_01, q_10) from V transition counts.

    q_01 | . ~ Beta(a_prior + n_01, b_prior + n_00)
    q_10 | . ~ Beta(a_prior + n_10, b_prior + n_11)
    """
    n_00 = n_01 = n_10 = n_11 = 0
    for m in markets:
        V_prev = m.V[:-1]
        V_next = m.V[1:]
        n_00 += int(np.sum((V_prev == 0) & (V_next == 0)))
        n_01 += int(np.sum((V_prev == 0) & (V_next == 1)))
        n_10 += int(np.sum((V_prev == 1) & (V_next == 0)))
        n_11 += int(np.sum((V_prev == 1) & (V_next == 1)))

    q_01 = float(rng.beta(a_prior + n_01, b_prior + n_00))
    q_10 = float(rng.beta(a_prior + n_10, b_prior + n_11))
    return q_01, q_10


def update_theta_w(
    markets: list[MarketLatents],
    n_wallets: int,
    a: float,
    b: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Approximate Beta-Bernoulli Gibbs for theta_w (decision #6).

    Treats Z_i | theta_{w_i} as Bernoulli(theta_{w_i}) — exact at β=0 and a
    documented approximation otherwise. Z counts are pooled across markets
    (decision #8). Index i=0 is excluded since Z_0 := 0 is deterministic.
    """
    z_count = np.zeros(n_wallets)
    n_count = np.zeros(n_wallets)
    for m in markets:
        w_idx = np.asarray(m.wallet_ids[1:], dtype=np.int64)
        Z_i = m.Z[1:].astype(float)
        n_count += np.bincount(w_idx, minlength=n_wallets).astype(float)
        z_count += np.bincount(w_idx, weights=Z_i, minlength=n_wallets)

    return rng.beta(a + z_count, b + n_count - z_count)


# ---------------- MH updates ----------------

def _log_lik_Z(
    Z: np.ndarray,
    wallet_ids: np.ndarray,
    log_size_ratio: np.ndarray,
    logit_theta: np.ndarray,
    beta_S: float,
    beta_Z: float,
) -> float:
    """Σ_{i>=1} log Bernoulli(Z_i | sigmoid(logit θ_{w_i} + β_S log(S/S̄) + β_Z 1{Z_{i-1}=1}))."""
    Z_prev = Z[:-1]
    Z_curr = Z[1:].astype(float)
    logit_pi = (
        logit_theta[wallet_ids[1:]]
        + beta_S * log_size_ratio[1:]
        + beta_Z * (Z_prev == 1).astype(float)
    )
    # log Bernoulli = z·logit - log1pexp(logit)
    return float(np.sum(Z_curr * logit_pi - log1pexp(logit_pi)))


def update_beta(
    beta_S: float,
    beta_Z: float,
    markets: list[MarketLatents],
    theta_w: np.ndarray,
    config: InferenceConfig,
    rng: np.random.Generator,
    *,
    prior_sd: float = 10.0,
) -> tuple[float, float, bool, bool]:
    """Independent random-walk MH on β_S, then β_Z (decision #11).

    Prior: N(0, prior_sd²) on each; symmetric Gaussian proposal so the q-ratio cancels.
    """
    logit_theta = logit(theta_w)

    def log_post(bS: float, bZ: float) -> float:
        lik = sum(
            _log_lik_Z(m.Z, m.wallet_ids, m.log_size_ratio, logit_theta, bS, bZ)
            for m in markets
        )
        prior = -0.5 * (bS * bS + bZ * bZ) / (prior_sd * prior_sd)
        return lik + prior

    log_post_cur = log_post(beta_S, beta_Z)

    bS_star = beta_S + config.mh_step_beta_S * rng.standard_normal()
    log_post_star = log_post(bS_star, beta_Z)
    acc_S = bool(rng.random() < np.exp(min(0.0, log_post_star - log_post_cur)))
    if acc_S:
        beta_S = bS_star
        log_post_cur = log_post_star

    bZ_star = beta_Z + config.mh_step_beta_Z * rng.standard_normal()
    log_post_star = log_post(beta_S, bZ_star)
    acc_Z = bool(rng.random() < np.exp(min(0.0, log_post_star - log_post_cur)))
    if acc_Z:
        beta_Z = bZ_star

    return beta_S, beta_Z, acc_S, acc_Z


def _log_lik_Y(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    log_size_ratio: np.ndarray,
    tau2_0: float,
    tau2_1: float,
    gamma: float,
) -> float:
    """Σ_i log N(Y_i | X_i, tau2_{Z_i} / max(1 + γ log(S/S̄), 0.1))."""
    tau2 = np.where(Z == 0, tau2_0, tau2_1)
    denom = np.maximum(1.0 + gamma * log_size_ratio, 0.1)
    R = tau2 / denom
    resid = Y - X
    return float(np.sum(-0.5 * (np.log(2.0 * np.pi * R) + resid * resid / R)))


def update_tau2(
    tau2_0: float,
    tau2_1: float,
    markets: list[MarketLatents],
    gamma: float,
    config: InferenceConfig,
    rng: np.random.Generator,
) -> tuple[float, float, bool, bool]:
    """Log-normal random-walk MH for (tau2_0, tau2_1) with Jeffreys prior.

    Proposal: τ²* = τ² · exp(ε), ε ~ N(0, step²). Under p(τ²) ∝ 1/τ², the
    log-prior ratio -ε exactly cancels the log-normal Jacobian +ε, so the
    acceptance ratio is the bare likelihood ratio.
    """
    def log_lik(t0: float, t1: float) -> float:
        return sum(
            _log_lik_Y(m.Y, m.X, m.Z, m.log_size_ratio, t0, t1, gamma)
            for m in markets
        )

    log_lik_cur = log_lik(tau2_0, tau2_1)

    eps = config.mh_step_log_tau2_0 * rng.standard_normal()
    t0_star = tau2_0 * float(np.exp(eps))
    log_lik_star = log_lik(t0_star, tau2_1)
    acc_0 = bool(rng.random() < np.exp(min(0.0, log_lik_star - log_lik_cur)))
    if acc_0:
        tau2_0 = t0_star
        log_lik_cur = log_lik_star

    eps = config.mh_step_log_tau2_1 * rng.standard_normal()
    t1_star = tau2_1 * float(np.exp(eps))
    log_lik_star = log_lik(tau2_0, t1_star)
    acc_1 = bool(rng.random() < np.exp(min(0.0, log_lik_star - log_lik_cur)))
    if acc_1:
        tau2_1 = t1_star

    return tau2_0, tau2_1, acc_0, acc_1


# ---------------- Orchestrator ----------------

@dataclass
class GibbsSweepDiag:
    """Acceptance flags from one Gibbs sweep."""
    acc_beta_S: bool
    acc_beta_Z: bool
    acc_tau2_0: bool
    acc_tau2_1: bool


def gibbs_sweep(
    params: ModelParams,
    theta_w: np.ndarray,
    markets: list[MarketLatents],
    config: InferenceConfig,
    rng: np.random.Generator,
) -> tuple[ModelParams, np.ndarray, GibbsSweepDiag]:
    """Run one full Gibbs sweep over φ and θ_w (decision #6).

    Order (any cycle of full conditionals leaves the posterior invariant):
        1. (σ²_0, σ²_1)   — Inv-Gamma
        2. (q_01, q_10)   — Beta
        3. θ_w            — Beta-Bernoulli (pooled across markets)
        4. (β_S, β_Z)     — Gaussian random-walk MH, conditioning on θ_w_new
        5. (τ²_0, τ²_1)   — log-normal MH with Jeffreys prior
    """
    n_wallets = len(theta_w)

    sigma2_0, sigma2_1 = update_sigma2(markets, rng)
    q_01, q_10 = update_q(markets, rng)
    theta_w_new = update_theta_w(markets, n_wallets, params.a, params.b, rng)
    beta_S_new, beta_Z_new, acc_S, acc_Z = update_beta(
        params.beta_S, params.beta_Z, markets, theta_w_new, config, rng,
    )
    tau2_0_new, tau2_1_new, acc_t0, acc_t1 = update_tau2(
        params.tau2_0, params.tau2_1, markets, params.gamma, config, rng,
    )

    new_params = replace(
        params,
        sigma2_0=sigma2_0, sigma2_1=sigma2_1,
        q_01=q_01, q_10=q_10,
        beta_S=beta_S_new, beta_Z=beta_Z_new,
        tau2_0=tau2_0_new, tau2_1=tau2_1_new,
    )
    diag = GibbsSweepDiag(
        acc_beta_S=acc_S, acc_beta_Z=acc_Z,
        acc_tau2_0=acc_t0, acc_tau2_1=acc_t1,
    )
    return new_params, theta_w_new, diag
