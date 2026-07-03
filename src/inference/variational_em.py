"""Variational EM (ADF + moment-matched M-step) for the switching SSM.

Replaces CSMC Monte Carlo with deterministic assumed-density filtering:
one forward pass per market per EM iteration, O(4T) per market vs O(NT)
for CSMC with N particles. Suitable for fast wallet-ranking when exact
posterior uncertainty is not required.

E-step: single-mode ADF forward pass - 4 (V,Z) combos share one incoming
        Kalman state (the mixture mean/variance from the previous step).
        This collapses the path structure but keeps the algorithm O(T).
M-step: conjugate Beta updates for theta_w and q-transitions; IG MAP for
        sigma2; moment-matched update for tau2. beta_S, beta_Z held fixed.

Reference: Ghahramani & Hinton (2000) "Variational Learning for Switching
State-Space Models"; also known as Assumed Density Filtering for SSMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from config.default_params import InferenceConfig, ModelParams
from src.inference.kalman import _kalman_step_all_combos
from src.inference.particle_gibbs import MarketData
from src.utils.transforms import log1pexp, logit

if TYPE_CHECKING:
    pass


@dataclass
class VEMOutput:
    """Output of variational_em: deterministic posterior summaries + fitted params."""

    params: ModelParams
    theta_w: np.ndarray          # (n_wallets,) posterior mean of theta_w
    Z_prob: list[np.ndarray]     # per-market (T_k,) q(Z_t=1) = filter-marginal P(Z_t=1|Y)
    V_prob: list[np.ndarray]     # per-market (T_k,) q(V_t=1) = filter-marginal P(V_t=1|Y)
    X_mean: list[np.ndarray]     # per-market (T_k,) mixed E[X_t | Y_{0:t}]
    elbo_trace: np.ndarray       # (n_iter_run,) log-marginal per EM iteration (proxy for ELBO)
    n_iter_run: int              # actual EM iterations completed


def _vem_e_step(
    Y: np.ndarray,
    delta: np.ndarray,
    log_size_ratio: np.ndarray,
    wallet_ids: np.ndarray,
    theta_w: np.ndarray,
    params: ModelParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Single-mode ADF forward pass for one market.

    Args:
        Y: (T,) logit-price observations.
        delta: (T,) inter-trade times; delta[0] = 0.
        log_size_ratio: (T,) log(S/S_bar) features.
        wallet_ids: (T,) integer wallet index per trade.
        theta_w: (n_wallets,) current per-wallet propensity estimates.
        params: Current model parameters.

    Returns:
        q_vz: (T, 4) soft (V_t, Z_t) assignments — q_vz[t, k] = q(V_t=v, Z_t=z)
              where k = 2*v + z.
        mu_filt: (T,) mixed E[X_t | Y_{0:t}].
        sigma2_filt: (T,) mixed Var[X_t | Y_{0:t}].
        log_marginal: scalar approximate log p(Y | params, theta_w).
    """
    T = len(Y)
    logit_theta = logit(theta_w)
    q_01 = params.q_01
    q_10 = params.q_10

    denom_q = q_01 + q_10
    rho_V = q_01 / denom_q if denom_q > 0 else 0.5

    mu = np.zeros(1)
    sigma2 = np.array([params.s0_2])

    q_vz = np.empty((T, 4))
    mu_filt = np.empty(T)
    sigma2_filt = np.empty(T)
    log_marginal = 0.0

    prev_q_V = np.array([1.0 - rho_V, rho_V])
    prev_E_Z = 0.0

    for t in range(T):
        if t == 0:
            log_p_V = np.array(
                [
                    np.log(max(1.0 - rho_V, 1e-300)),
                    np.log(max(rho_V, 1e-300)),
                ]
            )
            log_p_Z = np.array([0.0, -500.0])
        else:
            p_V0 = prev_q_V[0] * (1.0 - q_01) + prev_q_V[1] * q_10
            p_V1 = prev_q_V[0] * q_01 + prev_q_V[1] * (1.0 - q_10)
            log_p_V = np.array(
                [np.log(max(p_V0, 1e-300)), np.log(max(p_V1, 1e-300))]
            )
            logit_pi = (
                float(logit_theta[int(wallet_ids[t])])
                + params.beta_S * float(log_size_ratio[t])
                + params.beta_Z * prev_E_Z
            )
            lp = float(log1pexp(logit_pi))
            log_p_Z = np.array([-lp, logit_pi - lp])

        log_prior_joint = (log_p_V[:, None] + log_p_Z[None, :]).reshape(4)

        mu_combos, sigma2_combos, log_lik = _kalman_step_all_combos(
            mu,
            sigma2,
            float(Y[t]),
            float(delta[t]),
            float(log_size_ratio[t]),
            params.sigma2_0,
            params.sigma2_1,
            params.tau2_0,
            params.tau2_1,
            params.gamma,
        )

        log_joint = log_prior_joint + log_lik[0]
        log_Z_t = float(logsumexp(log_joint))
        log_marginal += log_Z_t
        q_t = np.exp(log_joint - log_Z_t)
        q_vz[t] = q_t

        mu_c = mu_combos[0]
        sigma2_c = sigma2_combos[0]
        mu_mixed = float(q_t @ mu_c)
        sigma2_mixed = float(q_t @ (sigma2_c + (mu_c - mu_mixed) ** 2))
        mu_filt[t] = mu_mixed
        sigma2_filt[t] = sigma2_mixed

        mu = np.array([mu_mixed])
        sigma2 = np.array([sigma2_mixed])
        prev_q_V = np.array([q_t[0] + q_t[1], q_t[2] + q_t[3]])
        prev_E_Z = float(q_t[1] + q_t[3])

    return q_vz, mu_filt, sigma2_filt, log_marginal


def _vem_m_step(
    markets: list[MarketData],
    q_vz_list: list[np.ndarray],
    mu_filt_list: list[np.ndarray],
    sigma2_filt_list: list[np.ndarray],
    params: ModelParams,
    theta_w: np.ndarray,
    n_wallets: int,
) -> tuple[ModelParams, np.ndarray]:
    """Moment-matched M-step: update params and theta_w from soft assignments.

    Args:
        markets: Input market data.
        q_vz_list: Per-market soft (V,Z) assignments, each (T_k, 4).
        mu_filt_list: Per-market mixed Kalman means, each (T_k,).
        sigma2_filt_list: Per-market mixed Kalman variances, each (T_k,).
        params: Current model parameters.
        theta_w: Current per-wallet propensity estimates.
        n_wallets: Total wallet count.

    Returns:
        Updated (params, theta_w).
    """
    from dataclasses import replace

    # ---- theta_w: Beta conjugate update ----
    alpha_w = np.full(n_wallets, params.a)
    beta_w = np.full(n_wallets, params.b)
    for md, q_vz in zip(markets, q_vz_list):
        E_Z = q_vz[:, 1] + q_vz[:, 3]
        for t in range(1, len(md.Y)):
            w = int(md.wallet_ids[t])
            alpha_w[w] += E_Z[t]
            beta_w[w] += 1.0 - E_Z[t]
    theta_w_new = alpha_w / (alpha_w + beta_w)

    # ---- q_01, q_10: product-of-marginals Beta update ----
    a_prior = b_prior = 1.0
    n_00 = n_01 = n_10 = n_11 = 0.0
    for q_vz in q_vz_list:
        q_V0 = q_vz[:, 0] + q_vz[:, 1]
        q_V1 = q_vz[:, 2] + q_vz[:, 3]
        n_00 += float((q_V0[:-1] * q_V0[1:]).sum())
        n_01 += float((q_V0[:-1] * q_V1[1:]).sum())
        n_10 += float((q_V1[:-1] * q_V0[1:]).sum())
        n_11 += float((q_V1[:-1] * q_V1[1:]).sum())
    q_01_new = (a_prior + n_01) / (2 * a_prior + n_01 + n_00)
    q_10_new = (a_prior + n_10) / (2 * a_prior + n_10 + n_11)
    # Clamp to (0,1) open interval to avoid degenerate regimes
    q_01_new = float(np.clip(q_01_new, 1e-6, 1.0 - 1e-6))
    q_10_new = float(np.clip(q_10_new, 1e-6, 1.0 - 1e-6))

    # ---- sigma2_0, sigma2_1: IG MAP update (mode = beta/(alpha+1)) ----
    alpha_prior_s = 2.0
    beta_prior_s = 1.0
    SS_v = [0.0, 0.0]
    N_v = [0.0, 0.0]
    for md, q_vz, mu_f, sigma2_f in zip(
        markets, q_vz_list, mu_filt_list, sigma2_filt_list
    ):
        dt = md.delta[1:]
        valid = dt > 0
        if not valid.any():
            continue
        resid2 = (mu_f[1:] - mu_f[:-1]) ** 2
        extra_var = sigma2_f[1:] + sigma2_f[:-1]
        for v in (0, 1):
            q_V_v = (q_vz[:, 2 * v] + q_vz[:, 2 * v + 1])[1:]
            SS_v[v] += float(
                (q_V_v[valid] * (resid2[valid] + extra_var[valid]) / dt[valid]).sum()
            )
            N_v[v] += float(q_V_v[valid].sum())
    sigma2_0_new = max(
        (beta_prior_s + SS_v[0] / 2.0) / (alpha_prior_s + N_v[0] / 2.0 + 1.0),
        1e-6,
    )
    sigma2_1_new = max(
        (beta_prior_s + SS_v[1] / 2.0) / (alpha_prior_s + N_v[1] / 2.0 + 1.0),
        1e-6,
    )
    sigma2_1_new = max(sigma2_1_new, sigma2_0_new)

    # ---- tau2_0, tau2_1: moment-matched update ----
    SS_z = [0.0, 0.0]
    N_z = [0.0, 0.0]
    for md, q_vz, mu_f, sigma2_f in zip(
        markets, q_vz_list, mu_filt_list, sigma2_filt_list
    ):
        denom_t = np.maximum(1.0 + md.log_size_ratio * params.gamma, 0.1)
        resid2 = (md.Y - mu_f) ** 2
        for z in (0, 1):
            q_Z_z = q_vz[:, z] + q_vz[:, 2 + z]
            SS_z[z] += float((q_Z_z * (resid2 + sigma2_f) * denom_t).sum())
            N_z[z] += float(q_Z_z.sum())
    tau2_0_new = max(SS_z[0] / max(N_z[0], 1e-10), 1e-6)
    tau2_1_new = max(SS_z[1] / max(N_z[1], 1e-10), 1e-6)
    # Insiders have tighter obs variance (more price-informative trades)
    tau2_1_new = min(tau2_1_new, tau2_0_new)

    new_params = replace(
        params,
        q_01=q_01_new,
        q_10=q_10_new,
        sigma2_0=sigma2_0_new,
        sigma2_1=sigma2_1_new,
        tau2_0=tau2_0_new,
        tau2_1=tau2_1_new,
    )
    return new_params, theta_w_new


def variational_em(
    markets: list[MarketData],
    config: InferenceConfig,
    *,
    n_wallets: int | None = None,
    params_init: ModelParams | None = None,
    theta_w_init: np.ndarray | None = None,
    n_iter: int = 50,
    tol: float = 1e-3,
    n_jobs: int = 1,
) -> VEMOutput:
    """Fit the switching SSM by variational EM (ADF E-step + moment-matched M-step).

    Substantially faster than Particle Gibbs: no sampling, no MCMC chains.
    Suitable for the "fast tier" wallet ranking. Approximate posteriors are
    good enough for AUC/ranking; credible intervals are not reliable.

    Args:
        markets: List of K markets; K = 1 is valid.
        config: InferenceConfig; warm-start params come from `ModelParams.warm_start`
            if `params_init` is None.
        n_wallets: Total wallet count; inferred from market data if None.
        params_init: Optional model parameter initialization.
        theta_w_init: Optional initial per-wallet propensities. Defaults to
            `Beta(a, b)` mean = a/(a+b) for all wallets.
        n_iter: Maximum EM iterations.
        tol: Convergence tolerance on the relative change in log-marginal.
        n_jobs: Reserved for future joblib parallelism over markets; currently
            always sequential.

    Returns:
        VEMOutput with fitted params, posterior marginals, and convergence trace.
    """
    if n_wallets is None:
        n_wallets = int(max(int(m.wallet_ids.max()) for m in markets)) + 1

    if params_init is None:
        Y_concat = np.concatenate([m.Y for m in markets])
        params = ModelParams.warm_start(Y_concat)
    else:
        params = params_init

    if theta_w_init is None:
        theta_w = np.full(n_wallets, params.a / (params.a + params.b))
    else:
        theta_w = np.array(theta_w_init, copy=True)

    elbo_trace: list[float] = []
    prev_lm = float("-inf")

    q_vz_list: list[np.ndarray] = []
    mu_filt_list: list[np.ndarray] = []
    sigma2_filt_list: list[np.ndarray] = []

    for em_it in range(n_iter):
        # ---- E-step ----
        q_vz_list = []
        mu_filt_list = []
        sigma2_filt_list = []
        total_lm = 0.0
        for md in markets:
            q_vz, mu_f, sigma2_f, lm = _vem_e_step(
                md.Y,
                md.delta,
                md.log_size_ratio,
                md.wallet_ids,
                theta_w,
                params,
            )
            q_vz_list.append(q_vz)
            mu_filt_list.append(mu_f)
            sigma2_filt_list.append(sigma2_f)
            total_lm += lm

        elbo_trace.append(total_lm)

        # ---- Convergence check ----
        if em_it > 0:
            denom = max(abs(prev_lm), 1.0)
            if abs(total_lm - prev_lm) / denom < tol:
                break
        prev_lm = total_lm

        # ---- M-step ----
        params, theta_w = _vem_m_step(
            markets,
            q_vz_list,
            mu_filt_list,
            sigma2_filt_list,
            params,
            theta_w,
            n_wallets,
        )

    n_iter_run = len(elbo_trace)

    # Final posterior summaries
    Z_prob_list = [q_vz[:, 1] + q_vz[:, 3] for q_vz in q_vz_list]
    V_prob_list = [q_vz[:, 2] + q_vz[:, 3] for q_vz in q_vz_list]

    return VEMOutput(
        params=params,
        theta_w=theta_w,
        Z_prob=Z_prob_list,
        V_prob=V_prob_list,
        X_mean=mu_filt_list,
        elbo_trace=np.asarray(elbo_trace),
        n_iter_run=n_iter_run,
    )
