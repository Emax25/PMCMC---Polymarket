"""Kalman filter for X | V, Z — Rao-Blackwellized SMC core.

Conditional on the discrete latents (V, Z), X is a 1D linear-Gaussian random
walk with regime-switched process variance and size-modulated observation
variance:

    X_{t_0} ~ N(0, s_0^2)
    X_{t_i} | X_{t_{i-1}}, V_{t_i} ~ N(X_{t_{i-1}}, sigma2_{V_t_i} * Delta_i)
    Y_i | X_{t_i}, Z_i, S_i ~ N(X_{t_i}, R_i),
        R_i = tau2_{Z_i} / max(1 + gamma * log(S_i / S_bar), _DENOM_FLOOR).

Each particle carries its own (mu, sigma2) — different (V, Z) trajectories
accumulate different conditional variances (decision #1).
"""

from __future__ import annotations

import numpy as np

from config.default_params import ModelParams

_LOG_2PI = float(np.log(2.0 * np.pi))
_DENOM_FLOOR = 0.1  # mirror src/data/synthetic.py
# Floor on per-particle log predictive density. With Gaussian observations
# the model assigns vanishing density to outlier prints (real Polymarket
# prices can swing from 0.001 to 0.999 within seconds, which the Gaussian
# observation model rules out under any plausible variance). Without a floor
# such steps return log_lik = -inf for every particle, collapsing logsumexp
# to -inf and producing NaN in the normalized weights. The floor is well
# below any realistic value (exp(-500) ~ 7e-218) so it never affects the
# posterior on typical data.
_LOG_LIK_FLOOR = -500.0


def process_variance(V: np.ndarray, delta: float, params: ModelParams) -> np.ndarray:
    """Compute process variance for each particle state.

    Implements the random-walk transition variance
    ``Q_i = sigma2_{V_i} * Delta_i``.

    Args:
        V: Particle regime indicators in ``{0, 1}``.
        delta: Inter-trade time for the current step.
        params: Model parameters providing ``sigma2_0`` and ``sigma2_1``.

    Returns:
        Per-particle transition variances with the same leading shape as ``V``.
    """
    sigma2_v = np.where(np.asarray(V) == 0, params.sigma2_0, params.sigma2_1)
    return sigma2_v * delta


def obs_variance(
    Z: np.ndarray, log_size_ratio: float, params: ModelParams
) -> np.ndarray:
    """Compute observation variance for each particle state.

    Implements
    ``R_i = tau2_{Z_i} / max(1 + gamma * log(S_i / S_bar), _DENOM_FLOOR)``.

    Args:
        Z: Particle insider indicators in ``{0, 1}``.
        log_size_ratio: ``log(S_i / S_bar)`` at the current trade.
        params: Model parameters providing ``tau2_0``, ``tau2_1``, and ``gamma``.

    Returns:
        Per-particle observation variances with the same leading shape as ``Z``.
    """
    tau2_z = np.where(np.asarray(Z) == 0, params.tau2_0, params.tau2_1)
    denom = max(1.0 + params.gamma * log_size_ratio, _DENOM_FLOOR)
    return tau2_z / denom


def kalman_step(
    mu: np.ndarray,
    sigma2: np.ndarray,
    y: float,
    V: np.ndarray,
    Z: np.ndarray,
    delta: float,
    log_size_ratio: float,
    params: ModelParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict + update for one trade i. Vectorized over the leading particle axis.

    The transition is a random walk so mu_pred = mu; only sigma2 grows by Q.

    For i = 0 (no prior trade), pass delta = 0 with mu = 0, sigma2 = s_0^2:
    Q = sigma2_v * 0 = 0 collapses the predict, leaving the update of the
    t_0 prior with Y_0.

    Args:
        mu: Prior filtered means for the incoming particles.
        sigma2: Prior filtered variances aligned with ``mu``.
        y: Current logit-price observation.
        V: Current regime indicators per particle (0 calm, 1 news).
        Z: Current insider indicators per particle (0 no insider, 1 insider).
        delta: Inter-trade time for this update step.
        log_size_ratio: Current ``log(S_i / S_bar)`` feature.
        params: Model parameters controlling process and observation variances.

    Returns:
        Tuple ``(mu_new, sigma2_new, log_lik)`` for the updated filtering
        moments and per-particle predictive log densities, where
        ``log_lik_n = log p(y_i | y_{1:i-1}, V_{1:i}^{(n)}, Z_{1:i}^{(n)})``.
    """
    Q = process_variance(V, delta, params)
    R = obs_variance(Z, log_size_ratio, params)

    sigma2_pred = sigma2 + Q
    S = sigma2_pred + R
    innov = y - mu  # mu_pred = mu (random-walk transition)
    K = sigma2_pred / S

    mu_new = mu + K * innov
    sigma2_new = (1.0 - K) * sigma2_pred
    log_lik = -0.5 * (_LOG_2PI + np.log(S) + innov * innov / S)
    log_lik = np.maximum(log_lik, _LOG_LIK_FLOOR)
    return mu_new, sigma2_new, log_lik


def kalman_filter(
    Y: np.ndarray,
    V: np.ndarray,
    Z: np.ndarray,
    delta: np.ndarray,
    log_size_ratio: np.ndarray,
    params: ModelParams,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Forward filter for a single (V, Z) trajectory.

    Args:
        Y: Logit-price observations of shape ``(T,)``.
        V: Regime trajectory of shape ``(T,)`` with values in ``{0, 1}``.
        Z: Insider trajectory of shape ``(T,)`` with values in ``{0, 1}``.
        delta: Inter-trade times of shape ``(T,)`` with ``delta[0] == 0``.
        log_size_ratio: ``log(S_i / S_bar)`` features of shape ``(T,)``.
        params: Model parameters used by the transition and observation models.

    Returns:
        mu_filt:      (T,) E[X_i | Y_{1:i}, V_{1:i}, Z_{1:i}]
        sigma2_filt:  (T,) Var[X_i | Y_{1:i}, V_{1:i}, Z_{1:i}]
        log_marginal: scalar log p(Y_{1:T} | V_{1:T}, Z_{1:T})
    """
    T = len(Y)
    mu_filt = np.empty(T)
    sigma2_filt = np.empty(T)
    log_marginal = 0.0

    mu = np.array([0.0])
    sigma2 = np.array([params.s0_2])

    for i in range(T):
        mu, sigma2, log_lik = kalman_step(
            mu,
            sigma2,
            float(Y[i]),
            np.array([V[i]]),
            np.array([Z[i]]),
            float(delta[i]),
            float(log_size_ratio[i]),
            params,
        )
        mu_filt[i] = mu[0]
        sigma2_filt[i] = sigma2[0]
        log_marginal += float(log_lik[0])

    return mu_filt, sigma2_filt, log_marginal


def ffbs_sample(
    Y: np.ndarray,
    V: np.ndarray,
    Z: np.ndarray,
    delta: np.ndarray,
    log_size_ratio: np.ndarray,
    params: ModelParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Forward-filter, backward-sample one X trajectory from p(X | Y, V, Z).

    Uses the Rauch-Tung-Striebel backward recursion specialized to the
    random-walk transition (cross-covariance Cov(X_i, X_{i+1} | Y_{1:i}) = sigma2_i).

    Args:
        Y: Logit-price observations of shape ``(T,)``.
        V: Regime trajectory of shape ``(T,)``.
        Z: Insider trajectory of shape ``(T,)``.
        delta: Inter-trade times of shape ``(T,)``.
        log_size_ratio: ``log(S_i / S_bar)`` features of shape ``(T,)``.
        params: Model parameters used by filtering and backward sampling.
        rng: Random generator used for all normal draws in backward simulation.

    Returns:
        One sampled latent ``X`` trajectory of shape ``(T,)``.
    """
    T = len(Y)
    mu_filt, sigma2_filt, _ = kalman_filter(Y, V, Z, delta, log_size_ratio, params)

    X = np.empty(T)
    X[T - 1] = rng.normal(mu_filt[T - 1], np.sqrt(sigma2_filt[T - 1]))

    for i in range(T - 2, -1, -1):
        sigma2_v = params.sigma2_0 if V[i + 1] == 0 else params.sigma2_1
        sigma2_pred = sigma2_filt[i] + sigma2_v * float(delta[i + 1])
        J = sigma2_filt[i] / sigma2_pred
        smooth_mean = mu_filt[i] + J * (X[i + 1] - mu_filt[i])
        smooth_var = sigma2_filt[i] * (1.0 - J)
        X[i] = rng.normal(smooth_mean, np.sqrt(max(smooth_var, 0.0)))

    return X
