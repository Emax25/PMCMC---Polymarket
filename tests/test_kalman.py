"""Tests for src.inference.kalman: Kalman filter, variance helpers, and FFBS."""

from __future__ import annotations

import numpy as np
import pytest

from config.default_params import ModelParams
from src.data.synthetic import generate_market
from src.inference.kalman import (
    ffbs_sample,
    kalman_filter,
    kalman_step,
    obs_variance,
    process_variance,
)


@pytest.fixture
def params():
    """Warm-started ModelParams derived from a 200-step dummy series."""
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    return ModelParams.warm_start(Y_dummy)


def test_process_variance_regime_dependence(params):
    """Q_i equals sigma2_v * delta for each regime v ∈ {0, 1}."""
    V = np.array([0, 1, 0, 1])
    Q = process_variance(V, delta=2.0, params=params)
    assert np.allclose(Q[V == 0], params.sigma2_0 * 2.0)
    assert np.allclose(Q[V == 1], params.sigma2_1 * 2.0)


def test_obs_variance_regime_dependence(params):
    """R_i equals tau2_z when log_size_ratio=0 (size denominator is 1)."""
    Z = np.array([0, 1, 0, 1])
    R = obs_variance(Z, log_size_ratio=0.0, params=params)  # denom = 1
    assert np.allclose(R[Z == 0], params.tau2_0)
    assert np.allclose(R[Z == 1], params.tau2_1)


def test_obs_variance_floor_active(params):
    """Very negative log_size_ratio triggers the denominator floor at 0.1."""
    R = obs_variance(np.array([0]), log_size_ratio=-100.0, params=params)
    assert np.allclose(R, params.tau2_0 / 0.1)


def test_kalman_step_initial_observation(params):
    """delta=0 with the t_0 prior gives the conjugate update of N(0, s_0^2) by Y_0."""
    mu = np.array([0.0])
    sigma2 = np.array([params.s0_2])
    y = 0.5

    mu_new, sigma2_new, log_lik = kalman_step(
        mu,
        sigma2,
        y,
        V=np.array([0]),
        Z=np.array([0]),
        delta=0.0,
        log_size_ratio=0.0,
        params=params,
    )

    # Posterior of N(0, s_0^2) updated by N(y; X, tau2_0).
    expected_var = (params.s0_2 * params.tau2_0) / (params.s0_2 + params.tau2_0)
    expected_mu = expected_var * y / params.tau2_0
    assert np.allclose(mu_new[0], expected_mu)
    assert np.allclose(sigma2_new[0], expected_var)
    # Predictive marginal: N(0, s_0^2 + tau2_0)
    expected_S = params.s0_2 + params.tau2_0
    expected_log_lik = -0.5 * (np.log(2 * np.pi * expected_S) + y * y / expected_S)
    assert np.allclose(log_lik[0], expected_log_lik)


def test_kalman_step_vectorizes_over_particles(params):
    """Output shapes are (N,) and all variances / log-likelihoods are valid."""
    N = 32
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(N)
    sigma2 = np.full(N, 0.1)
    V = rng.integers(0, 2, size=N)
    Z = rng.integers(0, 2, size=N)

    mu_new, sigma2_new, log_lik = kalman_step(
        mu,
        sigma2,
        y=0.3,
        V=V,
        Z=Z,
        delta=1.0,
        log_size_ratio=0.5,
        params=params,
    )

    assert mu_new.shape == (N,)
    assert sigma2_new.shape == (N,)
    assert log_lik.shape == (N,)
    assert np.all(sigma2_new > 0)
    assert np.all(np.isfinite(log_lik))


def test_kalman_step_posterior_variance_decreases_below_prior(params):
    """Posterior variance drops below the predictive prior after an observation."""
    mu = np.array([0.0])
    sigma2 = np.array([1.0])
    _, sigma2_new, _ = kalman_step(
        mu,
        sigma2,
        y=0.0,
        V=np.array([0]),
        Z=np.array([1]),  # informed -> tight obs
        delta=1.0,
        log_size_ratio=0.0,
        params=params,
    )
    sigma2_pred = 1.0 + params.sigma2_0
    assert sigma2_new[0] < sigma2_pred


def test_filter_recovers_truth_high_snr():
    """High-SNR linear-Gaussian setup: filtered mean tracks true X with small RMSE."""
    rng = np.random.default_rng(0)
    T = 500
    sigma2 = 0.01
    tau2 = 0.001
    s0_2 = 1.0

    delta = np.ones(T)
    delta[0] = 0.0
    X = np.empty(T)
    X[0] = rng.normal(0.0, np.sqrt(s0_2))
    for i in range(1, T):
        X[i] = rng.normal(X[i - 1], np.sqrt(sigma2 * delta[i]))
    Y = X + np.sqrt(tau2) * rng.standard_normal(T)

    V = np.zeros(T, dtype=int)
    Z = np.zeros(T, dtype=int)
    log_sz = np.zeros(T)

    params = ModelParams(
        sigma2_0=sigma2,
        sigma2_1=sigma2,
        tau2_0=tau2,
        tau2_1=tau2,
        s0_2=s0_2,
    )
    mu_filt, sigma2_filt, log_marg = kalman_filter(Y, V, Z, delta, log_sz, params)

    rmse = np.sqrt(np.mean((mu_filt - X) ** 2))
    # Steady-state posterior std ~0.03 in this regime; allow margin.
    assert rmse < 0.05
    assert np.isfinite(log_marg)
    assert np.all(sigma2_filt > 0)


def test_filter_log_marginal_matches_step_sum(params):
    """log_marginal returned by kalman_filter equals the sum of step log-likelihoods."""
    rng = np.random.default_rng(0)
    T = 50
    Y = rng.standard_normal(T)
    V = rng.integers(0, 2, size=T)
    Z = rng.integers(0, 2, size=T)
    Z[0] = 0
    delta = rng.exponential(1.0, size=T)
    delta[0] = 0.0
    log_sz = rng.standard_normal(T)

    _, _, log_marg = kalman_filter(Y, V, Z, delta, log_sz, params)

    log_marg_manual = 0.0
    mu = np.array([0.0])
    sigma2 = np.array([params.s0_2])
    for i in range(T):
        mu, sigma2, ll = kalman_step(
            mu,
            sigma2,
            float(Y[i]),
            np.array([V[i]]),
            np.array([Z[i]]),
            float(delta[i]),
            float(log_sz[i]),
            params,
        )
        log_marg_manual += float(ll[0])
    assert np.isclose(log_marg, log_marg_manual)


def test_filter_recovers_synthetic_X(params):
    """On synthetic data given true (V, Z), the filter beats the raw observation."""
    mkt = generate_market(params, n_trades=500, rng=np.random.default_rng(7))
    log_sz = np.log(mkt.S / mkt.S_bar)

    mu_filt, _, _ = kalman_filter(mkt.Y, mkt.V, mkt.Z, mkt.delta, log_sz, params)
    rmse_filt = np.sqrt(np.mean((mu_filt - mkt.X) ** 2))
    rmse_obs = np.sqrt(np.mean((mkt.Y - mkt.X) ** 2))
    assert rmse_filt < rmse_obs


def test_ffbs_last_marginal_matches_filter(params):
    """The smoothed marginal at i = T-1 equals the filtered marginal there."""
    rng = np.random.default_rng(0)
    T = 20
    Y = rng.standard_normal(T)
    V = rng.integers(0, 2, size=T)
    Z = rng.integers(0, 2, size=T)
    Z[0] = 0
    delta = rng.exponential(1.0, size=T)
    delta[0] = 0.0
    log_sz = rng.standard_normal(T)

    mu_filt, sigma2_filt, _ = kalman_filter(Y, V, Z, delta, log_sz, params)

    n_samples = 5000
    last = np.empty(n_samples)
    for s in range(n_samples):
        X = ffbs_sample(Y, V, Z, delta, log_sz, params, rng=np.random.default_rng(s))
        last[s] = X[T - 1]

    se_mean = np.sqrt(sigma2_filt[T - 1] / n_samples)
    assert abs(last.mean() - mu_filt[T - 1]) < 5 * se_mean
    # Empirical variance within ~15% of analytical (chi-square fluctuation).
    assert abs(last.var() - sigma2_filt[T - 1]) / sigma2_filt[T - 1] < 0.15


def test_ffbs_smoother_beats_filter_on_synthetic(params):
    """Averaged FFBS samples (smoother mean) beat the forward filter on RMSE."""
    mkt = generate_market(params, n_trades=300, rng=np.random.default_rng(11))
    log_sz = np.log(mkt.S / mkt.S_bar)

    mu_filt, _, _ = kalman_filter(mkt.Y, mkt.V, mkt.Z, mkt.delta, log_sz, params)

    n_samples = 200
    X_acc = np.zeros(len(mkt.Y))
    for s in range(n_samples):
        X_acc += ffbs_sample(
            mkt.Y,
            mkt.V,
            mkt.Z,
            mkt.delta,
            log_sz,
            params,
            rng=np.random.default_rng(1000 + s),
        )
    X_smooth = X_acc / n_samples

    rmse_filt = np.sqrt(np.mean((mu_filt - mkt.X) ** 2))
    rmse_smooth = np.sqrt(np.mean((X_smooth - mkt.X) ** 2))
    assert rmse_smooth <= rmse_filt
