from dataclasses import replace

import numpy as np
import pytest

from config.default_params import InferenceConfig, ModelParams
from src.data.synthetic import generate_market
from src.inference.kalman import kalman_filter
from src.inference.smc import bootstrap_smc, smooth_paths


@pytest.fixture
def params():
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    return ModelParams.warm_start(Y_dummy)


@pytest.fixture
def config():
    return InferenceConfig(N=200, seed=42)


@pytest.fixture
def market(params):
    # Short inter-trade time -> small per-step process variance -> filter has
    # discriminative power (otherwise the random walk forgets each step).
    return generate_market(
        params, n_trades=200, n_wallets=20, n_insider_wallets=3,
        mean_inter_trade_time=1.0, rng=np.random.default_rng(7),
    )


def _smc_args(mkt, params, config):
    return (
        mkt.Y, mkt.delta, np.log(mkt.S / mkt.S_bar),
        mkt.wallet_ids, mkt.theta_w, params, config,
    )


def test_smc_output_shapes(params, config, market):
    out = bootstrap_smc(*_smc_args(market, params, config), rng=np.random.default_rng(0))
    T, N = len(market.Y), config.N
    assert out.X_filt.shape == (T,)
    assert out.V_prob_filt.shape == (T,)
    assert out.Z_prob_filt.shape == (T,)
    assert out.ess_per_step.shape == (T,)
    assert out.V_hist.shape == (T, N)
    assert out.Z_hist.shape == (T, N)
    assert out.mu_hist.shape == (T, N)
    assert out.ancestors.shape == (T, N)
    assert out.final_log_W.shape == (N,)


def test_smc_log_marginal_finite(params, config, market):
    out = bootstrap_smc(*_smc_args(market, params, config), rng=np.random.default_rng(0))
    assert np.isfinite(out.log_marginal)


def test_smc_ess_within_bounds(params, config, market):
    out = bootstrap_smc(*_smc_args(market, params, config), rng=np.random.default_rng(0))
    assert np.all(out.ess_per_step >= 1.0 - 1e-9)
    assert np.all(out.ess_per_step <= config.N + 1e-6)


def test_smc_probabilities_in_unit_interval(params, config, market):
    out = bootstrap_smc(*_smc_args(market, params, config), rng=np.random.default_rng(0))
    assert np.all((out.V_prob_filt >= 0.0) & (out.V_prob_filt <= 1.0))
    assert np.all((out.Z_prob_filt >= 0.0) & (out.Z_prob_filt <= 1.0))


def test_smc_resamples_at_least_once(params, config, market):
    out = bootstrap_smc(*_smc_args(market, params, config), rng=np.random.default_rng(0))
    assert len(out.resample_steps) > 0


def test_smc_reproducibility(params, config, market):
    args = _smc_args(market, params, config)
    out1 = bootstrap_smc(*args, rng=np.random.default_rng(123))
    out2 = bootstrap_smc(*args, rng=np.random.default_rng(123))
    assert out1.log_marginal == out2.log_marginal
    np.testing.assert_array_equal(out1.X_filt, out2.X_filt)
    np.testing.assert_array_equal(out1.final_V, out2.final_V)
    np.testing.assert_array_equal(out1.V_hist, out2.V_hist)


def test_smc_filter_beats_raw_observation(params, config, market):
    out = bootstrap_smc(*_smc_args(market, params, config), rng=np.random.default_rng(0))
    rmse_filt = np.sqrt(np.mean((out.X_filt - market.X) ** 2))
    rmse_obs = np.sqrt(np.mean((market.Y - market.X) ** 2))
    assert rmse_filt < rmse_obs


def test_smc_log_marginal_matches_kalman_when_prior_degenerate(params, market):
    """Force V_i ≡ 0 (q_01=0) and θ_w → 0 so the prior is effectively a point mass
    at (V≡0, Z≡0). The bootstrap PF then reduces to a single Kalman filter
    and log_marginal must match `kalman_filter` log-marg on that trajectory."""
    p_det = replace(params, q_01=0.0, beta_S=0.0, beta_Z=0.0)
    theta_w_zero = np.full(20, 1e-12)  # logit clips to 1e-6 -> P(Z=1) ~ 1e-6
    cfg = InferenceConfig(N=100, seed=0)

    out = bootstrap_smc(
        market.Y, market.delta, np.log(market.S / market.S_bar),
        market.wallet_ids, theta_w_zero,
        p_det, cfg, rng=np.random.default_rng(0),
    )

    log_sz = np.log(market.S / market.S_bar)
    T = len(market.Y)
    _, _, log_marg_kalman = kalman_filter(
        market.Y, np.zeros(T, dtype=int), np.zeros(T, dtype=int),
        market.delta, log_sz, p_det,
    )

    # Verify the prior actually was degenerate on this seed.
    assert np.all(out.final_V == 0)
    assert np.all(out.V_hist == 0)
    assert np.all(out.Z_hist == 0)
    # With identical particle trajectories, SMC = Kalman to floating-point precision.
    assert np.isclose(out.log_marginal, log_marg_kalman, rtol=1e-10, atol=1e-9)


def test_smooth_paths_shapes_and_ranges(params, config, market):
    out = bootstrap_smc(*_smc_args(market, params, config), rng=np.random.default_rng(0))
    X_smooth, V_smooth, Z_smooth = smooth_paths(out)
    T = len(market.Y)
    assert X_smooth.shape == (T,)
    assert V_smooth.shape == (T,)
    assert Z_smooth.shape == (T,)
    assert np.all(np.isfinite(X_smooth))
    assert np.all((V_smooth >= 0) & (V_smooth <= 1))
    assert np.all((Z_smooth >= 0) & (Z_smooth <= 1))


def test_smooth_path_at_T_matches_filter(params, config, market):
    """At i = T-1 the smoothed and filtered estimates coincide (no descendants to weight)."""
    out = bootstrap_smc(*_smc_args(market, params, config), rng=np.random.default_rng(0))
    X_smooth, V_smooth, Z_smooth = smooth_paths(out)
    assert np.isclose(X_smooth[-1], out.X_filt[-1])
    assert np.isclose(V_smooth[-1], out.V_prob_filt[-1])
    assert np.isclose(Z_smooth[-1], out.Z_prob_filt[-1])
