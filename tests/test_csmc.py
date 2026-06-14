from dataclasses import replace

import numpy as np
import pytest

from config.default_params import InferenceConfig, ModelParams
from src.data.synthetic import generate_market
from src.inference.csmc import REFERENCE_INDEX, conditional_smc
from src.inference.kalman import kalman_filter
from src.inference.smc import bootstrap_smc, sample_path


@pytest.fixture
def params():
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    return ModelParams.warm_start(Y_dummy)


@pytest.fixture
def config():
    return InferenceConfig(N=50, seed=42)


@pytest.fixture
def market(params):
    return generate_market(
        params, n_trades=150, n_wallets=20, n_insider_wallets=3,
        mean_inter_trade_time=1.0, rng=np.random.default_rng(7),
    )


def _csmc_args(mkt, params, config, V_ref, Z_ref):
    return (
        mkt.Y, mkt.delta, np.log(mkt.S / mkt.S_bar),
        mkt.wallet_ids, mkt.theta_w, params, config,
        V_ref, Z_ref,
    )


def test_csmc_shapes(params, config, market):
    out = conditional_smc(
        *_csmc_args(market, params, config, market.V, market.Z),
        rng=np.random.default_rng(0),
    )
    T, N = len(market.Y), config.N
    assert out.X_filt.shape == (T,)
    assert out.V_prob_filt.shape == (T,)
    assert out.Z_prob_filt.shape == (T,)
    assert out.V_hist.shape == (T, N)
    assert out.Z_hist.shape == (T, N)
    assert out.mu_hist.shape == (T, N)
    assert out.ancestors.shape == (T, N)
    assert out.final_log_W.shape == (N,)


def test_csmc_reference_pinned_at_index_0(params, config, market):
    out = conditional_smc(
        *_csmc_args(market, params, config, market.V, market.Z),
        rng=np.random.default_rng(0),
    )
    np.testing.assert_array_equal(out.V_hist[:, REFERENCE_INDEX], market.V)
    np.testing.assert_array_equal(out.Z_hist[:, REFERENCE_INDEX], market.Z)
    assert int(out.final_V[REFERENCE_INDEX]) == int(market.V[-1])
    assert int(out.final_Z[REFERENCE_INDEX]) == int(market.Z[-1])


def test_csmc_outputs_binary(params, config, market):
    out = conditional_smc(
        *_csmc_args(market, params, config, market.V, market.Z),
        rng=np.random.default_rng(0),
    )
    assert set(np.unique(out.V_hist)).issubset({0, 1})
    assert set(np.unique(out.Z_hist)).issubset({0, 1})


def test_csmc_log_marginal_finite(params, config, market):
    out = conditional_smc(
        *_csmc_args(market, params, config, market.V, market.Z),
        rng=np.random.default_rng(0),
    )
    assert np.isfinite(out.log_marginal)


def test_csmc_ess_in_bounds(params, config, market):
    out = conditional_smc(
        *_csmc_args(market, params, config, market.V, market.Z),
        rng=np.random.default_rng(0),
    )
    assert np.all(out.ess_per_step >= 1.0 - 1e-9)
    assert np.all(out.ess_per_step <= config.N + 1e-6)


def test_csmc_probabilities_in_unit_interval(params, config, market):
    out = conditional_smc(
        *_csmc_args(market, params, config, market.V, market.Z),
        rng=np.random.default_rng(0),
    )
    assert np.all((out.V_prob_filt >= 0.0) & (out.V_prob_filt <= 1.0))
    assert np.all((out.Z_prob_filt >= 0.0) & (out.Z_prob_filt <= 1.0))


def test_csmc_reproducibility(params, config, market):
    args = _csmc_args(market, params, config, market.V, market.Z)
    out1 = conditional_smc(*args, rng=np.random.default_rng(123))
    out2 = conditional_smc(*args, rng=np.random.default_rng(123))
    assert out1.log_marginal == out2.log_marginal
    np.testing.assert_array_equal(out1.V_hist, out2.V_hist)
    np.testing.assert_array_equal(out1.final_V, out2.final_V)


def test_csmc_filter_beats_raw_observation(params, config, market):
    out = conditional_smc(
        *_csmc_args(market, params, config, market.V, market.Z),
        rng=np.random.default_rng(0),
    )
    rmse_filt = np.sqrt(np.mean((out.X_filt - market.X) ** 2))
    rmse_obs = np.sqrt(np.mean((market.Y - market.X) ** 2))
    assert rmse_filt < rmse_obs


def test_csmc_locally_optimal_keeps_higher_ess(params, config, market):
    """Locally-optimal proposal should produce higher average ESS than bootstrap
    prior, since it folds the likelihood into the proposal."""
    out_csmc = conditional_smc(
        *_csmc_args(market, params, config, market.V, market.Z),
        rng=np.random.default_rng(0),
    )
    out_boot = bootstrap_smc(
        market.Y, market.delta, np.log(market.S / market.S_bar),
        market.wallet_ids, market.theta_w, params, config,
        rng=np.random.default_rng(0),
    )
    assert out_csmc.ess_per_step.mean() > out_boot.ess_per_step.mean()


def test_csmc_rejects_invalid_reference_length(params, config, market):
    with pytest.raises(ValueError):
        conditional_smc(
            market.Y, market.delta, np.log(market.S / market.S_bar),
            market.wallet_ids, market.theta_w, params, config,
            market.V[:-1], market.Z,
            rng=np.random.default_rng(0),
        )


def test_csmc_rejects_z_ref_starting_at_one(params, config, market):
    bad_Z = market.Z.copy()
    bad_Z[0] = 1
    with pytest.raises(ValueError):
        conditional_smc(
            *_csmc_args(market, params, config, market.V, bad_Z),
            rng=np.random.default_rng(0),
        )


def test_sample_path_returns_valid_trajectory(params, config, market):
    out = conditional_smc(
        *_csmc_args(market, params, config, market.V, market.Z),
        rng=np.random.default_rng(0),
    )
    V_path, Z_path = sample_path(out, rng=np.random.default_rng(1))
    T = len(market.Y)
    assert V_path.shape == (T,)
    assert Z_path.shape == (T,)
    assert set(np.unique(V_path)).issubset({0, 1})
    assert set(np.unique(Z_path)).issubset({0, 1})
    assert int(Z_path[0]) == 0  # Z_0 := 0


def test_csmc_log_marginal_matches_kalman_when_prior_degenerate(params, market):
    """With q_01=0 and θ_w → 0, the locally-optimal proposal collapses onto
    the V≡0, Z≡0 reference and log_marginal must match the conditional Kalman
    log-marg on that trajectory."""
    p_det = replace(params, q_01=0.0, beta_S=0.0, beta_Z=0.0)
    theta_w_zero = np.full(20, 1e-12)
    cfg = InferenceConfig(N=50, seed=0)
    T = len(market.Y)
    V_ref = np.zeros(T, dtype=np.int8)
    Z_ref = np.zeros(T, dtype=np.int8)

    out = conditional_smc(
        market.Y, market.delta, np.log(market.S / market.S_bar),
        market.wallet_ids, theta_w_zero,
        p_det, cfg,
        V_ref, Z_ref,
        rng=np.random.default_rng(0),
    )

    log_sz = np.log(market.S / market.S_bar)
    _, _, log_marg_kalman = kalman_filter(
        market.Y, V_ref, Z_ref, market.delta, log_sz, p_det,
    )
    assert np.all(out.V_hist == 0)
    assert np.all(out.Z_hist == 0)
    # CSMC marginalizes over all 4 (V, Z) states, so the Z=1 prior mass
    # (clipped to ε=1e-6 by logit) leaks O(T·1e-6) into log_marginal even
    # though Z_hist is all-zero. That's the right answer, not a bug — but it
    # diverges from the strictly-conditional Kalman value at O(1e-4).
    assert np.isclose(out.log_marginal, log_marg_kalman, atol=1e-3)
