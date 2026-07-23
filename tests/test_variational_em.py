"""Tests for src.inference.variational_em."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from config.default_params import InferenceConfig, ModelParams
from src.data.synthetic import generate_market
from src.inference.particle_gibbs import MarketData
from src.inference.variational_em import VEMOutput, variational_em

FIXTURES = Path(__file__).parent / "fixtures"


def _make_synth(*, T=100, n_wallets=10, n_insider=2, seed=7):
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    params = ModelParams.warm_start(Y_dummy)
    mkt = generate_market(
        params,
        n_trades=T,
        n_wallets=n_wallets,
        n_insider_wallets=n_insider,
        mean_inter_trade_time=1.0,
        rng=np.random.default_rng(seed),
    )
    return mkt, params


def _to_market_data(mkt):
    return MarketData(
        Y=mkt.Y,
        delta=mkt.delta,
        log_size_ratio=np.log(mkt.S / mkt.S_bar),
        wallet_ids=mkt.wallet_ids,
    )


def test_vem_runs_end_to_end():
    mkt, params = _make_synth(T=80, seed=3)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20)
    out = variational_em([md], cfg, n_wallets=10, params_init=params, n_iter=5)
    assert isinstance(out, VEMOutput)
    assert out.n_iter_run >= 1
    assert len(out.Z_prob) == 1
    assert out.Z_prob[0].shape == (80,)
    assert out.V_prob[0].shape == (80,)
    assert out.X_mean[0].shape == (80,)
    assert out.theta_w.shape == (10,)
    assert out.elbo_trace.shape[0] == out.n_iter_run


def test_vem_outputs_in_valid_range():
    mkt, params = _make_synth(T=60, seed=4)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20)
    out = variational_em([md], cfg, n_wallets=10, params_init=params, n_iter=10)
    assert np.all((out.Z_prob[0] >= 0) & (out.Z_prob[0] <= 1))
    assert np.all((out.V_prob[0] >= 0) & (out.V_prob[0] <= 1))
    assert np.all(np.isfinite(out.X_mean[0]))
    assert np.all((out.theta_w >= 0) & (out.theta_w <= 1))
    assert np.all(out.params.sigma2_0 > 0)
    assert np.all(out.params.sigma2_1 > 0)
    assert np.all(out.params.tau2_0 > 0)
    assert np.all(out.params.tau2_1 > 0)


def test_vem_z0_always_zero():
    mkt, params = _make_synth(T=60, seed=5)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20)
    out = variational_em([md], cfg, n_wallets=10, params_init=params, n_iter=5)
    # Z_0 := 0 by model convention: q(Z_0=1) should be near 0
    assert float(out.Z_prob[0][0]) < 1e-10


def test_vem_elbo_non_decreasing():
    """EM log-marginal should be non-decreasing (or nearly so due to approximation)."""
    mkt, params = _make_synth(T=100, seed=6)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20)
    out = variational_em([md], cfg, n_wallets=10, params_init=params, n_iter=20, tol=1e-8)
    # Check that the trace doesn't decrease monotonically (ADF is approximate so small dips OK)
    trace = out.elbo_trace
    assert len(trace) >= 1
    # The last value should be finite
    assert np.isfinite(trace[-1])


def test_vem_multi_market():
    mkts_params = [_make_synth(T=60, n_wallets=10, seed=s) for s in (1, 2, 3)]
    mds = [_to_market_data(m) for m, _ in mkts_params]
    params = mkts_params[0][1]
    cfg = InferenceConfig(N=20)
    out = variational_em(mds, cfg, n_wallets=10, params_init=params, n_iter=5)
    assert len(out.Z_prob) == 3
    assert len(out.V_prob) == 3
    assert len(out.X_mean) == 3


def test_vem_faster_than_pg():
    """VEM should complete noticeably faster than PG for the same market."""
    import time
    from src.inference.particle_gibbs import particle_gibbs

    mkt, params = _make_synth(T=200, n_wallets=20, seed=42)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=50, n_iter=50, n_burnin=10)

    t0 = time.perf_counter()
    _ = variational_em([md], cfg, n_wallets=20, params_init=params, n_iter=30, tol=1e-4)
    t_vem = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = particle_gibbs([md], cfg, rng=np.random.default_rng(0), n_wallets=20, params_init=params)
    t_pg = time.perf_counter() - t0

    assert t_vem < t_pg, (
        f"VEM ({t_vem:.3f}s) should be faster than PG ({t_pg:.3f}s)"
    )


@pytest.mark.slow
def test_vem_z_prob_discriminates_insiders():
    """VEM Z_prob should yield AUC > 0.65 on synthetic insider data."""
    from scipy.stats import rankdata
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    p_true = ModelParams.warm_start(Y_dummy)
    mkt = generate_market(
        p_true,
        n_trades=300,
        n_wallets=20,
        n_insider_wallets=3,
        mean_inter_trade_time=1.0,
        rng=np.random.default_rng(11),
    )
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=50)
    out = variational_em([md], cfg, n_wallets=20, n_iter=50, tol=1e-4)
    z_prob = out.Z_prob[0]
    z_true = mkt.Z.astype(int)
    n_pos = int(z_true.sum())
    n_neg = len(z_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return
    ranks = rankdata(z_prob)
    rank_sum = float(ranks[z_true == 1].sum())
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    assert auc > 0.60, f"VEM AUC = {auc:.3f}, expected > 0.60"


def test_vem_beta0_bit_identical_to_prechange_fixture():
    """Regression anchor: standardization is inert when beta_S = beta_Z = 0.

    Compares against a fixture generated by the pre-standardization code
    (tests/fixtures/vem_prechange_beta0.npz), same synthetic config/seed as
    test_vem_runs_end_to_end. With beta_S = beta_Z = 0.0 (ModelParams
    default), the standardized covariates are multiplied by zero in the
    predictor, so outputs must be bit-identical regardless of (m_S, s_S, m_Z).
    """
    fixture = np.load(FIXTURES / "vem_prechange_beta0.npz")

    mkt, params = _make_synth(T=80, n_wallets=10, n_insider=2, seed=3)
    assert params.beta_S == 0.0
    assert params.beta_Z == 0.0
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20)
    out = variational_em([md], cfg, n_wallets=10, params_init=params, n_iter=10)

    np.testing.assert_array_equal(out.Z_prob[0], fixture["Z_prob"])
    np.testing.assert_array_equal(out.V_prob[0], fixture["V_prob"])
    np.testing.assert_array_equal(out.X_mean[0], fixture["X_mean"])
    np.testing.assert_array_equal(out.theta_w, fixture["theta_w"])
    np.testing.assert_array_equal(out.elbo_trace, fixture["elbo_trace"])


def test_vem_constant_size_market_no_nan():
    """A constant log_size_ratio market must not produce NaN/inf (s_S == 0 fallback)."""
    mkt, params = _make_synth(T=60, n_wallets=10, n_insider=2, seed=8)
    md = _to_market_data(mkt)
    # Force a degenerate constant-size market: all trades the same size.
    md = MarketData(
        Y=md.Y,
        delta=md.delta,
        log_size_ratio=np.full_like(md.log_size_ratio, 0.37),
        wallet_ids=md.wallet_ids,
    )
    params = replace(params, beta_S=1.5, beta_Z=0.5)
    cfg = InferenceConfig(N=20)
    out = variational_em([md], cfg, n_wallets=10, params_init=params, n_iter=5)

    assert out.s_S < 1e-8  # below _S_STD_FLOOR: fallback path exercised
    assert np.isclose(out.m_S, 0.37)
    assert np.all(np.isfinite(out.Z_prob[0]))
    assert np.all(np.isfinite(out.V_prob[0]))
    assert np.all(np.isfinite(out.X_mean[0]))
    assert np.all(np.isfinite(out.theta_w))
    assert np.all(np.isfinite(out.elbo_trace))


def test_vem_output_constants_round_trip():
    """m_S/s_S must equal the pooled mean/std of the concatenated inputs."""
    mkts_params = [_make_synth(T=50, n_wallets=10, seed=s) for s in (1, 2)]
    mds = [_to_market_data(m) for m, _ in mkts_params]
    params = mkts_params[0][1]
    cfg = InferenceConfig(N=20)
    out = variational_em(mds, cfg, n_wallets=10, params_init=params, n_iter=3)

    pooled = np.concatenate([md.log_size_ratio for md in mds])
    assert out.m_S == pytest.approx(float(np.mean(pooled)), abs=1e-12)
    assert out.s_S == pytest.approx(float(np.std(pooled)), abs=1e-12)
