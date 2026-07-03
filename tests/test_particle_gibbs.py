"""Tests for src.inference.particle_gibbs: shapes, ranges, and recovery."""

from __future__ import annotations

import numpy as np
import pytest

from config.default_params import InferenceConfig, ModelParams
from src.data.synthetic import generate_market
from src.inference.particle_gibbs import MarketData, PGOutput, filter_screen, particle_gibbs


@pytest.fixture
def warm_params():
    """Warm-started ModelParams derived from a 200-step dummy series."""
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    return ModelParams.warm_start(Y_dummy)


def _to_market_data(mkt):
    """Convert a synthetic market to a MarketData input struct."""
    return MarketData(
        Y=mkt.Y,
        delta=mkt.delta,
        log_size_ratio=np.log(mkt.S / mkt.S_bar),
        wallet_ids=mkt.wallet_ids,
    )


def _make_synth(*, T=100, n_wallets=10, n_insider=2, seed=7, params=None):
    """Generate a synthetic market; creates warm-start params if none provided."""
    if params is None:
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
    return mkt


def _roc_auc(y_true, y_score):
    """Compute ROC AUC via the Wilcoxon rank-sum formula."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def test_pg_runs_end_to_end():
    """particle_gibbs completes and returns a PGOutput with correct array shapes."""
    mkt = _make_synth(T=80, n_wallets=10, n_insider=2, seed=3)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, n_iter=15, n_burnin=5, seed=0)
    out = particle_gibbs([md], cfg, rng=np.random.default_rng(0))
    assert isinstance(out, PGOutput)
    assert out.sigma2_0.shape == (15,)
    assert out.theta_w.shape == (15, 10)
    assert out.X[0].shape == (15, 80)
    assert out.V[0].shape == (15, 80)
    assert out.Z[0].shape == (15, 80)
    assert out.log_marg.shape == (15, 1)
    assert out.acc_beta_S.shape == (15,)


def test_pg_outputs_are_finite_and_in_range():
    """All sampled parameters are finite, positive (variances), and in valid ranges."""
    mkt = _make_synth(T=60, n_wallets=10, seed=4)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, n_iter=15, n_burnin=5, seed=0)
    out = particle_gibbs([md], cfg, rng=np.random.default_rng(0))
    assert np.all(np.isfinite(out.sigma2_0)) and np.all(out.sigma2_0 > 0)
    assert np.all(np.isfinite(out.sigma2_1)) and np.all(out.sigma2_1 > 0)
    assert np.all(np.isfinite(out.tau2_0)) and np.all(out.tau2_0 > 0)
    assert np.all(np.isfinite(out.tau2_1)) and np.all(out.tau2_1 > 0)
    assert np.all((out.q_01 >= 0) & (out.q_01 <= 1))
    assert np.all((out.q_10 >= 0) & (out.q_10 <= 1))
    assert np.all(np.isfinite(out.beta_S))
    assert np.all(np.isfinite(out.beta_Z))
    assert np.all(np.isfinite(out.log_marg))
    assert np.all((out.theta_w >= 0) & (out.theta_w <= 1))
    assert set(np.unique(out.V[0])).issubset({0, 1})
    assert set(np.unique(out.Z[0])).issubset({0, 1})


def test_pg_z_at_step_0_always_zero():
    """Z_0 := 0 by model construction; every sampled trajectory must obey this."""
    mkt = _make_synth(T=60, seed=4)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, n_iter=10, n_burnin=2, seed=0)
    out = particle_gibbs([md], cfg, rng=np.random.default_rng(0))
    assert np.all(out.Z[0][:, 0] == 0)  # Z_0 := 0 by model


def test_pg_reproducibility():
    """Identical seeds produce bit-exact identical chain outputs."""
    mkt = _make_synth(T=60, seed=5)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, n_iter=12, n_burnin=4, seed=0)
    out1 = particle_gibbs([md], cfg, rng=np.random.default_rng(42))
    out2 = particle_gibbs([md], cfg, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(out1.sigma2_0, out2.sigma2_0)
    np.testing.assert_array_equal(out1.Z[0], out2.Z[0])
    np.testing.assert_array_equal(out1.theta_w, out2.theta_w)


def test_pg_does_not_mutate_caller_config():
    """Adaptive tuning happens on a local copy; the caller's config stays put."""
    mkt = _make_synth(T=60, seed=5)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, n_iter=50, n_burnin=20, seed=0)
    cfg_steps_before = (
        cfg.mh_step_beta_S,
        cfg.mh_step_beta_Z,
        cfg.mh_step_log_tau2_0,
        cfg.mh_step_log_tau2_1,
    )
    _ = particle_gibbs([md], cfg, rng=np.random.default_rng(0))
    cfg_steps_after = (
        cfg.mh_step_beta_S,
        cfg.mh_step_beta_Z,
        cfg.mh_step_log_tau2_0,
        cfg.mh_step_log_tau2_1,
    )
    assert cfg_steps_before == cfg_steps_after


def test_pg_multi_market():
    """Multi-market run produces one latent array per market and pooled theta_w."""
    mkts = [_make_synth(T=60, n_wallets=10, seed=s) for s in (1, 2, 3)]
    mds = [_to_market_data(m) for m in mkts]
    cfg = InferenceConfig(N=20, n_iter=10, seed=0)
    n_wallets = 10
    out = particle_gibbs(mds, cfg, rng=np.random.default_rng(0), n_wallets=n_wallets)
    assert len(out.X) == 3
    assert len(out.V) == 3
    assert len(out.Z) == 3
    assert out.log_marg.shape == (10, 3)
    assert out.theta_w.shape == (10, n_wallets)


def test_pg_respects_provided_initial_reference():
    """If V_ref_init / Z_ref_init is supplied, the bootstrap-seed step is skipped
    and the first CSMC pass pins the provided reference."""
    mkt = _make_synth(T=60, seed=8)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, n_iter=1, seed=0)
    V_init = np.zeros(md.T, dtype=np.int8)
    Z_init = np.zeros(md.T, dtype=np.int8)
    # Init params/theta so the iter-0 CSMC has well-defined likelihoods
    Y_dummy = np.random.default_rng(0).standard_normal(200)
    p_init = ModelParams.warm_start(Y_dummy)
    out = particle_gibbs(
        [md],
        cfg,
        rng=np.random.default_rng(0),
        n_wallets=10,
        params_init=p_init,
        theta_w_init=np.full(10, 0.05),
        V_ref_init=[V_init],
        Z_ref_init=[Z_init],
    )
    assert out.V[0].shape == (1, md.T)


def test_pg_parallel_shapes_match_sequential():
    """particle_gibbs with n_jobs=2 produces same-shaped outputs as n_jobs=1."""
    from dataclasses import replace
    mkt = _make_synth(T=60, n_wallets=10, seed=5)
    md = _to_market_data(mkt)
    cfg_seq = InferenceConfig(N=20, n_iter=10, n_burnin=2, seed=0)
    cfg_par = replace(cfg_seq, n_jobs=2)
    out_seq = particle_gibbs([md], cfg_seq, rng=np.random.default_rng(0))
    out_par = particle_gibbs([md], cfg_par, rng=np.random.default_rng(0))
    assert out_par.sigma2_0.shape == out_seq.sigma2_0.shape
    assert out_par.theta_w.shape == out_seq.theta_w.shape
    assert out_par.Z[0].shape == out_seq.Z[0].shape
    assert out_par.log_marg.shape == out_seq.log_marg.shape


def test_pg_parallel_produces_finite_valid_outputs():
    """parallel particle_gibbs produces finite, in-range outputs."""
    from dataclasses import replace
    mkts = [_make_synth(T=60, n_wallets=10, seed=s) for s in (1, 2)]
    mds = [_to_market_data(m) for m in mkts]
    cfg = InferenceConfig(N=20, n_iter=10, n_burnin=2, seed=0, n_jobs=2)
    out = particle_gibbs(mds, cfg, rng=np.random.default_rng(0), n_wallets=10)
    assert np.all(np.isfinite(out.sigma2_0))
    assert np.all(out.sigma2_0 > 0)
    assert np.all(np.isfinite(out.log_marg))
    assert np.all((out.theta_w >= 0) & (out.theta_w <= 1))


def test_filter_screen_returns_valid_wallet_scores():
    """filter_screen returns per-wallet scores in [0, 1] with correct shape."""
    mkt = _make_synth(T=80, n_wallets=10, n_insider=2, seed=3)
    md = _to_market_data(mkt)
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    params = ModelParams.warm_start(Y_dummy)
    theta_w = np.full(10, 0.05)
    cfg = InferenceConfig(N=20)
    scores = filter_screen([md], params, theta_w, cfg, rng=rng)
    assert scores.shape == (10,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    assert np.all(np.isfinite(scores))


def test_filter_screen_insiders_score_higher():
    """filter_screen assigns higher Z_prob to true insider wallets on average."""
    mkt = _make_synth(T=200, n_wallets=20, n_insider=3, seed=7)
    md = _to_market_data(mkt)
    rng = np.random.default_rng(1)
    Y_dummy = rng.standard_normal(200)
    params = ModelParams.warm_start(Y_dummy)
    theta_w_true = np.where(
        np.isin(np.arange(20), mkt.insider_wallet_ids), 0.9, 0.05
    )
    cfg = InferenceConfig(N=50)
    scores = filter_screen([md], params, theta_w_true, cfg, rng=rng)
    insider_mean = scores[mkt.insider_wallet_ids].mean()
    regular_ids = [w for w in range(20) if w not in mkt.insider_wallet_ids]
    regular_mean = scores[regular_ids].mean()
    assert insider_mean > regular_mean, (
        f"Insider mean Z_prob {insider_mean:.3f} should exceed regular {regular_mean:.3f}"
    )


def test_filter_screen_multi_market():
    """filter_screen aggregates correctly across multiple markets."""
    mkts = [_make_synth(T=60, n_wallets=10, seed=s) for s in (1, 2, 3)]
    mds = [_to_market_data(m) for m in mkts]
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    params = ModelParams.warm_start(Y_dummy)
    theta_w = np.full(10, 0.05)
    cfg = InferenceConfig(N=20)
    scores = filter_screen(mds, params, theta_w, cfg, rng=rng)
    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))


def test_filter_screen_reproducible():
    """filter_screen produces identical results for the same seed."""
    mkt = _make_synth(T=80, n_wallets=10, seed=4)
    md = _to_market_data(mkt)
    Y_dummy = np.random.default_rng(0).standard_normal(200)
    params = ModelParams.warm_start(Y_dummy)
    theta_w = np.full(10, 0.05)
    cfg = InferenceConfig(N=20)
    s1 = filter_screen([md], params, theta_w, cfg, rng=np.random.default_rng(99))
    s2 = filter_screen([md], params, theta_w, cfg, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(s1, s2)


@pytest.mark.slow
def test_pg_z_posterior_recovers_insider_trades():
    """Headline §9 validation: P(Z_i = 1 | D) should discriminate true insider
    trades (ROC AUC well above random)."""
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    p_true = ModelParams.warm_start(Y_dummy)
    mkt = generate_market(
        p_true,
        n_trades=200,
        n_wallets=20,
        n_insider_wallets=3,
        mean_inter_trade_time=1.0,
        rng=np.random.default_rng(11),
    )
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=50, n_iter=200, n_burnin=100, seed=0)
    out = particle_gibbs([md], cfg, rng=np.random.default_rng(42))

    Z_prob = out.Z[0][cfg.n_burnin :].mean(axis=0)
    auc = _roc_auc(mkt.Z, Z_prob)
    assert auc > 0.70, f"AUC = {auc:.3f}; expected > 0.70"


@pytest.mark.slow
def test_pg_theta_w_recovers_insider_wallets():
    """θ_w posterior mean for true insider wallets should exceed that for
    regular wallets."""
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    p_true = ModelParams.warm_start(Y_dummy)
    mkt = generate_market(
        p_true,
        n_trades=200,
        n_wallets=20,
        n_insider_wallets=3,
        mean_inter_trade_time=1.0,
        rng=np.random.default_rng(11),
    )
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=50, n_iter=200, n_burnin=100, seed=0)
    out = particle_gibbs([md], cfg, rng=np.random.default_rng(42))

    theta_post = out.theta_w[cfg.n_burnin :].mean(axis=0)
    insider_mean = theta_post[mkt.insider_wallet_ids].mean()
    regular_mean = np.delete(theta_post, mkt.insider_wallet_ids).mean()
    assert insider_mean > regular_mean
    # Insiders' true θ ≈ 0.9, regular ≈ 0.05; posterior should pick this up
    assert insider_mean > 2.0 * regular_mean
