import numpy as np
import pytest

from config.default_params import InferenceConfig, ModelParams
from src.data.synthetic import generate_market
from src.inference.particle_gibbs import MarketData, PGOutput, particle_gibbs


@pytest.fixture
def warm_params():
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    return ModelParams.warm_start(Y_dummy)


def _to_market_data(mkt):
    return MarketData(
        Y=mkt.Y, delta=mkt.delta,
        log_size_ratio=np.log(mkt.S / mkt.S_bar),
        wallet_ids=mkt.wallet_ids,
    )


def _make_synth(*, T=100, n_wallets=10, n_insider=2, seed=7, params=None):
    if params is None:
        rng = np.random.default_rng(0)
        Y_dummy = rng.standard_normal(200)
        params = ModelParams.warm_start(Y_dummy)
    mkt = generate_market(
        params, n_trades=T, n_wallets=n_wallets, n_insider_wallets=n_insider,
        mean_inter_trade_time=1.0, rng=np.random.default_rng(seed),
    )
    return mkt


def _roc_auc(y_true, y_score):
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
    mkt = _make_synth(T=60, seed=4)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, n_iter=10, n_burnin=2, seed=0)
    out = particle_gibbs([md], cfg, rng=np.random.default_rng(0))
    assert np.all(out.Z[0][:, 0] == 0)  # Z_0 := 0 by model


def test_pg_reproducibility():
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
    cfg_steps_before = (cfg.mh_step_beta_S, cfg.mh_step_beta_Z,
                        cfg.mh_step_log_tau2_0, cfg.mh_step_log_tau2_1)
    _ = particle_gibbs([md], cfg, rng=np.random.default_rng(0))
    cfg_steps_after = (cfg.mh_step_beta_S, cfg.mh_step_beta_Z,
                       cfg.mh_step_log_tau2_0, cfg.mh_step_log_tau2_1)
    assert cfg_steps_before == cfg_steps_after


def test_pg_multi_market():
    mkts = [_make_synth(T=60, n_wallets=10, seed=s) for s in (1, 2, 3)]
    mds = [_to_market_data(m) for m in mkts]
    cfg = InferenceConfig(N=20, n_iter=10, seed=0)
    # Force a shared wallet space across markets
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
        [md], cfg, rng=np.random.default_rng(0),
        n_wallets=10,
        params_init=p_init,
        theta_w_init=np.full(10, 0.05),
        V_ref_init=[V_init], Z_ref_init=[Z_init],
    )
    assert out.V[0].shape == (1, md.T)


@pytest.mark.slow
def test_pg_z_posterior_recovers_insider_trades():
    """Headline §9 validation: P(Z_i = 1 | D) should discriminate true insider
    trades (ROC AUC well above random)."""
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    p_true = ModelParams.warm_start(Y_dummy)
    mkt = generate_market(
        p_true, n_trades=200, n_wallets=20, n_insider_wallets=3,
        mean_inter_trade_time=1.0, rng=np.random.default_rng(11),
    )
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=50, n_iter=200, n_burnin=100, seed=0)
    out = particle_gibbs([md], cfg, rng=np.random.default_rng(42))

    Z_prob = out.Z[0][cfg.n_burnin:].mean(axis=0)
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
        p_true, n_trades=200, n_wallets=20, n_insider_wallets=3,
        mean_inter_trade_time=1.0, rng=np.random.default_rng(11),
    )
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=50, n_iter=200, n_burnin=100, seed=0)
    out = particle_gibbs([md], cfg, rng=np.random.default_rng(42))

    theta_post = out.theta_w[cfg.n_burnin:].mean(axis=0)
    insider_mean = theta_post[mkt.insider_wallet_ids].mean()
    regular_mean = np.delete(theta_post, mkt.insider_wallet_ids).mean()
    assert insider_mean > regular_mean
    # Insiders' true θ ≈ 0.9, regular ≈ 0.05; posterior should pick this up
    assert insider_mean > 2.0 * regular_mean
