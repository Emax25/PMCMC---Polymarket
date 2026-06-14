import numpy as np
import pytest

from config.default_params import InferenceConfig, ModelParams
from src.data.synthetic import generate_market
from src.inference.ipmcmc import iPMCMCOutput, ipmcmc
from src.inference.particle_gibbs import MarketData


def _to_market_data(mkt):
    return MarketData(
        Y=mkt.Y, delta=mkt.delta,
        log_size_ratio=np.log(mkt.S / mkt.S_bar),
        wallet_ids=mkt.wallet_ids,
    )


def _make_synth(*, T=80, n_wallets=10, n_insider=2, seed=7):
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    p = ModelParams.warm_start(Y_dummy)
    return generate_market(
        p, n_trades=T, n_wallets=n_wallets, n_insider_wallets=n_insider,
        mean_inter_trade_time=1.0, rng=np.random.default_rng(seed),
    )


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
    return (float(ranks[y_true == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def test_ipmcmc_runs_end_to_end():
    mkt = _make_synth(T=60, n_wallets=10, seed=3)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, M=4, P=2, n_iter=10, n_burnin=2, seed=0)
    out = ipmcmc([md], cfg, rng=np.random.default_rng(0))
    assert isinstance(out, iPMCMCOutput)
    assert out.sigma2_0.shape == (10, 2)
    assert out.theta_w.shape == (10, 2, 10)
    assert out.log_marg.shape == (10, 4)
    assert out.chain_indices.shape == (10, 2)
    assert out.X[0].shape == (10, 2, 60)
    assert out.V[0].shape == (10, 2, 60)
    assert out.Z[0].shape == (10, 2, 60)


def test_ipmcmc_outputs_in_valid_ranges():
    mkt = _make_synth(T=60, seed=4)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, M=4, P=2, n_iter=10, n_burnin=2, seed=0)
    out = ipmcmc([md], cfg, rng=np.random.default_rng(0))
    assert np.all(out.sigma2_0 > 0) and np.all(np.isfinite(out.sigma2_0))
    assert np.all(out.sigma2_1 > 0) and np.all(np.isfinite(out.sigma2_1))
    assert np.all(out.tau2_0 > 0) and np.all(np.isfinite(out.tau2_0))
    assert np.all(out.tau2_1 > 0) and np.all(np.isfinite(out.tau2_1))
    assert np.all((out.q_01 >= 0) & (out.q_01 <= 1))
    assert np.all((out.q_10 >= 0) & (out.q_10 <= 1))
    assert np.all((out.theta_w >= 0) & (out.theta_w <= 1))
    assert set(np.unique(out.V[0])).issubset({0, 1})
    assert set(np.unique(out.Z[0])).issubset({0, 1})
    assert np.all(np.isfinite(out.log_marg))


def test_ipmcmc_z_initial_step_always_zero():
    mkt = _make_synth(T=60, seed=4)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, M=4, P=2, n_iter=8, n_burnin=2, seed=0)
    out = ipmcmc([md], cfg, rng=np.random.default_rng(0))
    assert np.all(out.Z[0][:, :, 0] == 0)


def test_ipmcmc_chain_indices_in_candidate_set():
    """chain_indices[it, j] ∈ {j} ∪ {P, ..., M-1} for every iter."""
    mkt = _make_synth(T=60, seed=4)
    md = _to_market_data(mkt)
    M, P = 4, 2
    cfg = InferenceConfig(N=20, M=M, P=P, n_iter=20, n_burnin=2, seed=0)
    out = ipmcmc([md], cfg, rng=np.random.default_rng(0))
    for j in range(P):
        allowed = {j, *range(P, M)}
        observed = set(out.chain_indices[:, j].tolist())
        assert observed.issubset(allowed), f"slot {j}: {observed} ⊄ {allowed}"


def test_ipmcmc_swap_step_actually_swaps():
    """Over enough iterations, the swap step should pull in at least one
    non-self chain index for at least one slot."""
    mkt = _make_synth(T=60, seed=4)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, M=4, P=2, n_iter=30, n_burnin=2, seed=0)
    out = ipmcmc([md], cfg, rng=np.random.default_rng(0))
    non_self = (out.chain_indices != np.arange(2)[None, :]).sum()
    assert non_self > 0


def test_ipmcmc_reproducibility():
    mkt = _make_synth(T=60, seed=5)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, M=4, P=2, n_iter=10, n_burnin=2, seed=0)
    out1 = ipmcmc([md], cfg, rng=np.random.default_rng(42))
    out2 = ipmcmc([md], cfg, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(out1.sigma2_0, out2.sigma2_0)
    np.testing.assert_array_equal(out1.chain_indices, out2.chain_indices)
    np.testing.assert_array_equal(out1.Z[0], out2.Z[0])
    np.testing.assert_array_equal(out1.theta_w, out2.theta_w)


def test_ipmcmc_does_not_mutate_caller_config():
    mkt = _make_synth(T=60, seed=5)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, M=4, P=2, n_iter=30, n_burnin=10, seed=0)
    before = (cfg.mh_step_beta_S, cfg.mh_step_beta_Z,
              cfg.mh_step_log_tau2_0, cfg.mh_step_log_tau2_1)
    _ = ipmcmc([md], cfg, rng=np.random.default_rng(0))
    after = (cfg.mh_step_beta_S, cfg.mh_step_beta_Z,
             cfg.mh_step_log_tau2_0, cfg.mh_step_log_tau2_1)
    assert before == after


def test_ipmcmc_multi_market():
    mkts = [_make_synth(T=50, seed=s) for s in (1, 2)]
    mds = [_to_market_data(m) for m in mkts]
    cfg = InferenceConfig(N=20, M=4, P=2, n_iter=8, seed=0)
    out = ipmcmc(mds, cfg, rng=np.random.default_rng(0), n_wallets=10)
    assert len(out.X) == 2
    assert out.log_marg.shape == (8, 4)
    assert out.theta_w.shape == (8, 2, 10)


def test_ipmcmc_m_equals_p_degenerates_to_parallel_pg():
    """With M == P there are no unconditional chains; every chain_index entry
    must be its own slot."""
    mkt = _make_synth(T=60, seed=6)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, M=2, P=2, n_iter=8, seed=0)
    out = ipmcmc([md], cfg, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(
        out.chain_indices, np.tile(np.arange(2), (8, 1))
    )


def test_ipmcmc_rejects_p_greater_than_m():
    mkt = _make_synth(T=40, seed=7)
    md = _to_market_data(mkt)
    bad_cfg = InferenceConfig(N=20, M=2, P=4, n_iter=4, seed=0)
    with pytest.raises(ValueError):
        ipmcmc([md], bad_cfg, rng=np.random.default_rng(0))


@pytest.mark.slow
def test_ipmcmc_z_posterior_recovers_insider_trades():
    """Headline §9 validation with iPMCMC: posterior P(Z_i=1 | D) should
    discriminate true insider trades. Pool over all P chains."""
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    p_true = ModelParams.warm_start(Y_dummy)
    mkt = generate_market(
        p_true, n_trades=150, n_wallets=20, n_insider_wallets=3,
        mean_inter_trade_time=1.0, rng=np.random.default_rng(11),
    )
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=30, M=4, P=2, n_iter=80, n_burnin=30, seed=0)
    out = ipmcmc([md], cfg, rng=np.random.default_rng(42))

    # Pool across post-burn-in iterations and both chains
    Z_post = out.Z[0][cfg.n_burnin:]                # (n_post, P, T)
    Z_prob = Z_post.reshape(-1, Z_post.shape[-1]).mean(axis=0)
    auc = _roc_auc(mkt.Z, Z_prob)
    assert auc > 0.65, f"AUC = {auc:.3f}; expected > 0.65"
