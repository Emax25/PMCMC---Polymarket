from dataclasses import replace

import numpy as np
import pytest

from config.default_params import InferenceConfig, ModelParams
from src.data.synthetic import generate_market
from src.inference.parameter_updates import (
    GibbsSweepDiag,
    MarketLatents,
    gibbs_sweep,
    update_beta,
    update_q,
    update_sigma2,
    update_tau2,
    update_theta_w,
)


@pytest.fixture
def params():
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    return ModelParams.warm_start(Y_dummy)


@pytest.fixture
def config():
    return InferenceConfig(N=50, seed=42)


def _make_market(*, T=200, n_wallets=20, seed=7):
    """Generate a synthetic market and bundle it as a MarketLatents using
    the ground-truth latents."""
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    p = ModelParams.warm_start(Y_dummy)
    mkt = generate_market(
        p, n_trades=T, n_wallets=n_wallets, n_insider_wallets=3,
        mean_inter_trade_time=1.0, rng=np.random.default_rng(seed),
    )
    state = MarketLatents(
        Y=mkt.Y, delta=mkt.delta,
        log_size_ratio=np.log(mkt.S / mkt.S_bar),
        wallet_ids=mkt.wallet_ids,
        X=mkt.X.astype(float),
        V=mkt.V.astype(np.int8),
        Z=mkt.Z.astype(np.int8),
    )
    return mkt, state, p


# -------------------- update_sigma2 --------------------

def test_update_sigma2_recovers_truth():
    """With known X, V trajectories, the Inv-Gamma posterior concentrates on truth."""
    rng = np.random.default_rng(0)
    T = 2000
    sigma2_true = np.array([0.1, 1.0])
    delta = np.ones(T)
    delta[0] = 0.0
    # 50/50 regime mix so both N_v are large
    V = rng.integers(0, 2, size=T).astype(np.int8)
    X = np.empty(T)
    X[0] = rng.normal()
    for i in range(1, T):
        X[i] = X[i - 1] + np.sqrt(sigma2_true[V[i]] * delta[i]) * rng.standard_normal()

    market = MarketLatents(
        Y=np.zeros(T), delta=delta, log_size_ratio=np.zeros(T),
        wallet_ids=np.zeros(T, dtype=np.int64),
        X=X, V=V, Z=np.zeros(T, dtype=np.int8),
    )

    samples = np.array([
        update_sigma2([market], np.random.default_rng(s)) for s in range(200)
    ])
    assert abs(samples[:, 0].mean() - sigma2_true[0]) < 0.02
    assert abs(samples[:, 1].mean() - sigma2_true[1]) < 0.15


def test_update_sigma2_handles_delta_zero_steps():
    """Same-second trades (Δ_i = 0) on real data must be masked, not divided
    by — otherwise the inverse-Gamma posterior parameters become NaN."""
    T = 30
    rng = np.random.default_rng(0)
    delta = np.r_[0.0, rng.exponential(1.0, T - 1)]
    delta[5] = 0.0     # inject a same-second trade
    delta[18] = 0.0
    V = np.zeros(T, dtype=np.int8)
    X = np.cumsum(np.r_[0.0, rng.standard_normal(T - 1)])
    market = MarketLatents(
        Y=np.zeros(T), delta=delta, log_size_ratio=np.zeros(T),
        wallet_ids=np.zeros(T, dtype=np.int64),
        X=X, V=V, Z=np.zeros(T, dtype=np.int8),
    )
    s0, _ = update_sigma2([market], np.random.default_rng(0))
    assert np.isfinite(s0) and s0 > 0


def test_update_sigma2_handles_empty_regime():
    """If V has no occurrences of one regime, the update falls back to the prior."""
    T = 50
    market = MarketLatents(
        Y=np.zeros(T), delta=np.ones(T), log_size_ratio=np.zeros(T),
        wallet_ids=np.zeros(T, dtype=np.int64),
        X=np.cumsum(np.r_[0.0, np.random.default_rng(0).standard_normal(T - 1)]),
        V=np.zeros(T, dtype=np.int8),       # never enters regime 1
        Z=np.zeros(T, dtype=np.int8),
    )
    s0, s1 = update_sigma2([market], np.random.default_rng(0))
    assert s0 > 0 and np.isfinite(s0)
    assert s1 > 0 and np.isfinite(s1)


# -------------------- update_q --------------------

def test_update_q_recovers_truth():
    rng = np.random.default_rng(0)
    T = 8000
    q_true = (0.05, 0.5)
    V = np.zeros(T, dtype=np.int8)
    rho = q_true[0] / (q_true[0] + q_true[1])
    V[0] = int(rng.random() < rho)
    for i in range(1, T):
        flip = q_true[0] if V[i - 1] == 0 else q_true[1]
        V[i] = (1 - V[i - 1]) if rng.random() < flip else V[i - 1]

    market = MarketLatents(
        Y=np.zeros(T), delta=np.ones(T), log_size_ratio=np.zeros(T),
        wallet_ids=np.zeros(T, dtype=np.int64),
        X=np.zeros(T), V=V, Z=np.zeros(T, dtype=np.int8),
    )

    samples = np.array([
        update_q([market], np.random.default_rng(s)) for s in range(200)
    ])
    assert abs(samples[:, 0].mean() - q_true[0]) < 0.02
    assert abs(samples[:, 1].mean() - q_true[1]) < 0.05


# -------------------- update_theta_w --------------------

def test_update_theta_w_recovers_truth():
    rng = np.random.default_rng(0)
    T = 30000
    n_wallets = 10
    theta_true = rng.beta(2.0, 5.0, size=n_wallets)
    wallet_ids = rng.integers(0, n_wallets, size=T)
    Z = (rng.random(T) < theta_true[wallet_ids]).astype(np.int8)
    Z[0] = 0

    market = MarketLatents(
        Y=np.zeros(T), delta=np.ones(T), log_size_ratio=np.zeros(T),
        wallet_ids=wallet_ids,
        X=np.zeros(T), V=np.zeros(T, dtype=np.int8), Z=Z,
    )

    theta_mean = np.mean(
        [update_theta_w([market], n_wallets, 1.0, 1.0, np.random.default_rng(s))
         for s in range(200)],
        axis=0,
    )
    assert np.mean(np.abs(theta_mean - theta_true)) < 0.02


def test_update_theta_w_unobserved_wallet_uses_prior():
    """A wallet that never trades draws from the Beta(a, b) prior."""
    T = 100
    n_wallets = 5
    wallet_ids = np.zeros(T, dtype=np.int64)  # only wallet 0 trades
    Z = np.zeros(T, dtype=np.int8)
    market = MarketLatents(
        Y=np.zeros(T), delta=np.ones(T), log_size_ratio=np.zeros(T),
        wallet_ids=wallet_ids, X=np.zeros(T),
        V=np.zeros(T, dtype=np.int8), Z=Z,
    )
    a, b = 2.0, 8.0  # prior mean 0.2
    samples = np.array([
        update_theta_w([market], n_wallets, a, b, np.random.default_rng(s))[4]
        for s in range(2000)
    ])
    # Unobserved wallet posterior == prior == Beta(2, 8); mean = 0.2
    assert abs(samples.mean() - a / (a + b)) < 0.02


def test_update_theta_w_pools_across_markets():
    """The posterior for a shared wallet should use Z counts from every market."""
    n_wallets = 3
    a, b = 1.0, 1.0
    T = 100
    # Market 1: wallet 0 trades 100 times, Z always 1
    m1 = MarketLatents(
        Y=np.zeros(T), delta=np.ones(T), log_size_ratio=np.zeros(T),
        wallet_ids=np.zeros(T, dtype=np.int64),
        X=np.zeros(T), V=np.zeros(T, dtype=np.int8),
        Z=np.r_[0, np.ones(T - 1)].astype(np.int8),
    )
    # Market 2: wallet 0 trades 100 times, Z always 0
    m2 = MarketLatents(
        Y=np.zeros(T), delta=np.ones(T), log_size_ratio=np.zeros(T),
        wallet_ids=np.zeros(T, dtype=np.int64),
        X=np.zeros(T), V=np.zeros(T, dtype=np.int8),
        Z=np.zeros(T, dtype=np.int8),
    )
    # Pooled: 99 ones + 99 zeros (i=0 dropped from each) → Beta(1+99, 1+99) ≈ 0.5
    samples = np.array([
        update_theta_w([m1, m2], n_wallets, a, b, np.random.default_rng(s))[0]
        for s in range(500)
    ])
    assert 0.45 < samples.mean() < 0.55


# -------------------- update_beta --------------------

def test_update_beta_outputs_finite(params, config):
    _, state, _ = _make_market()
    bS, bZ, acc_S, acc_Z = update_beta(
        params.beta_S, params.beta_Z, [state],
        theta_w=np.full(20, 0.5),
        config=config, rng=np.random.default_rng(0),
    )
    assert np.isfinite(bS) and np.isfinite(bZ)
    assert isinstance(acc_S, bool) and isinstance(acc_Z, bool)


def test_update_beta_zero_step_always_accepts(params, config):
    _, state, _ = _make_market()
    cfg = replace(config, mh_step_beta_S=0.0, mh_step_beta_Z=0.0)
    bS, bZ, acc_S, acc_Z = update_beta(
        0.3, -0.2, [state],
        theta_w=np.full(20, 0.5),
        config=cfg, rng=np.random.default_rng(0),
    )
    # Proposal == current ⇒ log-ratio = 0 ⇒ accept with probability 1
    assert acc_S is True and acc_Z is True
    assert bS == 0.3 and bZ == -0.2


def test_update_beta_huge_step_mostly_rejects(params, config):
    _, state, _ = _make_market(T=500)
    cfg = replace(config, mh_step_beta_S=10.0, mh_step_beta_Z=10.0)
    accs = []
    for s in range(200):
        _, _, acc_S, _ = update_beta(
            0.0, 0.0, [state],
            theta_w=np.full(20, 0.5),
            config=cfg, rng=np.random.default_rng(s),
        )
        accs.append(acc_S)
    rate = np.mean(accs)
    assert rate < 0.10  # huge step → almost everything rejected


def test_update_beta_reproducible(params, config):
    _, state, _ = _make_market()
    args = (params.beta_S, params.beta_Z, [state], np.full(20, 0.5), config)
    o1 = update_beta(*args, rng=np.random.default_rng(123))
    o2 = update_beta(*args, rng=np.random.default_rng(123))
    assert o1 == o2


# -------------------- update_tau2 --------------------

def test_update_tau2_outputs_finite_and_positive(params, config):
    _, state, _ = _make_market()
    t0, t1, acc0, acc1 = update_tau2(
        params.tau2_0, params.tau2_1, [state], params.gamma, config,
        rng=np.random.default_rng(0),
    )
    assert t0 > 0 and t1 > 0
    assert np.isfinite(t0) and np.isfinite(t1)


def test_update_tau2_zero_step_always_accepts(params, config):
    _, state, _ = _make_market()
    cfg = replace(config, mh_step_log_tau2_0=0.0, mh_step_log_tau2_1=0.0)
    t0, t1, acc0, acc1 = update_tau2(
        0.7, 0.005, [state], params.gamma, cfg,
        rng=np.random.default_rng(0),
    )
    assert acc0 is True and acc1 is True
    assert t0 == 0.7 and t1 == 0.005


def test_update_tau2_huge_step_mostly_rejects(params, config):
    _, state, _ = _make_market(T=500)
    cfg = replace(config, mh_step_log_tau2_0=3.0, mh_step_log_tau2_1=3.0)
    accs = []
    for s in range(200):
        _, _, acc0, _ = update_tau2(
            params.tau2_0, params.tau2_1, [state], params.gamma, cfg,
            rng=np.random.default_rng(s),
        )
        accs.append(acc0)
    assert np.mean(accs) < 0.15


# -------------------- gibbs_sweep --------------------

def test_gibbs_sweep_returns_valid_state(params, config):
    _, state, _ = _make_market()
    theta_w_in = np.full(20, 0.1)
    new_params, theta_w_new, diag = gibbs_sweep(
        params, theta_w_in, [state], config, rng=np.random.default_rng(0),
    )
    assert isinstance(new_params, ModelParams)
    assert isinstance(diag, GibbsSweepDiag)
    assert theta_w_new.shape == (20,)
    assert np.all((theta_w_new >= 0) & (theta_w_new <= 1))
    for fld in ("sigma2_0", "sigma2_1", "tau2_0", "tau2_1"):
        v = getattr(new_params, fld)
        assert v > 0 and np.isfinite(v), fld
    for fld in ("q_01", "q_10"):
        v = getattr(new_params, fld)
        assert 0.0 <= v <= 1.0, fld
    assert np.isfinite(new_params.beta_S)
    assert np.isfinite(new_params.beta_Z)


def test_gibbs_sweep_reproducible(params, config):
    _, state, _ = _make_market()
    theta_w_in = np.full(20, 0.1)
    p1, t1, d1 = gibbs_sweep(params, theta_w_in, [state], config,
                              rng=np.random.default_rng(7))
    p2, t2, d2 = gibbs_sweep(params, theta_w_in, [state], config,
                              rng=np.random.default_rng(7))
    assert p1 == p2
    np.testing.assert_array_equal(t1, t2)
    assert d1 == d2


def test_gibbs_sweep_chain_recovers_sigma2():
    """Iterating gibbs_sweep with fixed truth-latents should leave (σ²_0, σ²_1)
    concentrated near the truth."""
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    p_true = ModelParams.warm_start(Y_dummy)
    mkt = generate_market(
        p_true, n_trades=2000, n_wallets=20, n_insider_wallets=3,
        mean_inter_trade_time=1.0, rng=np.random.default_rng(11),
    )
    state = MarketLatents(
        Y=mkt.Y, delta=mkt.delta,
        log_size_ratio=np.log(mkt.S / mkt.S_bar),
        wallet_ids=mkt.wallet_ids,
        X=mkt.X.astype(float), V=mkt.V.astype(np.int8),
        Z=mkt.Z.astype(np.int8),
    )
    # Initialize far from truth
    p_init = replace(p_true, sigma2_0=10.0, sigma2_1=10.0, tau2_0=10.0, tau2_1=10.0,
                     beta_S=0.0, beta_Z=0.0, q_01=0.5, q_10=0.5)
    theta_w = np.full(20, 0.5)
    cfg = InferenceConfig(N=50, seed=0)

    p = p_init
    s0_samples, s1_samples = [], []
    rng_chain = np.random.default_rng(42)
    for it in range(200):
        p, theta_w, _ = gibbs_sweep(p, theta_w, [state], cfg, rng_chain)
        if it >= 100:                       # discard burn-in
            s0_samples.append(p.sigma2_0)
            s1_samples.append(p.sigma2_1)
    assert abs(np.mean(s0_samples) - p_true.sigma2_0) / p_true.sigma2_0 < 0.20
    assert abs(np.mean(s1_samples) - p_true.sigma2_1) / p_true.sigma2_1 < 0.20
