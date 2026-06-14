import numpy as np
import pytest

from config.default_params import InferenceConfig, ModelParams
from src.data.synthetic import generate_market
from src.inference.diagnostics import (
    PHI_PARAM_NAMES,
    compute_ess,
    compute_rhat,
    diagnose_ipmcmc,
    iPMCMCDiagnostics,
    particle_degeneracy_rate,
    smc_particle_degeneracy,
)
from src.inference.ipmcmc import ipmcmc
from src.inference.particle_gibbs import MarketData
from src.inference.smc import bootstrap_smc


def _to_market_data(mkt):
    return MarketData(
        Y=mkt.Y, delta=mkt.delta,
        log_size_ratio=np.log(mkt.S / mkt.S_bar),
        wallet_ids=mkt.wallet_ids,
    )


def _make_synth(*, T=80, n_wallets=10, seed=7):
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    p = ModelParams.warm_start(Y_dummy)
    return generate_market(
        p, n_trades=T, n_wallets=n_wallets, n_insider_wallets=2,
        mean_inter_trade_time=1.0, rng=np.random.default_rng(seed),
    )


# ---------------- compute_rhat ----------------

def test_rhat_iid_chains_close_to_one():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((2000, 4))   # (n_iter, P)
    r = compute_rhat(samples)
    assert 0.99 < r < 1.05


def test_rhat_single_chain_returns_nan():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((500, 1))
    assert np.isnan(compute_rhat(samples))


def test_rhat_diverging_chains_exceeds_one():
    """Chains drifting from different starting means → R-hat > 1.01."""
    rng = np.random.default_rng(0)
    P = 4
    n_iter = 500
    # Each chain is a random walk starting at a different mean
    starts = np.array([-3.0, -1.0, 1.0, 3.0])
    samples = np.empty((n_iter, P))
    for p in range(P):
        samples[:, p] = starts[p] + 0.1 * np.cumsum(rng.standard_normal(n_iter))
    r = compute_rhat(samples)
    assert r > 1.05


def test_rhat_handles_multi_dim_params():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((1000, 4, 5))   # 5 sub-params
    r = compute_rhat(samples)
    assert r.shape == (5,)
    assert np.all((r > 0.95) & (r < 1.10))


# ---------------- compute_ess ----------------

def test_ess_iid_samples_near_n_iter():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((4000, 4))
    e = compute_ess(samples)
    # 4 chains × 4000 ≈ 16000 effective draws for iid Gaussian
    assert e > 0.5 * 16000


def test_ess_ar1_samples_below_n_iter():
    """Strongly autocorrelated AR(1) gives ESS much less than n_iter."""
    rng = np.random.default_rng(0)
    n_iter = 4000
    P = 4
    phi = 0.95
    samples = np.empty((n_iter, P))
    samples[0] = rng.standard_normal(P)
    for t in range(1, n_iter):
        samples[t] = phi * samples[t - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal(P)
    e = compute_ess(samples)
    assert e < 0.2 * n_iter * P    # heavy correlation crushes ESS


def test_ess_accepts_1d_single_chain():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(2000)
    e = compute_ess(samples)
    assert 0.5 * 2000 < e < 1.5 * 2000


def test_ess_handles_multi_dim_params():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((1000, 4, 7))
    e = compute_ess(samples)
    assert e.shape == (7,)
    assert np.all(e > 1000)


# ---------------- particle_degeneracy_rate ----------------

def test_particle_degeneracy_rate_all_above_threshold():
    ess = np.full(100, 30.0)
    assert particle_degeneracy_rate(ess, N=50) == 0.0    # 30 >= 50/4


def test_particle_degeneracy_rate_all_below_threshold():
    ess = np.full(100, 5.0)
    assert particle_degeneracy_rate(ess, N=50) == 1.0    # 5 < 50/4 = 12.5


def test_particle_degeneracy_rate_mixed():
    ess = np.array([20.0, 5.0, 5.0, 20.0])
    # threshold = 50/4 = 12.5 → 2/4 below
    assert particle_degeneracy_rate(ess, N=50) == 0.5


def test_particle_degeneracy_rate_custom_threshold():
    ess = np.array([10.0, 20.0, 30.0, 40.0])
    # threshold = 0.5*50 = 25 → 2/4 below
    assert particle_degeneracy_rate(ess, N=50, threshold_fraction=0.5) == 0.5


def test_smc_particle_degeneracy_on_real_smc_output():
    mkt = _make_synth(T=60, seed=3)
    md = _to_market_data(mkt)
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    p = ModelParams.warm_start(Y_dummy)
    theta_w = rng.beta(p.a, p.b, size=10)
    cfg = InferenceConfig(N=30, seed=0)
    out = bootstrap_smc(
        md.Y, md.delta, md.log_size_ratio, md.wallet_ids,
        theta_w, p, cfg, rng=np.random.default_rng(0),
    )
    rate = smc_particle_degeneracy(out, N=cfg.N)
    assert 0.0 <= rate <= 1.0


# ---------------- diagnose_ipmcmc ----------------

def test_diagnose_ipmcmc_end_to_end():
    """diagnose_ipmcmc returns a populated dataclass on a small iPMCMC run."""
    mkt = _make_synth(T=80, seed=4)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, M=4, P=2, n_iter=30, n_burnin=10, seed=0)
    out = ipmcmc([md], cfg, rng=np.random.default_rng(0))

    diag = diagnose_ipmcmc(out, n_burnin=cfg.n_burnin)
    assert isinstance(diag, iPMCMCDiagnostics)
    assert set(diag.rhat.keys()) == set(PHI_PARAM_NAMES)
    assert set(diag.ess_bulk.keys()) == set(PHI_PARAM_NAMES)
    for name in PHI_PARAM_NAMES:
        # ESS must be positive; R-hat must be finite (R-hat of constant chains
        # can occur on β at iter 0 → finite if non-degenerate).
        assert diag.ess_bulk[name] > 0, name
    assert np.isfinite(diag.rhat_max)
    assert diag.ess_bulk_min > 0
    assert isinstance(diag.rhat_flagged, list)
    assert np.isfinite(diag.rhat_theta_w_max)
    assert diag.ess_bulk_theta_w_min > 0


def test_diagnose_ipmcmc_burnin_drop_changes_estimate():
    """Dropping more burn-in should change the R-hat/ESS estimates (not crash)."""
    mkt = _make_synth(T=60, seed=4)
    md = _to_market_data(mkt)
    cfg = InferenceConfig(N=20, M=4, P=2, n_iter=40, n_burnin=10, seed=0)
    out = ipmcmc([md], cfg, rng=np.random.default_rng(0))
    d0 = diagnose_ipmcmc(out, n_burnin=0)
    d_late = diagnose_ipmcmc(out, n_burnin=20)
    # Both runs should produce finite headline numbers.
    assert np.isfinite(d0.ess_bulk_min) and np.isfinite(d_late.ess_bulk_min)
    assert np.isfinite(d0.rhat_max) and np.isfinite(d_late.rhat_max)
