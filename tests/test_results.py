"""Unit tests for src/analysis/results.py."""

from __future__ import annotations

import numpy as np
import pytest

from config.default_params import InferenceConfig, ModelParams
from src.analysis.results import (
    _flatten_param,
    _is_ipmcmc,
    count_wallet_trades,
    credible_interval,
    flagged_trade_indices,
    insider_recall_at_k,
    kendall_theta_w,
    posterior_pi_mean,
    posterior_regime_probability,
    posterior_X_mean,
    posterior_Z_probability,
    recall_k_cutoff,
    roc_auc,
    roc_curve,
    spearman_theta_w,
    summarize_chain,
    wallet_ranking,
)
from src.data.preprocess import WalletIndex
from src.data.synthetic import generate_market
from src.inference.ipmcmc import ipmcmc
from src.inference.particle_gibbs import MarketData, particle_gibbs

# ---------------- Fixtures ----------------


@pytest.fixture
def warm_params():
    """ModelParams warm-started from 200 synthetic N(0,1) samples."""
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    return ModelParams.warm_start(Y_dummy)


@pytest.fixture
def synth_market(warm_params):
    """Small synthetic market: 80 trades, 10 wallets, 2 insiders."""
    return generate_market(
        warm_params,
        n_trades=80,
        n_wallets=10,
        n_insider_wallets=2,
        mean_inter_trade_time=1.0,
        rng=np.random.default_rng(3),
    )


def _to_md(mkt):
    """Wrap a SyntheticMarket into the MarketData inference struct."""
    return MarketData(
        Y=mkt.Y,
        delta=mkt.delta,
        log_size_ratio=np.log(mkt.S / mkt.S_bar),
        wallet_ids=mkt.wallet_ids,
    )


@pytest.fixture
def pg_output(synth_market):
    """PG chain: 15 iters (5 burn-in), N=20."""
    cfg = InferenceConfig(N=20, n_iter=15, n_burnin=5, seed=0)
    return particle_gibbs([_to_md(synth_market)], cfg, rng=np.random.default_rng(0))


@pytest.fixture
def ipmcmc_output(synth_market):
    """iPMCMC chain: 12 iters (4 burn-in), N=20, M=4, P=2."""
    cfg = InferenceConfig(N=20, M=4, P=2, n_iter=12, n_burnin=4, seed=0)
    return ipmcmc([_to_md(synth_market)], cfg, rng=np.random.default_rng(0))


# ---------------- Internal helpers ----------------


def test_is_ipmcmc_discriminates_output_types(pg_output, ipmcmc_output):
    """_is_ipmcmc returns True for iPMCMC output, False for PG."""
    assert _is_ipmcmc(ipmcmc_output) is True
    assert _is_ipmcmc(pg_output) is False


def test_flatten_param_is_identity_for_pg():
    """PG samples (n_iter, n_w) pass through unchanged."""
    s = np.arange(10.0).reshape(10, 1)
    out = _flatten_param(s, is_ipmcmc=False)
    np.testing.assert_array_equal(out, s)


def test_flatten_param_collapses_chain_axis_for_ipmcmc():
    """iPMCMC (n_iter, P, n_w) flattened to (n_iter*P, n_w)."""
    s = np.arange(2 * 3 * 4.0).reshape(2, 3, 4)  # (n_iter=2, P=3, n_w=4)
    out = _flatten_param(s, is_ipmcmc=True)
    assert out.shape == (6, 4)
    # First chain-major flatten: row 0 of out should equal s[0, 0]
    np.testing.assert_array_equal(out[0], s[0, 0])
    np.testing.assert_array_equal(out[1], s[0, 1])
    np.testing.assert_array_equal(out[3], s[1, 0])


# ---------------- Per-trade quantities ----------------


def test_posterior_Z_probability_shape_and_range(pg_output, synth_market):
    """Z probabilities have shape (T,) and lie in [0, 1]."""
    z = posterior_Z_probability(pg_output, market_idx=0, n_burnin=5)
    assert z.shape == (synth_market.Y.shape[0],)
    assert np.all((z >= 0.0) & (z <= 1.0))


def test_posterior_Z_probability_works_for_ipmcmc(ipmcmc_output, synth_market):
    """Z probabilities valid for iPMCMC output too."""
    z = posterior_Z_probability(ipmcmc_output, market_idx=0, n_burnin=4)
    assert z.shape == (synth_market.Y.shape[0],)
    assert np.all((z >= 0.0) & (z <= 1.0))


def test_posterior_regime_probability_shape_and_range(pg_output, synth_market):
    """Regime probabilities have shape (T,) and lie in [0, 1]."""
    v = posterior_regime_probability(pg_output, market_idx=0, n_burnin=5)
    assert v.shape == (synth_market.Y.shape[0],)
    assert np.all((v >= 0.0) & (v <= 1.0))


def test_posterior_pi_mean_in_unit_interval(pg_output, synth_market):
    """Price-process pi mean has shape (T,) strictly in (0, 1)."""
    pi = posterior_pi_mean(pg_output, market_idx=0, n_burnin=5)
    assert pi.shape == (synth_market.Y.shape[0],)
    assert np.all((pi > 0.0) & (pi < 1.0))


def test_posterior_X_mean_is_finite(pg_output, synth_market):
    """Latent price mean has shape (T,) with all finite values."""
    x = posterior_X_mean(pg_output, market_idx=0, n_burnin=5)
    assert x.shape == (synth_market.Y.shape[0],)
    assert np.all(np.isfinite(x))


# ---------------- Credible intervals + flagging ----------------


def test_credible_interval_covers_central_mass():
    """95% CI for N(0,1) covers near (-1.96, +1.96)."""
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((5000, 4))
    lo, hi = credible_interval(samples, alpha=0.05)
    assert lo.shape == (4,) and hi.shape == (4,)
    # 95% CI for N(0,1) should land near (-1.96, +1.96)
    assert np.all(lo < -1.5) and np.all(lo > -2.3)
    assert np.all(hi > 1.5) and np.all(hi < 2.3)


def test_flagged_trade_indices_threshold():
    """Indices above threshold returned in order."""
    z_prob = np.array([0.1, 0.6, 0.4, 0.9])
    idx = flagged_trade_indices(z_prob, threshold=0.5)
    np.testing.assert_array_equal(idx, [1, 3])


# ---------------- Wallet ranking ----------------


def test_wallet_ranking_columns_and_sort(pg_output, synth_market):
    """DataFrame has expected columns, sorted by posterior_mean desc."""
    idx = WalletIndex()
    for w in range(int(synth_market.wallet_ids.max()) + 1):
        idx.add(f"0xWALLET{w:03d}")
    n_trades = count_wallet_trades([synth_market.wallet_ids])

    df = wallet_ranking(
        pg_output,
        idx,
        n_burnin=5,
        n_trades_per_wallet=n_trades,
    )
    expected_cols = {
        "wallet_id",
        "wallet_address",
        "posterior_mean",
        "posterior_median",
        "ci_lo",
        "ci_hi",
        "n_trades",
    }
    assert set(df.columns) == expected_cols
    # Sorted by posterior_mean descending
    assert df["posterior_mean"].is_monotonic_decreasing
    # CI sanity
    assert (df["ci_lo"] <= df["posterior_mean"]).all()
    assert (df["posterior_mean"] <= df["ci_hi"]).all()
    # Total trades pooled across i>=1
    assert df["n_trades"].sum() == int(len(synth_market.wallet_ids) - 1)


def test_wallet_ranking_handles_ipmcmc(ipmcmc_output, synth_market):
    """wallet_ranking accepts iPMCMC output; rows equal n_wallets."""
    idx = WalletIndex()
    for w in range(int(synth_market.wallet_ids.max()) + 1):
        idx.add(f"0xW{w}")
    df = wallet_ranking(ipmcmc_output, idx, n_burnin=4)
    assert df.shape[0] == int(synth_market.wallet_ids.max()) + 1
    assert df["posterior_mean"].is_monotonic_decreasing


def test_count_wallet_trades_excludes_initial_trade():
    """First trade (i=0) excluded from per-wallet counts."""
    w1 = np.array([0, 1, 1, 2, 0])  # 5 trades; index 0 excluded
    w2 = np.array([2, 0, 0])  # 3 trades; index 0 excluded
    counts = count_wallet_trades([w1, w2], n_wallets=3)
    # Counts after excluding index 0 of each market:
    #   wallet 0: w1[4] + w2[1,2] = 1 + 2 = 3
    #   wallet 1: w1[1,2]         = 2
    #   wallet 2: w1[3]           = 1
    assert counts == {0: 3, 1: 2, 2: 1}


# ---------------- Chain summary ----------------


def test_summarize_chain_columns_and_phi_coverage(pg_output):
    """All phi params summarised; PG R-hat is NaN, ESS positive."""
    df = summarize_chain(pg_output, n_burnin=5)
    expected_phi = {
        "sigma2_0",
        "sigma2_1",
        "q_01",
        "q_10",
        "beta_S",
        "beta_Z",
        "tau2_0",
        "tau2_1",
    }
    assert set(df["parameter"]) == expected_phi
    # PG has one chain — R-hat is NaN
    assert df["rhat"].isna().all()
    assert (df["ess"] > 0).all()
    # CI consistent
    assert (df["ci_lo"] <= df["posterior_mean"]).all()
    assert (df["posterior_mean"] <= df["ci_hi"]).all()


def test_summarize_chain_returns_finite_rhat_for_ipmcmc(ipmcmc_output):
    """iPMCMC chains yield finite positive R-hat values."""
    df = summarize_chain(ipmcmc_output, n_burnin=4)
    assert df["rhat"].notna().all()
    assert (df["rhat"] > 0).all()


# ---------------- Insider recall@K ----------------


def test_recall_k_cutoff_top_decile_and_insiders():
    """K is at least the insider count and the top decile."""
    assert recall_k_cutoff(20, 3) == 3
    assert recall_k_cutoff(100, 3) == 10


def test_insider_recall_at_k_perfect_and_partial():
    """Recall@K counts insiders in the top-K score ranks."""
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    insiders = [0, 2, 4]
    assert insider_recall_at_k(scores, insiders, k=3) == pytest.approx(1.0)
    assert insider_recall_at_k(scores, insiders, k=2) == pytest.approx(2 / 3)


def test_insider_recall_at_k_empty_insiders():
    """Empty insider list yields recall 1.0."""
    scores = np.array([0.5, 0.1, 0.9])
    assert insider_recall_at_k(scores, [], k=2) == 1.0


# ---------------- ROC ----------------


def test_roc_auc_perfect_separation_is_one():
    """AUC=1 for perfectly separated scores."""
    z_true = np.array([0, 0, 0, 1, 1, 1])
    z_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert roc_auc(z_true, z_score) == pytest.approx(1.0)


def test_roc_auc_random_score_near_half():
    """Uninformative scores yield AUC near 0.5."""
    rng = np.random.default_rng(0)
    z_true = (rng.random(5000) < 0.3).astype(int)
    z_score = rng.random(5000)
    auc = roc_auc(z_true, z_score)
    assert 0.45 < auc < 0.55


def test_roc_auc_empty_class_returns_half():
    """All-same labels return AUC=0.5 (undefined → neutral)."""
    assert roc_auc(np.zeros(5), np.arange(5)) == 0.5
    assert roc_auc(np.ones(5), np.arange(5)) == 0.5


def test_roc_auc_ties_handled():
    """All-tied scores should give AUC = 0.5 regardless of label distribution."""
    z_true = np.array([0, 1, 0, 1])
    z_score = np.ones(4)
    assert roc_auc(z_true, z_score) == pytest.approx(0.5)


def test_roc_curve_monotonic_and_bounded():
    """FPR and TPR are non-decreasing and span [0, 1]."""
    rng = np.random.default_rng(0)
    z_true = (rng.random(500) < 0.3).astype(int)
    z_score = rng.random(500) + 0.6 * z_true
    fpr, tpr, _ = roc_curve(z_true, z_score)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == pytest.approx(1.0)
    assert tpr[-1] == pytest.approx(1.0)
    assert np.all(np.diff(fpr) >= -1e-9)  # non-decreasing
    assert np.all(np.diff(tpr) >= -1e-9)


def test_spearman_theta_w_perfect_and_reversed():
    """Spearman rho is ~1 for matching ranks and ~-1 when reversed."""
    theta = np.array([0.1, 0.3, 0.5, 0.9])
    assert spearman_theta_w(theta, theta) == pytest.approx(1.0)
    assert spearman_theta_w(theta, theta[::-1]) == pytest.approx(-1.0)


def test_spearman_theta_w_constant_returns_nan():
    """Constant inputs yield nan (degenerate ranks)."""
    theta = np.array([0.5, 0.5, 0.5])
    assert np.isnan(spearman_theta_w(theta, theta))


def test_kendall_theta_w_perfect_and_reversed():
    """Kendall tau is ~1 for matching ranks and ~-1 when reversed."""
    theta = np.array([0.1, 0.3, 0.5, 0.9])
    assert kendall_theta_w(theta, theta) == pytest.approx(1.0)
    assert kendall_theta_w(theta, theta[::-1]) == pytest.approx(-1.0)


def test_kendall_theta_w_constant_returns_nan():
    """Constant inputs yield nan (degenerate ranks)."""
    theta = np.array([0.5, 0.5, 0.5])
    assert np.isnan(kendall_theta_w(theta, theta))
