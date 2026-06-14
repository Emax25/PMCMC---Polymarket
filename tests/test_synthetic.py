import numpy as np
import pytest

from config.default_params import ModelParams
from src.data.synthetic import generate_market, generate_dataset


@pytest.fixture
def params():
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    return ModelParams.warm_start(Y_dummy)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_shapes(params, rng):
    T, W = 100, 20
    mkt = generate_market(params, n_trades=T, n_wallets=W, rng=rng)
    assert mkt.X.shape == (T,)
    assert mkt.V.shape == (T,)
    assert mkt.Z.shape == (T,)
    assert mkt.Y.shape == (T,)
    assert mkt.p.shape == (T,)
    assert mkt.S.shape == (T,)
    assert mkt.t.shape == (T,)
    assert mkt.delta.shape == (T,)
    assert mkt.wallet_ids.shape == (T,)
    assert mkt.theta_w.shape == (W,)


def test_binary_indicators(params, rng):
    mkt = generate_market(params, n_trades=200, rng=rng)
    assert set(np.unique(mkt.V)).issubset({0, 1})
    assert set(np.unique(mkt.Z)).issubset({0, 1})


def test_prices_in_unit_interval(params, rng):
    # Use short inter-trade time so the non-mean-reverting random walk stays bounded
    mkt = generate_market(params, n_trades=200, mean_inter_trade_time=1.0, rng=rng)
    assert np.all(mkt.p >= 0) and np.all(mkt.p <= 1)
    assert np.all(np.isfinite(mkt.Y))


def test_times_monotone_nonneg(params, rng):
    mkt = generate_market(params, n_trades=200, rng=rng)
    assert mkt.delta[0] == 0.0
    assert np.all(mkt.delta[1:] > 0)
    assert np.all(np.diff(mkt.t) > 0)


def test_sizes_positive(params, rng):
    mkt = generate_market(params, n_trades=200, rng=rng)
    assert np.all(mkt.S > 0)


def test_wallet_ids_in_range(params, rng):
    n_wallets = 30
    mkt = generate_market(params, n_trades=200, n_wallets=n_wallets, rng=rng)
    assert np.all(mkt.wallet_ids >= 0)
    assert np.all(mkt.wallet_ids < n_wallets)


def test_theta_w_in_unit_interval(params, rng):
    mkt = generate_market(params, n_trades=200, rng=rng)
    assert np.all(mkt.theta_w >= 0) and np.all(mkt.theta_w <= 1)


def test_insider_wallets_have_high_propensity(params, rng):
    mkt = generate_market(params, n_trades=500, n_insider_wallets=5, rng=rng)
    insider_theta = mkt.theta_w[mkt.insider_wallet_ids]
    regular_theta = np.delete(mkt.theta_w, mkt.insider_wallet_ids)
    assert insider_theta.mean() > regular_theta.mean()


def test_z0_always_zero(params, rng):
    for _ in range(10):
        mkt = generate_market(params, n_trades=50, rng=rng)
        assert mkt.Z[0] == 0


def test_obs_variance_tighter_for_insiders(params, rng):
    # Insider trades (Z=1) should on average be closer to X than non-insider trades
    mkt = generate_market(params, n_trades=1000, n_insider_wallets=10, rng=rng)
    residuals = np.abs(mkt.Y - mkt.X)
    insider_mask = mkt.Z == 1
    if insider_mask.sum() > 10 and (~insider_mask).sum() > 10:
        assert residuals[insider_mask].mean() < residuals[~insider_mask].mean()


def test_generate_dataset(params, rng):
    K = 3
    dataset = generate_dataset(params, n_markets=K, n_trades=50, rng=rng)
    assert len(dataset) == K
    for mkt in dataset:
        assert mkt.Y.shape == (50,)


def test_reproducibility(params):
    mkt1 = generate_market(params, n_trades=100, rng=np.random.default_rng(7))
    mkt2 = generate_market(params, n_trades=100, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(mkt1.Y, mkt2.Y)
    np.testing.assert_array_equal(mkt1.Z, mkt2.Z)
