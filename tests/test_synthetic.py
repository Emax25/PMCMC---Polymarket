"""Tests for src.data.synthetic: market generation shapes and constraints."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from config.default_params import ModelParams, PhiPrior
from src.data.synthetic import (
    generate_dataset,
    generate_market,
    generate_prior_predictive_market,
    params_from_prior,
)


@pytest.fixture
def params():
    """Warm-started ModelParams derived from a 200-step dummy series."""
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    return ModelParams.warm_start(Y_dummy)


@pytest.fixture
def proper_prior():
    """PhiPrior with both variance blocks proper enough to sample from.

    The shipped tau2 default IG(1e-9, 1e-9) is improper (STATUS.md P11); SBC
    simulation must supply real hyperparameters, so tests do the same.
    """
    return PhiPrior(tau2_ig_alpha=2.0, tau2_ig_beta=1.0)


@pytest.fixture
def rng():
    """Default RNG for deterministic test execution."""
    return np.random.default_rng(42)


def test_shapes(params, rng):
    """All market arrays have the expected length-T shape."""
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
    """V and Z contain only values in {0, 1}."""
    mkt = generate_market(params, n_trades=200, rng=rng)
    assert set(np.unique(mkt.V)).issubset({0, 1})
    assert set(np.unique(mkt.Z)).issubset({0, 1})


def test_prices_in_unit_interval(params, rng):
    # Use short inter-trade time so the non-mean-reverting random walk stays bounded
    mkt = generate_market(params, n_trades=200, mean_inter_trade_time=1.0, rng=rng)
    assert np.all(mkt.p >= 0) and np.all(mkt.p <= 1)
    assert np.all(np.isfinite(mkt.Y))


def test_times_monotone_nonneg(params, rng):
    """delta[0]=0 by convention; all subsequent deltas and time diffs are positive."""
    mkt = generate_market(params, n_trades=200, rng=rng)
    assert mkt.delta[0] == 0.0
    assert np.all(mkt.delta[1:] > 0)
    assert np.all(np.diff(mkt.t) > 0)


def test_sizes_positive(params, rng):
    """Trade sizes S are strictly positive."""
    mkt = generate_market(params, n_trades=200, rng=rng)
    assert np.all(mkt.S > 0)


def test_wallet_ids_in_range(params, rng):
    """wallet_ids lie in [0, n_wallets)."""
    n_wallets = 30
    mkt = generate_market(params, n_trades=200, n_wallets=n_wallets, rng=rng)
    assert np.all(mkt.wallet_ids >= 0)
    assert np.all(mkt.wallet_ids < n_wallets)


def test_theta_w_in_unit_interval(params, rng):
    """Per-wallet propensities theta_w lie in [0, 1]."""
    mkt = generate_market(params, n_trades=200, rng=rng)
    assert np.all(mkt.theta_w >= 0) and np.all(mkt.theta_w <= 1)


def test_insider_wallets_have_high_propensity(params, rng):
    """Insider theta_w mean exceeds regular-wallet mean."""
    mkt = generate_market(params, n_trades=500, n_insider_wallets=5, rng=rng)
    insider_theta = mkt.theta_w[mkt.insider_wallet_ids]
    regular_theta = np.delete(mkt.theta_w, mkt.insider_wallet_ids)
    assert insider_theta.mean() > regular_theta.mean()


def test_z0_always_zero(params, rng):
    """Z_0 := 0 by model convention across multiple markets."""
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
    """generate_dataset returns a list of K markets each with the requested T."""
    K = 3
    dataset = generate_dataset(params, n_markets=K, n_trades=50, rng=rng)
    assert len(dataset) == K
    for mkt in dataset:
        assert mkt.Y.shape == (50,)


def test_reproducibility(params):
    """Same RNG seed produces bit-exact identical market data."""
    mkt1 = generate_market(params, n_trades=100, rng=np.random.default_rng(7))
    mkt2 = generate_market(params, n_trades=100, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(mkt1.Y, mkt2.Y)
    np.testing.assert_array_equal(mkt1.Z, mkt2.Z)


# ---------------- params_from_prior ----------------


def test_params_from_prior_in_domain(proper_prior):
    """500 prior draws stay in the parameter domain with no NaN/inf."""
    rng = np.random.default_rng(11)
    draws = [params_from_prior(proper_prior, rng) for _ in range(500)]
    variances = np.array([[d.sigma2_0, d.sigma2_1, d.tau2_0, d.tau2_1] for d in draws])
    q = np.array([[d.q_01, d.q_10] for d in draws])
    betas = np.array([[d.beta_S, d.beta_Z] for d in draws])
    assert np.all(np.isfinite(variances)) and np.all(variances > 0.0)
    assert np.all(np.isfinite(q)) and np.all((q > 0.0) & (q < 1.0))
    assert np.all(np.isfinite(betas))


def test_params_from_prior_betas_untruncated(proper_prior):
    """Cauchy beta draws keep their heavy tails — no clipping or rejection.

    P(|Cauchy(0, 2.5)| > 5) ~ 0.30, so over 1000 draws an all-|beta| <= 5 sample
    would be conclusive evidence of truncation, which breaks SBC rank uniformity.
    """
    rng = np.random.default_rng(12)
    draws = [params_from_prior(proper_prior, rng) for _ in range(500)]
    betas = np.array([[d.beta_S, d.beta_Z] for d in draws])
    assert np.max(np.abs(betas)) > 5.0


def test_params_from_prior_fixed_hyperparameters(proper_prior):
    """Non-inferred fields (a, b, gamma, s0_2) keep their ModelParams defaults."""
    drawn = params_from_prior(proper_prior, np.random.default_rng(3))
    defaults = ModelParams()
    assert (drawn.a, drawn.b) == (defaults.a, defaults.b)
    assert (drawn.gamma, drawn.s0_2) == (defaults.gamma, defaults.s0_2)


def test_params_from_prior_deterministic(proper_prior):
    """Same seed produces bit-exact identical prior draws."""
    p1 = params_from_prior(proper_prior, np.random.default_rng(5))
    p2 = params_from_prior(proper_prior, np.random.default_rng(5))
    assert p1 == p2


def test_params_from_prior_rejects_improper_default():
    """The shipped PhiPrior tau2 block is improper and must be refused (P11)."""
    with pytest.raises(ValueError, match="P11"):
        params_from_prior(PhiPrior(), np.random.default_rng(0))


def test_params_from_prior_rejects_improper_sigma2(proper_prior):
    """The sigma2 block is guarded on the same footing as tau2."""
    improper = replace(proper_prior, sigma2_ig_alpha=1e-9, sigma2_ig_beta=1e-9)
    with pytest.raises(ValueError, match="P11"):
        params_from_prior(improper, np.random.default_rng(0))


# ---------------- prior-predictive generation ----------------


def test_prior_predictive_plants_no_insiders(params, rng):
    """No forced high-propensity cluster: theta_w is a plain Beta(a, b) sample."""
    mkt = generate_prior_predictive_market(params, n_trades=200, n_wallets=50, rng=rng)
    assert mkt.insider_wallet_ids == []
    # Planted wallets would sit near Beta(9, 1)'s 0.9 mean; Beta(1, 19) draws
    # exceeding 0.5 have probability ~2e-6 each.
    assert np.max(mkt.theta_w[:5]) < 0.5


def test_prior_predictive_theta_matches_beta_prior(params, rng):
    """Empirical mean theta_w matches the Beta(a, b) prior mean a/(a+b)."""
    n_wallets = 4000
    mkt = generate_prior_predictive_market(
        params, n_trades=50, n_wallets=n_wallets, rng=rng
    )
    expected = params.a / (params.a + params.b)
    assert mkt.theta_w.mean() == pytest.approx(expected, abs=0.005)


def test_prior_predictive_wallet_assignment_uniform(params, rng):
    """Trade counts are uniform over wallets — no insider frequency upweighting."""
    n_wallets, T = 10, 5000
    mkt = generate_prior_predictive_market(
        params, n_trades=T, n_wallets=n_wallets, rng=rng
    )
    counts = np.bincount(mkt.wallet_ids, minlength=n_wallets)
    # Under uniform assignment each wallet's share is 1/10; the first five hold
    # ~half the trades. Binomial sd of that share is ~0.007, so 0.05 is >> 5 sd.
    assert counts[:5].sum() / T == pytest.approx(0.5, abs=0.05)


def test_planted_mode_upweights_insider_trades(params, rng):
    """Contrast: the planted mode does skew trade counts toward insiders."""
    n_wallets, T = 10, 5000
    mkt = generate_market(
        params, n_trades=T, n_wallets=n_wallets, n_insider_wallets=5, rng=rng
    )
    counts = np.bincount(mkt.wallet_ids, minlength=n_wallets)
    # Weights 3 vs 1 over 5 vs 5 wallets => insider share 15/20 = 0.75.
    assert counts[:5].sum() / T > 0.6


def test_prior_predictive_rejects_insider_override(params, rng):
    """Planting insiders through the prior-predictive entry point is an error."""
    with pytest.raises(ValueError, match="n_insider_wallets"):
        generate_prior_predictive_market(
            params, n_trades=50, n_insider_wallets=3, rng=rng
        )


def test_prior_predictive_matches_generate_market_zero_insiders(params):
    """The entry point is a pure alias for n_insider_wallets=0 — same stream."""
    mkt1 = generate_prior_predictive_market(
        params, n_trades=100, rng=np.random.default_rng(9)
    )
    mkt2 = generate_market(
        params, n_trades=100, n_insider_wallets=0, rng=np.random.default_rng(9)
    )
    np.testing.assert_array_equal(mkt1.Y, mkt2.Y)
    np.testing.assert_array_equal(mkt1.theta_w, mkt2.theta_w)


# ---------------- Anonymous mode (Kalshi variant) ----------------

# logit(0.05) — the Beta(1, 19) prior mean, i.e. the base rate the wallet-mode
# hierarchy shrinks toward, expressed as the anonymous intercept.
ANON_ALPHA = float(np.log(0.05 / 0.95))


@pytest.fixture
def anon_params(params):
    """Anonymous-mode params: no wallet layer, per-market intercept `alpha`."""
    return replace(params, anonymous=True, alpha=ANON_ALPHA)


def test_anonymous_market_has_no_wallet_layer(anon_params, rng):
    """Anonymous generation emits no propensities, insiders or real wallet ids."""
    mkt = generate_market(anon_params, n_trades=200, n_wallets=20, rng=rng)

    assert mkt.theta_w.shape == (0,)
    assert mkt.insider_wallet_ids == []
    np.testing.assert_array_equal(mkt.wallet_ids, np.zeros(200, dtype=int))
    # Everything else keeps the wallet-mode contract, so downstream code needs
    # no anonymous-mode special case.
    assert mkt.Y.shape == mkt.Z.shape == mkt.V.shape == (200,)
    assert mkt.Z[0] == 0


def test_anonymous_wallet_kwargs_are_inert(anon_params):
    """`n_wallets`/`n_insider_wallets` cannot change an anonymous market."""
    mkt_a = generate_market(
        anon_params, n_trades=150, n_wallets=5, n_insider_wallets=0,
        rng=np.random.default_rng(11),
    )
    mkt_b = generate_market(
        anon_params, n_trades=150, n_wallets=500, n_insider_wallets=99,
        rng=np.random.default_rng(11),
    )
    np.testing.assert_array_equal(mkt_a.Y, mkt_b.Y)
    np.testing.assert_array_equal(mkt_a.Z, mkt_b.Z)


def test_anonymous_base_rate_tracks_alpha(params):
    """With both slopes zero, P(Z=1) is exactly sigmoid(alpha), not sigmoid(theta_w).

    The whole point of the anonymous variant: `alpha` — and nothing else —
    carries the insider base rate. Sampling noise at T = 4000 and a 5% rate is
    ~0.0034, so the 0.015 band is ~4 standard errors.
    """
    anon = replace(params, anonymous=True, alpha=ANON_ALPHA, beta_S=0.0, beta_Z=0.0)
    mkt = generate_market(anon, n_trades=4000, rng=np.random.default_rng(3))

    # Z[0] is pinned to 0 by convention and is excluded from the rate.
    observed = float(mkt.Z[1:].mean())
    assert abs(observed - 0.05) < 0.015, f"base rate {observed:.4f} != 0.05"


def test_warm_start_anonymous_seeds_alpha_at_the_prior_mean(params):
    """`warm_start(anonymous=True)` starts the level where theta_w starts."""
    rng = np.random.default_rng(0)
    Y = rng.standard_normal(200)
    anon = ModelParams.warm_start(Y, anonymous=True)

    assert anon.anonymous
    assert anon.alpha == pytest.approx(ANON_ALPHA, abs=1e-12)
    # The wallet-mode default is untouched — including `alpha`, which is not a
    # parameter there.
    assert not params.anonymous
    assert params.alpha == 0.0
    for field in ("sigma2_0", "sigma2_1", "tau2_0", "tau2_1"):
        assert getattr(anon, field) == getattr(params, field)
