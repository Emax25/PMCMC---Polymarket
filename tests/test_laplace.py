"""Tests for src.inference.laplace (PhiPosterior Laplace layer, plan U1)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from config.default_params import InferenceConfig, ModelParams, PhiPrior
from src.data.synthetic import generate_market
from src.inference.laplace import (
    PhiPosterior,
    _block_cov_from_precision,
    _to_constrained,
    _to_unconstrained,
    laplace_from_vem,
)
from src.inference.particle_gibbs import MarketData
from src.inference.variational_em import variational_em

_DIMS = ("sigma2_0", "sigma2_1", "q_01", "q_10", "beta_S", "beta_Z", "tau2_0", "tau2_1")


def _make_synth(*, T, n_wallets=15, n_insider=3, seed=3, beta_S=0.0, beta_Z=0.0):
    rng = np.random.default_rng(0)
    base = ModelParams.warm_start(rng.standard_normal(200))
    params = replace(base, beta_S=beta_S, beta_Z=beta_Z)
    mkt = generate_market(
        params,
        n_trades=T,
        n_wallets=n_wallets,
        n_insider_wallets=n_insider,
        mean_inter_trade_time=1.0,
        rng=np.random.default_rng(seed),
    )
    md = MarketData(
        Y=mkt.Y,
        delta=mkt.delta,
        log_size_ratio=np.log(mkt.S / mkt.S_bar),
        wallet_ids=mkt.wallet_ids,
    )
    return md, params


def _fit(md, params, *, T_wallets=15, n_iter=15, estimate_betas=False):
    return variational_em(
        [md],
        InferenceConfig(N=20),
        n_wallets=T_wallets,
        params_init=params,
        n_iter=n_iter,
        estimate_betas=estimate_betas,
    )


def _vem_point(params):
    return np.array(
        [
            params.sigma2_0,
            params.sigma2_1,
            params.q_01,
            params.q_10,
            params.beta_S,
            params.beta_Z,
            params.tau2_0,
            params.tau2_1,
        ]
    )


# ---------------- transforms (R1) ----------------


def test_transforms_round_trip_machine_precision():
    """Constrained<->unconstrained transforms invert to machine precision."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        phi = np.empty(8)
        phi[[0, 1, 6, 7]] = np.exp(rng.uniform(-6, 4, 4))  # positive variances
        phi[[2, 3]] = rng.uniform(1e-4, 1.0 - 1e-4, 2)  # probabilities
        phi[[4, 5]] = rng.uniform(-5, 5, 2)  # real betas
        rt = _to_constrained(_to_unconstrained(phi))
        assert np.max(np.abs(rt - phi)) < 1e-12


def test_transforms_round_trip_batched():
    """Transforms round-trip on a batch of vectors (shape preserved)."""
    rng = np.random.default_rng(1)
    u = rng.normal(0, 2, size=(50, 8))
    phi = _to_constrained(u)
    assert phi.shape == (50, 8)
    assert np.all(phi[:, [0, 1, 6, 7]] > 0)
    assert np.all((phi[:, [2, 3]] > 0) & (phi[:, [2, 3]] < 1))
    np.testing.assert_allclose(_to_unconstrained(phi), u, atol=1e-10)


# ---------------- structure / named dims (R1) ----------------


def test_phi_posterior_shapes_and_named_dims():
    """PhiPosterior has (8,) mean, (8,8) block-diagonal cov, named dimensions."""
    md, params = _make_synth(T=200)
    out = _fit(md, params)
    post = laplace_from_vem(out, [md])
    assert post.mean_u.shape == (8,)
    assert post.cov_u.shape == (8, 8)
    assert post.dims == _DIMS
    # Block-diagonal: only the (beta_S, beta_Z) 2x2 and the 6 scalar diagonals
    # are non-zero; every other off-diagonal is exactly zero.
    off = post.cov_u.copy()
    off[np.diag_indices(8)] = 0.0
    off[4, 5] = off[5, 4] = 0.0
    assert np.count_nonzero(off) == 0
    assert np.all(np.linalg.eigvalsh(post.cov_u) > 0.0)  # valid covariance


def test_mean_u_is_transformed_vem_point():
    """The posterior mean equals the unconstrained VEM point estimate."""
    md, params = _make_synth(T=200)
    out = _fit(md, params)
    post = laplace_from_vem(out, [md])
    np.testing.assert_allclose(
        post.mean_u, _to_unconstrained(_vem_point(out.params)), atol=1e-12
    )


# ---------------- sampling sanity (scenario 4) ----------------


def test_sample_means_within_two_posterior_sds():
    """sample(2000) back-transformed means fall within ~2 posterior sds of the point."""
    md, params = _make_synth(T=300)
    out = _fit(md, params)
    post = laplace_from_vem(out, [md])

    draws = post.sample(np.random.default_rng(0), 2000)
    assert draws.shape == (2000, 8)
    sample_mean = draws.mean(axis=0)
    sample_sd = draws.std(axis=0)
    point = _vem_point(out.params)
    # 2 posterior sds (constrained scale); the small extra slack absorbs the
    # log-/logit-normal Jensen shift plus Monte-Carlo error at n=2000.
    assert np.all(np.abs(sample_mean - point) < 2.0 * sample_sd + 1e-6)


def test_sample_respects_support():
    """Every draw has positive variances and q in (0, 1)."""
    md, params = _make_synth(T=200)
    out = _fit(md, params)
    post = laplace_from_vem(out, [md])
    draws = post.sample(np.random.default_rng(2), 1000)
    assert np.all(draws[:, [0, 1, 6, 7]] > 0.0)
    assert np.all((draws[:, [2, 3]] > 0.0) & (draws[:, [2, 3]] < 1.0))


def test_logpdf_finite_and_peaks_near_mean():
    """logpdf is finite and larger at the mean than a displaced point."""
    md, params = _make_synth(T=200)
    out = _fit(md, params)
    post = laplace_from_vem(out, [md])
    point = _vem_point(out.params)
    lp_mean = float(post.logpdf(point))
    assert np.isfinite(lp_mean)
    # Displace the well-identified sigma2_0 far in unconstrained space.
    displaced = point.copy()
    displaced[0] *= np.exp(1.0)
    assert post.logpdf(displaced) < lp_mean
    # Batched evaluation returns one logpdf per row.
    batch = post.logpdf(np.stack([point, displaced]))
    assert batch.shape == (2,)


# ---------------- non-PD / degenerate fallback (R3, scenario 5) ----------------


def test_block_cov_singular_beta_triggers_fallback():
    """A singular 2x2 precision yields a valid PD covariance and flags the fallback."""
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1, not PD
    cov, used_fb = _block_cov_from_precision(singular, 2.0 / 2.5**2)
    assert used_fb
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)


def test_block_cov_zero_beta_uses_prior_curvature():
    """An all-zero precision falls straight to the per-dimension prior floor."""
    cov, used_fb = _block_cov_from_precision(np.zeros((2, 2)), 2.0 / 2.5**2)
    assert used_fb
    # Per-dimension floor = Cauchy prior curvature 2/scale**2 -> variance scale**2/2.
    np.testing.assert_allclose(np.diag(cov), [2.5**2 / 2.0, 2.5**2 / 2.0])
    assert cov[0, 1] == 0.0


def test_estimate_betas_false_flags_beta_fallback():
    """Default fit (zero beta Fisher) sets the fallback flag on the beta block."""
    md, params = _make_synth(T=200)
    out = _fit(md, params, estimate_betas=False)
    assert np.allclose(out.beta_fisher_info, 0.0)
    post = laplace_from_vem(out, [md])
    assert post.curvature_fallback
    assert set(post.fallback_dims) >= {"beta_S", "beta_Z"}
    # Falls back to the Cauchy prior curvature -> beta variance = scale**2 / 2.
    np.testing.assert_allclose(
        np.diag(post.cov_u)[4:6], [2.5**2 / 2.0, 2.5**2 / 2.0]
    )


def test_singular_fisher_injected_gives_valid_distribution():
    """A hand-injected singular beta Fisher still yields a usable posterior."""
    md, params = _make_synth(T=200)
    out = _fit(md, params, estimate_betas=True)
    out_singular = replace(out, beta_fisher_info=np.array([[1.0, 1.0], [1.0, 1.0]]))
    post = laplace_from_vem(out_singular, [md])
    assert post.curvature_fallback
    assert np.all(np.linalg.eigvalsh(post.cov_u) > 0.0)
    draws = post.sample(np.random.default_rng(0), 100)
    assert np.all(np.isfinite(draws))
    assert np.isfinite(post.logpdf(_vem_point(out.params)))


# ---------------- beta curvature scales with data (scenario 6) ----------------


@pytest.mark.slow
def test_beta_block_variance_decreases_with_T():
    """The identified beta_S posterior variance shrinks as synthetic T grows."""
    md_small, p = _make_synth(T=500, beta_S=1.0, beta_Z=1.5, seed=7)
    md_large, _ = _make_synth(T=2000, beta_S=1.0, beta_Z=1.5, seed=7)

    out_small = _fit(md_small, p, n_iter=20, estimate_betas=True)
    out_large = _fit(md_large, p, n_iter=20, estimate_betas=True)
    post_small = laplace_from_vem(out_small, [md_small])
    post_large = laplace_from_vem(out_large, [md_large])

    # beta_S is the identifiable coefficient on this generator; its curvature
    # (Fisher info) scales with the number of trades, so its variance falls.
    assert post_large.cov_u[4, 4] < post_small.cov_u[4, 4]
    assert np.trace(out_large.beta_fisher_info) > np.trace(out_small.beta_fisher_info)


# ---------------- prior wiring ----------------


def test_prior_scale_changes_beta_fallback_variance():
    """Widening the Cauchy scale widens the fallback beta variance (prior sourced)."""
    md, params = _make_synth(T=150)
    out = _fit(md, params)  # zero Fisher -> prior-curvature fallback
    post_default = laplace_from_vem(out, [md], prior=PhiPrior())
    post_wide = laplace_from_vem(out, [md], prior=PhiPrior(beta_cauchy_scale=10.0))
    assert post_wide.cov_u[4, 4] > post_default.cov_u[4, 4]
    np.testing.assert_allclose(np.diag(post_wide.cov_u)[4:6], [50.0, 50.0])
