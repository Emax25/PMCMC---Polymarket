"""Tests for src.inference.online_scorer.

`OnlineScorer` is the streaming counterpart of batch VEM, so the suite is
anchored on two things: the frozen limit (`forgetting = 1.0`, no beta refresh)
must reproduce a bare `ADFFilter` bit-for-bit, and the adaptive limit must
actually track a parameter that moves. The remaining tests pin the online-only
edge cases — cold-start wallets, IRLS stability under separation, and the
`delta == 0` exclusion (ARCHITECTURE.md §6.1) that the decayed statistics would
otherwise carry forward forever.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from config.default_params import ModelParams, OnlineScorerConfig
from src.inference.adf_filter import ADFFilter
from src.inference.online_scorer import _MAX_BETA_WINDOW, OnlineScorer

# Centering/standardization constants. The scorer holds these fixed for a
# stream's lifetime, so the tests pass plain, already-representative values
# rather than refitting them per case.
M_S = 0.0
S_S = 0.5
M_Z = 0.0


def _params(**overrides) -> ModelParams:
    """Model parameters with all four variances set (never the NaN defaults)."""
    base = ModelParams(sigma2_0=0.01, sigma2_1=0.04, tau2_0=0.02, tau2_1=0.002)
    return replace(base, **overrides) if overrides else base


def _stream(
    n: int,
    *,
    sigma2: np.ndarray | float = 0.01,
    tau2: float = 1e-3,
    n_wallets: int = 5,
    delta: np.ndarray | float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a latent random walk observed with noise.

    Deliberately not `synthetic.generate_market`: these tests need to move a
    single generative quantity (the process variance) at a chosen trade and read
    the estimator's response, which the full generator does not expose.

    Args:
        n: Number of trades.
        sigma2: Per-trade process variance; a scalar or an (n,) schedule.
        tau2: Observation variance.
        n_wallets: Wallet ids are drawn uniformly from ``range(n_wallets)``.
        delta: Per-trade inter-trade time; a scalar or an (n,) array.
        seed: RNG seed.

    Returns:
        ``(Y, delta, log_size_ratio, wallet_ids)`` ready to feed to `step`.
    """
    rng = np.random.default_rng(seed)
    sigma2 = np.broadcast_to(np.asarray(sigma2, dtype=float), (n,))
    X = np.cumsum(rng.normal(0.0, np.sqrt(sigma2)))
    Y = X + rng.normal(0.0, np.sqrt(tau2), n)
    delta_arr = np.broadcast_to(np.asarray(delta, dtype=float), (n,)).copy()
    delta_arr[0] = 0.0
    log_size_ratio = rng.normal(0.0, S_S, n)
    wallet_ids = rng.integers(0, n_wallets, n)
    return Y, delta_arr, log_size_ratio, wallet_ids


def _run(scorer: OnlineScorer, stream) -> list:
    """Drive a scorer over a whole stream and collect the per-trade scores."""
    Y, delta, log_size_ratio, wallet_ids = stream
    return [
        scorer.step(Y[t], delta[t], log_size_ratio[t], wallet_ids[t])
        for t in range(len(Y))
    ]


# ---------------- Frozen limit: the regression anchor ----------------


def test_frozen_limit_matches_bare_adf_filter():
    """lambda = 1 with no beta refresh reproduces `ADFFilter` exactly."""
    params = _params(beta_S=0.7, beta_Z=1.3)
    theta_w = np.array([0.02, 0.05, 0.3, 0.5, 0.9])
    stream = _stream(300, n_wallets=len(theta_w), seed=1)

    scorer = OnlineScorer(
        params,
        theta_w,
        M_S,
        S_S,
        M_Z,
        config=OnlineScorerConfig(forgetting=1.0, n_refresh=None),
    )
    reference = ADFFilter(params, theta_w, M_S, S_S, M_Z)

    Y, delta, log_size_ratio, wallet_ids = stream
    for t in range(len(Y)):
        got = scorer.step(Y[t], delta[t], log_size_ratio[t], wallet_ids[t])
        want = reference.step(Y[t], delta[t], log_size_ratio[t], wallet_ids[t])
        assert got.Z_prob == want.Z_prob
        assert got.V_prob == want.V_prob
        assert got.X_mean == want.X_mean
        assert got.X_var == want.X_var
        assert got.log_evidence == want.log_evidence
        np.testing.assert_array_equal(got.q_vz, want.q_vz)

    # Nothing adapted: parameters and propensities are the ones handed in.
    assert scorer.params == params
    np.testing.assert_array_equal(scorer.theta_w[: len(theta_w)], theta_w)


def test_frozen_limit_holds_after_reset():
    """`reset` rewinds the carried state, not just the trade counter."""
    params = _params()
    theta_w = np.array([0.05, 0.2])
    stream = _stream(80, n_wallets=len(theta_w), seed=2)
    scorer = OnlineScorer(
        params,
        theta_w,
        M_S,
        S_S,
        M_Z,
        config=OnlineScorerConfig(forgetting=1.0, n_refresh=None),
    )

    first = [s.Z_prob for s in _run(scorer, stream)]
    scorer.reset()
    second = [s.Z_prob for s in _run(scorer, stream)]
    assert first == second
    assert scorer.t == len(stream[0])


# ---------------- Adaptation ----------------


def test_process_variance_tracks_a_mid_stream_regime_change():
    """A 16x jump in the true sigma2 moves the online estimate; lambda=1 pins it."""
    n = 1200
    # Variances well above the Inverse-Gamma(2, 1) prior's scale on purpose: an
    # exponential window of ~33 trades leaves the prior worth a few percent
    # here, whereas at sigma2 ~ 0.01 it would dominate the short-window
    # statistic and mask the very movement this test is checking for.
    sigma2_lo, sigma2_hi = 0.5, 8.0
    schedule = np.where(np.arange(n) < n // 2, sigma2_lo, sigma2_hi)
    stream = _stream(n, sigma2=schedule, tau2=0.01, seed=3)
    params = _params(sigma2_0=sigma2_lo, sigma2_1=sigma2_lo, tau2_0=0.01, tau2_1=1e-3)

    adaptive = OnlineScorer(
        params,
        np.full(5, 0.05),
        M_S,
        S_S,
        M_Z,
        config=OnlineScorerConfig(forgetting=0.97, n_refresh=None),
    )
    frozen = OnlineScorer(
        params,
        np.full(5, 0.05),
        M_S,
        S_S,
        M_Z,
        config=OnlineScorerConfig(forgetting=1.0, n_refresh=None),
    )

    Y, delta, log_size_ratio, wallet_ids = stream
    for t in range(n):
        adaptive.step(Y[t], delta[t], log_size_ratio[t], wallet_ids[t])
        frozen.step(Y[t], delta[t], log_size_ratio[t], wallet_ids[t])
        if t == n // 2 - 1:
            sigma2_at_switch = adaptive.params.sigma2_1

    # The order constraint pins sigma2_1 >= sigma2_0, so sigma2_1 is the
    # estimator's high-volatility read; it is the quantity the jump should move.
    assert sigma2_at_switch < 4.0 * sigma2_lo, "pre-switch estimate already inflated"
    assert adaptive.params.sigma2_1 > 3.0 * sigma2_at_switch
    assert frozen.params.sigma2_1 == params.sigma2_1


# ---------------- Cold start and theta_w ----------------


def test_unseen_wallet_cold_starts_at_the_prior_mean_then_rises():
    """A wallet the batch fit never saw scores at a/(a+b) and then learns."""
    params = _params(beta_S=3.0)
    theta_w = np.array([0.05, 0.05, 0.05])
    prior_mean = params.a / (params.a + params.b)
    scorer = OnlineScorer(
        params,
        theta_w,
        M_S,
        S_S,
        M_Z,
        config=OnlineScorerConfig(forgetting=0.98, n_refresh=None),
    )

    # Wallet 7 is beyond the supplied theta_w; every one of its trades is large
    # (a strongly positive standardized size covariate), so with beta_S = 3 its
    # q(Z) prior is near 1 and the Beta counts must accumulate insider mass.
    cold = scorer.step(0.0, 0.0, 4.0 * S_S, 7)
    assert cold.theta_w == pytest.approx(prior_mean)
    assert scorer.theta_w.size > 7

    for t in range(1, 60):
        scorer.step(0.01 * t, 1.0, 4.0 * S_S, 7)

    assert scorer.theta_w[7] > prior_mean
    # Untouched wallets keep the fit they came in with.
    np.testing.assert_allclose(scorer.theta_w[:3], theta_w)


def test_theta_w_is_the_beta_posterior_mean_of_the_decayed_counts():
    """Perfect insider evidence drives theta_w toward (a + n)/(a + b + n)."""
    params = _params(beta_S=6.0)
    scorer = OnlineScorer(
        params,
        np.empty(0),
        M_S,
        S_S,
        M_Z,
        config=OnlineScorerConfig(forgetting=0.995, n_refresh=None),
    )
    for t in range(200):
        scorer.step(0.001 * t, 1.0, 6.0 * S_S, 0)

    # q(Z) ~ 1 for every trade, so s_w ~ n_w and the posterior mean must sit
    # well above the 5% prior while staying a probability.
    assert 0.5 < scorer.theta_w[0] < 1.0


# ---------------- Beta refresh ----------------


def test_beta_refresh_is_smooth_and_bounded_under_separation():
    """Repeated decayed IRLS refreshes stay finite and move in small steps."""
    n = 900
    rng = np.random.default_rng(11)
    # Separation stress: insider trades are large *and* score q(Z) ~ 1, normal
    # trades are small and score q(Z) ~ 0, so the covariate perfectly orders the
    # target and the unpenalized MLE is at infinity.
    is_insider = rng.random(n) < 0.3
    log_size_ratio = np.where(is_insider, 5.0 * S_S, -5.0 * S_S)
    Y = np.cumsum(rng.normal(0.0, 0.1, n))
    delta = np.ones(n)
    delta[0] = 0.0
    wallet_ids = np.where(is_insider, 1, 0)

    # forgetting = 1.0 freezes the variance/transition blocks and theta_w, so
    # only the beta block moves and the assertions are unambiguous.
    scorer = OnlineScorer(
        _params(beta_S=4.0, beta_Z=0.0),
        np.array([0.02, 0.4]),
        M_S,
        S_S,
        M_Z,
        config=OnlineScorerConfig(forgetting=1.0, n_refresh=50, beta_window=100),
    )

    trace = []
    for t in range(n):
        scorer.step(Y[t], delta[t], log_size_ratio[t], wallet_ids[t])
        if (t + 1) % 50 == 0:
            trace.append((scorer.params.beta_S, scorer.params.beta_Z))

    beta = np.asarray(trace)
    assert np.isfinite(beta).all(), "IRLS diverged under separation"
    assert np.abs(beta).max() < 50.0, "Cauchy prior failed to bound the estimate"
    # Warm-started refreshes on overlapping windows: no refresh may jump.
    assert np.abs(np.diff(beta, axis=0)).max() < 10.0
    # And the trace must settle rather than drift off.
    assert np.abs(np.diff(beta[-4:], axis=0)).max() < 1.0


@pytest.mark.parametrize("n_refresh", [None, 0, -1])
def test_no_beta_refresh_leaves_the_coefficients_untouched(n_refresh):
    """Every documented never-refresh spelling holds beta_S/beta_Z fixed.

    `None` and any non-positive count all mean "never", matching
    `variational_em`'s ``estimate_betas=False``; 0 and -1 additionally have to
    not reach the ``self._t % n_refresh`` schedule check, which would be a
    ZeroDivisionError and a refresh on every trade respectively.
    """
    params = _params(beta_S=0.7, beta_Z=1.3)
    scorer = OnlineScorer(
        params,
        np.full(4, 0.05),
        M_S,
        S_S,
        M_Z,
        config=OnlineScorerConfig(forgetting=0.95, n_refresh=n_refresh),
    )
    _run(scorer, _stream(300, n_wallets=4, seed=5))
    assert scorer.params.beta_S == params.beta_S
    assert scorer.params.beta_Z == params.beta_Z


def test_beta_window_is_capped_when_it_follows_the_effective_window():
    """`forgetting = 1.0` must not size the refresh buffer at the 1e6 seed cap.

    `effective_window` doubles as the seed weight and is capped at 1e6 there, so
    a `beta_window = None` refresh config at ``forgetting = 1.0`` would allocate
    a million-slot deque for a window no one asked for. Non-degenerate settings
    are far below the cap and must be untouched by it.
    """

    def _maxlen(**cfg) -> int:
        scorer = OnlineScorer(
            _params(),
            np.full(2, 0.05),
            M_S,
            S_S,
            M_Z,
            config=OnlineScorerConfig(n_refresh=10, **cfg),
        )
        return scorer._buffer.maxlen

    assert _maxlen(forgetting=1.0) == _MAX_BETA_WINDOW + 1
    assert _maxlen(forgetting=0.98) == 51
    assert _maxlen(forgetting=0.999) == 1001


# ---------------- delta == 0 ----------------


def test_zero_delta_trades_do_not_corrupt_the_variance_statistics():
    """Same-second trades are excluded from SS_v, so no inf/NaN can accumulate."""
    n = 400
    Y, delta, log_size_ratio, wallet_ids = _stream(n, n_wallets=4, seed=7)
    # Every third trade shares a timestamp with its predecessor — the real-data
    # pattern from ARCHITECTURE.md §6.1.
    delta[::3] = 0.0

    scorer = OnlineScorer(
        _params(),
        np.full(4, 0.05),
        M_S,
        S_S,
        M_Z,
        config=OnlineScorerConfig(forgetting=0.95, n_refresh=None),
    )
    scores = _run(scorer, (Y, delta, log_size_ratio, wallet_ids))

    fitted = scorer.params
    for value in (
        fitted.sigma2_0,
        fitted.sigma2_1,
        fitted.tau2_0,
        fitted.tau2_1,
        fitted.q_01,
        fitted.q_10,
    ):
        assert np.isfinite(value)
    assert fitted.sigma2_0 > 0.0 and fitted.tau2_1 > 0.0
    assert np.isfinite([s.Z_prob for s in scores]).all()
    assert np.isfinite(scorer.theta_w).all()


def test_all_zero_delta_stream_stays_finite():
    """A stream with no positive gaps starves SS_v without ever dividing by zero."""
    n = 120
    Y, delta, log_size_ratio, wallet_ids = _stream(n, n_wallets=3, seed=9)
    delta[:] = 0.0

    scorer = OnlineScorer(
        _params(),
        np.full(3, 0.05),
        M_S,
        S_S,
        M_Z,
        config=OnlineScorerConfig(forgetting=0.9, n_refresh=None),
    )
    _run(scorer, (Y, delta, log_size_ratio, wallet_ids))
    assert np.isfinite(scorer.params.sigma2_0)
    assert np.isfinite(scorer.params.sigma2_1)
    assert scorer.params.sigma2_1 >= scorer.params.sigma2_0


# ---------------- Config validation ----------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forgetting": 0.0},
        {"forgetting": 1.5},
        {"rho_schedule": "adam"},
        {"rho_alpha": 0.5},
        {"rho_alpha": 1.5},
        {"rho_t0": 1.0},
        {"rho_t0": 0.5},
        {"beta_window": 1},
    ],
)
def test_config_rejects_invalid_schedules(kwargs):
    """Schedules outside the documented ranges raise rather than silently misfit."""
    with pytest.raises(ValueError):
        OnlineScorerConfig(**kwargs)


def test_robbins_monro_rate_decays_and_fixed_rate_does_not():
    """The two schedules differ exactly where the docstring says they do."""
    fixed = OnlineScorerConfig(forgetting=0.98)
    rm = OnlineScorerConfig(rho_schedule="robbins_monro", rho_alpha=0.6)
    assert fixed.rho(0) == pytest.approx(0.02)
    assert fixed.rho(10_000) == pytest.approx(0.02)
    assert rm.rho(0) == pytest.approx(50.0**-0.6)
    assert rm.rho(10_000) < rm.rho(10)
    assert OnlineScorerConfig(forgetting=1.0).rho(5) == 0.0
    # rho_0 < 1 is the load-bearing part: a rate of exactly 1 gives a decay
    # factor of 0, which would discard the seeded statistics unread.
    assert rm.rho(0) < 1.0


def test_robbins_monro_first_trade_keeps_the_seeded_params():
    """The Robbins-Monro rate must not erase the seed on trade 0.

    With ``rho_0 = 1`` the recursion's decay factor ``1 - rho_0`` is exactly 0,
    so the very first trade throws away the statistics seeded from the incoming
    fit and the parameters jump to the prior — ``q_01`` to the Beta(1, 1) mean
    0.5 whatever it was fitted at. The `rho_t0` offset is what keeps trade 1
    starting *at* the handed-over fit; only the long-run limit was pinned
    before, which this failure mode slips straight through.

    Variances are chosen above the Inverse-Gamma prior's implied floor so the
    seed inversion is exact rather than clamped (see `_seed_stats`), making the
    comparison a tight one.
    """
    # q_01 / q_10 well away from the Beta(1, 1) prior mean 0.5, which is where
    # a wiped transition statistic lands.
    seeded = _params(
        sigma2_0=0.1, sigma2_1=0.2, tau2_0=0.02, tau2_1=0.002, q_01=0.05, q_10=0.1
    )
    scorer = OnlineScorer(
        seeded,
        np.full(4, 0.05),
        M_S,
        S_S,
        M_Z,
        config=OnlineScorerConfig(rho_schedule="robbins_monro", rho_alpha=0.6),
    )
    scorer.step(0.1, 0.0, 0.2, 1)

    fitted = scorer.params
    for name in ("sigma2_0", "sigma2_1", "tau2_0", "tau2_1", "q_01", "q_10"):
        assert getattr(fitted, name) == pytest.approx(
            getattr(seeded, name), rel=0.15
        ), f"{name} moved off the seed on the first trade"


def test_robbins_monro_converges_toward_the_batch_variance():
    """On a stationary stream the decreasing rate settles near the truth."""
    sigma2_true = 0.05
    stream = _stream(2000, sigma2=sigma2_true, tau2=1e-5, seed=13)
    scorer = OnlineScorer(
        _params(sigma2_0=0.2, sigma2_1=0.2, tau2_0=1e-5, tau2_1=1e-6),
        np.full(5, 0.05),
        M_S,
        S_S,
        M_Z,
        config=OnlineScorerConfig(rho_schedule="robbins_monro", rho_alpha=0.6),
    )
    _run(scorer, stream)

    # A single-variance stream identifies only the regime the fitted V chain
    # actually occupies: the other accumulates almost no q(V) mass, so its
    # variance keeps whatever the seed said. Read the occupied regime off the
    # fitted transition probabilities rather than assuming which one it is.
    fitted = scorer.params
    rho_V = fitted.q_01 / (fitted.q_01 + fitted.q_10)
    occupied = fitted.sigma2_1 if rho_V > 0.5 else fitted.sigma2_0
    # Filtered (not smoothed) increments bias sigma2 low, so this is a
    # containment band, not a point estimate check.
    assert 0.2 * sigma2_true < occupied < 3.0 * sigma2_true
