"""Tests for src.analysis.validation (held-out predictive LL and PSIS-khat)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pytest
from scipy.special import digamma, polygamma
from scipy.stats import genpareto, norm

from config.default_params import ModelParams, PhiPrior
from src.analysis.validation import (
    HeldoutLL,
    PSIS_KHAT_KEY,
    _psislw,
    heldout_predictive_ll,
    heldout_predictive_summary,
    holdout_split,
    khat_interpretation,
    psis_khat,
)
from src.data.synthetic import generate_market
from src.inference.laplace import PhiPosterior, laplace_from_vem
from src.inference.particle_gibbs import MarketData
from src.inference.variational_em import VEMOutput


def _linear_market(T: int) -> MarketData:
    """A small market whose delta array makes the boundary gap easy to check."""
    return MarketData(
        Y=np.arange(T, dtype=float),
        delta=np.arange(T, dtype=float),  # delta[i] == i; delta[0] == 0 sentinel
        log_size_ratio=np.linspace(-1.0, 1.0, T),
        wallet_ids=np.arange(T, dtype=int) % 4,
    )


def _make_vem_output(
    params: ModelParams,
    theta_w: np.ndarray,
    m_S: float,
    s_S: float,
    m_Z: float = 0.0,
) -> VEMOutput:
    """Minimal VEMOutput carrying only the fields the scorer reads.

    `heldout_predictive_ll` consumes exactly `params`, `theta_w`, and the
    standardization constants `(m_S, s_S, m_Z)`; the remaining VEMOutput fields
    are filled with inert placeholders so a fitted run is not required to test
    the scoring math.
    """
    n_wallets = len(theta_w)
    return VEMOutput(
        params=params,
        theta_w=np.asarray(theta_w, dtype=float),
        Z_prob=[],
        V_prob=[],
        X_mean=[],
        elbo_trace=np.empty(0),
        n_iter_run=0,
        m_S=m_S,
        s_S=s_S,
        m_Z=m_Z,
        theta_w_logit_mean=np.zeros(n_wallets),
        theta_w_logit_var=np.zeros(n_wallets),
        beta_S_orig=0.0,
        beta_Z_orig=0.0,
        beta_fisher_info=np.zeros((2, 2)),
    )


def _true_params() -> ModelParams:
    """Explicit generating parameters (betas 0 -> covariates inert)."""
    return ModelParams(
        sigma2_0=0.01,
        sigma2_1=0.1,
        tau2_0=0.05,
        tau2_1=0.01,
    )


# ---------------- Split integrity (scenario 1) ----------------


def test_holdout_split_partitions_and_orders():
    """Head + tail partition the market, preserve order, and reconstruct it."""
    md = _linear_market(10)
    heads, tails = holdout_split([md], h=0.2)
    head, tail = heads[0], tails[0]

    # h * T = 2 -> 8 training, 2 held out.
    assert len(head.Y) == 8
    assert len(tail.Y) == 2

    # Concatenation reconstructs the original arrays exactly.
    np.testing.assert_array_equal(np.concatenate([head.Y, tail.Y]), md.Y)
    np.testing.assert_array_equal(
        np.concatenate([head.delta, tail.delta]), md.delta
    )
    np.testing.assert_array_equal(
        np.concatenate([head.wallet_ids, tail.wallet_ids]), md.wallet_ids
    )

    # Head keeps the fresh-market sentinel; cumulative time is non-decreasing.
    assert head.delta[0] == 0.0
    assert np.all(np.diff(np.cumsum(md.delta)) >= 0.0)


def test_holdout_split_boundary_delta_is_true_gap():
    """Tail's first delta equals the true inter-trade gap (KTD5), not 0."""
    md = _linear_market(10)
    _, tails = holdout_split([md], h=0.2)
    tail = tails[0]
    # n_head == 8, so the boundary gap is the original delta[8] == 8.0 (the
    # time from the last training trade to the first held-out trade), NOT the
    # delta[0] == 0 sentinel that a standalone market would carry.
    assert tail.delta[0] == md.delta[8]
    assert tail.delta[0] != 0.0


def test_holdout_split_h_zero_empty_tail():
    """h = 0 yields an empty tail handled gracefully by the scorer."""
    md = _linear_market(10)
    heads, tails = holdout_split([md], h=0.0)
    assert len(heads[0].Y) == 10
    assert len(tails[0].Y) == 0

    params = _true_params()
    vo = _make_vem_output(params, theta_w=np.full(4, 0.1), m_S=0.0, s_S=1.0)
    result = heldout_predictive_ll(vo, heads[0], tails[0])
    assert result.n_tail == 0
    assert result.total == 0.0
    assert math.isnan(result.mean)


# ---------------- Better model wins (scenario 2) ----------------


def _market_data(mkt) -> MarketData:
    """Adapt a SyntheticMarket to MarketData (log_size_ratio from S / S_bar)."""
    return MarketData(
        Y=mkt.Y,
        delta=mkt.delta,
        log_size_ratio=np.log(mkt.S / mkt.S_bar),
        wallet_ids=mkt.wallet_ids,
    )


def _score_market(mkt, params: ModelParams, h: float = 0.2) -> float:
    """Held-out tail log-likelihood for one market under `params`."""
    md = _market_data(mkt)
    m_S = float(np.mean(md.log_size_ratio))
    s_S = float(np.std(md.log_size_ratio))
    vo = _make_vem_output(params, mkt.theta_w, m_S, s_S)
    heads, tails = holdout_split([md], h=h)
    return heldout_predictive_ll(vo, heads[0], tails[0]).total


def test_generating_params_beat_corrupted():
    """Generating phi scores higher held-out LL than a tau2-doubled phi."""
    true_params = _true_params()
    mkt = generate_market(
        true_params,
        n_trades=500,
        n_wallets=20,
        n_insider_wallets=3,
        mean_inter_trade_time=1.0,
        rng=np.random.default_rng(11),
    )
    corrupt_params = replace(
        true_params,
        tau2_0=2.0 * true_params.tau2_0,
        tau2_1=2.0 * true_params.tau2_1,
    )
    ll_true = _score_market(mkt, true_params)
    ll_corrupt = _score_market(mkt, corrupt_params)
    assert ll_true > ll_corrupt


# ---------------- Determinism (scenario 3) ----------------


def test_scoring_is_deterministic():
    """Same seed/inputs -> identical held-out LL, bit for bit."""
    true_params = _true_params()
    mkt = generate_market(
        true_params,
        n_trades=200,
        n_wallets=12,
        n_insider_wallets=2,
        mean_inter_trade_time=1.0,
        rng=np.random.default_rng(3),
    )
    ll_a = _score_market(mkt, true_params)
    ll_b = _score_market(mkt, true_params)
    assert ll_a == ll_b


# ---------------- Pooled summary (R4) ----------------


def test_pooled_summary_aggregates_per_market():
    """Pooled total/count equal the summed per-market totals/counts."""
    true_params = _true_params()
    markets = [
        _market_data(
            generate_market(
                true_params,
                n_trades=120,
                n_wallets=10,
                n_insider_wallets=2,
                mean_inter_trade_time=1.0,
                rng=np.random.default_rng(seed),
            )
        )
        for seed in (1, 2, 3)
    ]
    # theta_w must span all wallets across markets; all use n_wallets == 10.
    vo = _make_vem_output(true_params, np.full(10, 0.1), m_S=0.0, s_S=1.0)
    heads, tails = holdout_split(markets, h=0.2)
    summary = heldout_predictive_summary(vo, heads, tails)

    assert len(summary.per_market) == 3
    assert summary.pooled_n == sum(m.n_tail for m in summary.per_market)
    assert summary.pooled_total == float(
        sum(m.total for m in summary.per_market)
    )
    assert summary.pooled_mean == summary.pooled_total / summary.pooled_n
    assert all(isinstance(m, HeldoutLL) for m in summary.per_market)


# ---------------- PSIS-khat (R5) ----------------

# A well-conditioned reference point on the unconstrained scale: log-variances,
# logit-transitions, and betas whose back-transform is a plausible fitted phi.
_REF_MEAN_U = np.array([-2.0, -1.0, -3.0, 0.0, 0.3, -0.2, -2.5, -3.0])
_REF_VAR_U = np.array([0.20, 0.25, 0.30, 0.25, 0.10, 0.15, 0.20, 0.25])

_VAR_IDX = [0, 1, 6, 7]  # log-transformed dims of phi
_Q_IDX = [2, 3]          # logit-transformed dims of phi


def _log_jacobian(phi: np.ndarray) -> np.ndarray:
    """log|du/dphi| for the unconstrained reparameterization, written out here.

    Deliberately an *independent* reimplementation of the transform Jacobian
    (`d log(x)/dx = 1/x`; `d logit(q)/dq = 1/(q(1-q))`) so the tests below do not
    inherit a bug from the production helper they are meant to police.
    """
    phi = np.atleast_2d(np.asarray(phi, dtype=float))
    q = phi[:, _Q_IDX]
    return (-np.log(phi[:, _VAR_IDX])).sum(axis=1) + (
        -np.log(q) - np.log1p(-q)
    ).sum(axis=1)


def _explicit_constrained_logpdf(
    phi: np.ndarray, mean_u: np.ndarray, var_u: np.ndarray
) -> np.ndarray:
    """Independent constrained-scale density of `u = g(phi) ~ N(mean_u, var_u)`.

    Equals a per-dimension product of lognormal (variances), logit-normal
    (transitions), and normal (betas) densities, assembled as "Gaussian on u
    plus the log-Jacobian". Written without touching `PhiPosterior` so it can be
    used as an oracle for it.
    """
    phi = np.atleast_2d(np.asarray(phi, dtype=float))
    u = phi.copy()
    u[:, _VAR_IDX] = np.log(phi[:, _VAR_IDX])
    u[:, _Q_IDX] = np.log(phi[:, _Q_IDX]) - np.log1p(-phi[:, _Q_IDX])
    gauss = -0.5 * (
        np.log(2.0 * np.pi * var_u) + (u - mean_u) ** 2 / var_u
    ).sum(axis=1)
    return gauss + _log_jacobian(phi)


def _log_beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Log Beta(a, b) density (constrained scale), written out for the oracle."""
    log_norm = (
        math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    )
    return (a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x) - log_norm


@dataclass
class _AnalyticPrior(PhiPrior):
    """Test target with an analytically known shape, in place of `PhiPrior`.

    `psis_khat` consumes the prior solely through `log_prior`, so overriding it
    swaps the PSIS *target* (with no markets, target = prior) for one a Gaussian
    proposal can actually cover. The production `PhiPrior` is unusable as a
    likelihood-free target: its Cauchy betas and InvGamma variances are heavier-
    tailed in every dimension than any Gaussian, so the weights are unbounded by
    construction and khat pins near 1 no matter how good the proposal is. That
    pathology is an artifact of dropping the likelihood — in the real diagnostic
    the target is `p(Y | phi) p(phi)`, which the data concentrates.

    Attributes:
        mean_u: Target mean on the unconstrained scale.
        var_u: Target per-dimension variance on the unconstrained scale.
        beta_q01: Optional `(a, b)` replacing the `q_01` factor with a Beta
            density on the *constrained* scale — the Jacobian probe.
    """

    mean_u: np.ndarray = field(default_factory=lambda: _REF_MEAN_U.copy())
    var_u: np.ndarray = field(default_factory=lambda: _REF_VAR_U.copy())
    beta_q01: tuple[float, float] | None = None

    def log_prior(self, phi: np.ndarray) -> np.ndarray:
        """Log target density at constrained `phi`."""
        phi = np.atleast_2d(np.asarray(phi, dtype=float))
        log_p = _explicit_constrained_logpdf(phi, self.mean_u, self.var_u)
        if self.beta_q01 is None:
            return log_p
        # Swap the q_01 factor: drop the logit-normal one, add Beta(a, b). Both
        # are constrained-scale densities, so only that dimension's ratio
        # survives in the importance weights.
        q = phi[:, 2]
        u = np.log(q) - np.log1p(-q)
        log_q01_normal = -0.5 * (
            np.log(2.0 * np.pi * self.var_u[2])
            + (u - self.mean_u[2]) ** 2 / self.var_u[2]
        ) + (-np.log(q) - np.log1p(-q))
        return log_p - log_q01_normal + _log_beta_pdf(q, *self.beta_q01)


def _posterior(mean_u: np.ndarray, var_u: np.ndarray) -> PhiPosterior:
    """Diagonal `PhiPosterior` from unconstrained means and variances."""
    return PhiPosterior(mean_u=np.asarray(mean_u), cov_u=np.diag(np.asarray(var_u)))


def _inert_vem_output() -> VEMOutput:
    """VEMOutput whose params each draw overwrites; only `params` is read."""
    return _make_vem_output(_true_params(), np.full(4, 0.1), m_S=0.0, s_S=1.0)


def test_psis_khat_good_when_proposal_covers_target():
    """Scenario 1: a proposal that covers the target scores khat < 0.5."""
    prior = _AnalyticPrior()
    # Same centre, 1.5x the target's sd: importance weights are then bounded
    # (the exponent's quadratic coefficient stays negative), the textbook
    # "proposal at least as diffuse as the target" safe case.
    q_post = _posterior(_REF_MEAN_U, _REF_VAR_U * 1.5**2)
    result = psis_khat(
        _inert_vem_output(),
        q_post,
        markets=[],  # no likelihood -> the PSIS target is exactly the prior
        rng=np.random.default_rng(0),
        n_draws=2000,
        prior=prior,
    )
    assert result.n_draws == 2000
    assert result.khat < 0.5
    assert np.all(np.isfinite(result.log_weights))
    assert result.interpretation.startswith("good")
    # Reported under the R5 name, with the scope-of-claim caveat attached.
    assert result.to_dict()[PSIS_KHAT_KEY] == result.khat
    assert "necessary-not-sufficient" in result.to_dict()["psis_scope_note"]


def test_psis_khat_degrades_under_mismatch():
    """Scenario 2: a shifted, over-concentrated proposal degrades khat."""
    prior = _AnalyticPrior()
    good = _posterior(_REF_MEAN_U, _REF_VAR_U * 1.5**2)
    # Shift by 2 target sds and shrink to a third of the target's sd. Shrinking
    # (not inflating) is the degrading direction: an *over-dispersed* proposal
    # has bounded weights and stays PSIS-safe, whereas a proposal narrower than
    # the target leaves the target's tails uncovered, which is what khat detects.
    sd = np.sqrt(_REF_VAR_U)
    bad = _posterior(_REF_MEAN_U + 2.0 * sd, _REF_VAR_U / 9.0)

    def _khat(phi_posterior: PhiPosterior) -> float:
        return psis_khat(
            _inert_vem_output(),
            phi_posterior,
            markets=[],
            rng=np.random.default_rng(1),
            n_draws=2000,
            prior=prior,
        ).khat

    khat_good, khat_bad = _khat(good), _khat(bad)
    # Direction only — the 0.7 rule is a reporting/stop-condition matter (R5),
    # deliberately not a CI threshold.
    assert khat_bad > khat_good
    assert khat_bad > 0.5


def test_psis_weights_compare_densities_on_one_scale():
    """Scenario 4: prior and q are compared on the same (constrained) scale.

    `PhiPrior.log_prior` is a constrained-scale density and `PhiPosterior.logpdf`
    adds the change-of-variables Jacobian so that it is one too. If that Jacobian
    went missing, every log-weight would pick up `log|du/dphi|` and the sampler
    would silently target a different distribution.

    The probe makes that detectable analytically: the target matches the proposal
    in all seven other dimensions, so every factor cancels and the log-weights
    must reduce *exactly* to the 1-D ratio `log Beta(3, 7)(q_01) - log
    logit-normal(q_01)`. A dropped Jacobian would offset them by `log|du/dphi|`.
    The same offset would also tilt the implied target by `1/(q(1-q))`, turning
    Beta(3, 7) into Beta(2, 6) and its mean from `3/10` into `2/8` — the second
    assertion pins the correct one.
    """
    a, b = 3.0, 7.0
    mean_u = _REF_MEAN_U.copy()
    var_u = _REF_VAR_U.copy()
    # Centre the q_01 proposal on the Beta target's logit-scale moments so the
    # 1-D importance sampler is well behaved.
    mean_u[2] = digamma(a) - digamma(b)
    var_u[2] = polygamma(1, a) + polygamma(1, b)
    prior = _AnalyticPrior(mean_u=mean_u, var_u=var_u, beta_q01=(a, b))
    q_post = _posterior(mean_u, var_u)

    n_draws, seed = 20000, 7
    result = psis_khat(
        _inert_vem_output(),
        q_post,
        markets=[],
        rng=np.random.default_rng(seed),
        n_draws=n_draws,
        prior=prior,
    )
    # Same seed and draw count -> exactly the draws the estimator scored.
    draws = q_post.sample(np.random.default_rng(seed), n_draws)
    q01 = draws[:, 2]
    u01 = np.log(q01) - np.log1p(-q01)
    log_q01 = -0.5 * (
        np.log(2.0 * np.pi * var_u[2]) + (u01 - mean_u[2]) ** 2 / var_u[2]
    ) + (-np.log(q01) - np.log1p(-q01))
    expected = _log_beta_pdf(q01, a, b) - log_q01
    np.testing.assert_allclose(result.log_weights, expected, atol=1e-9)

    # Self-normalized importance estimate of E[q_01] recovers the Beta(3, 7)
    # mean 0.3; the Jacobian-free weights would land on Beta(2, 6)'s 0.25.
    w = np.exp(result.log_weights - result.log_weights.max())
    assert abs(float(np.sum(w * q01) / np.sum(w)) - a / (a + b)) < 0.01
    assert abs(float(np.mean(_log_jacobian(draws))) - 0.0) > 1.0  # probe is live


def _psis_market_fixture() -> tuple[VEMOutput, list[MarketData], PhiPosterior]:
    """One small synthetic market plus its Laplace posterior, for the ADF path."""
    params = _true_params()
    mkt = generate_market(
        params,
        n_trades=150,
        n_wallets=10,
        n_insider_wallets=2,
        mean_inter_trade_time=1.0,
        rng=np.random.default_rng(5),
    )
    md = _market_data(mkt)
    vo = _make_vem_output(
        params,
        mkt.theta_w,
        m_S=float(np.mean(md.log_size_ratio)),
        s_S=float(np.std(md.log_size_ratio)),
    )
    return vo, [md], laplace_from_vem(vo, [md])


def test_psis_log_weights_finite_on_market_fixture():
    """Scenario 3: no NaN/inf log-weights when the ADF pass is in the loop."""
    vo, markets, phi_posterior = _psis_market_fixture()
    result = psis_khat(
        vo,
        phi_posterior,
        markets,
        rng=np.random.default_rng(2),
        n_draws=40,
    )
    assert result.log_weights.shape == (40,)
    assert np.all(np.isfinite(result.log_weights))
    assert np.all(np.isfinite(result.log_weights_smoothed))
    assert math.isfinite(result.khat)


def test_psis_khat_parallel_matches_sequential():
    """Scenario 5: n_jobs > 1 reproduces the sequential log-weights exactly."""
    vo, markets, phi_posterior = _psis_market_fixture()
    seq = psis_khat(
        vo, phi_posterior, markets, rng=np.random.default_rng(4), n_draws=30, n_jobs=1
    )
    par = psis_khat(
        vo, phi_posterior, markets, rng=np.random.default_rng(4), n_draws=30, n_jobs=2
    )
    np.testing.assert_array_equal(seq.log_weights, par.log_weights)
    assert seq.khat == par.khat


def test_psis_khat_rejects_too_few_draws():
    """A draw count below the PSIS tail-fit minimum fails loudly, not obscurely."""
    with pytest.raises(ValueError, match="n_draws"):
        psis_khat(
            _inert_vem_output(),
            _posterior(_REF_MEAN_U, _REF_VAR_U),
            markets=[],
            rng=np.random.default_rng(0),
            n_draws=5,
            prior=_AnalyticPrior(),
        )


def test_khat_interpretation_bands():
    """The reported string follows the standard <0.5 / 0.5-0.7 / >0.7 bands."""
    assert khat_interpretation(0.2).startswith("good")
    assert khat_interpretation(0.6).startswith("ok")
    assert khat_interpretation(0.9).startswith("bad")
    assert khat_interpretation(float("nan")).startswith("undefined")


def test_psislw_recovers_known_pareto_shape():
    """The PSIS tail fit is oriented correctly against an exact-khat reference.

    Regression guard for the arviz-stats 1.1.0 wrapper bugs documented in
    `validation._psislw`: both of that package's public entry points fit the
    wrong tail (or the wrong scale) and return khat near 0 for these samples, so
    an arviz upgrade could silently invert the diagnostic. Exact generalized-
    Pareto weights have khat equal to their shape parameter by construction.
    """
    for shape in (0.3, 0.7, 1.0):
        log_weights = np.log(
            genpareto.rvs(shape, size=20000, random_state=1) + 1.0
        )
        _, khat = _psislw(log_weights)
        assert abs(khat - shape) < 0.1

    # An over-dispersed proposal has bounded weights: khat must come out low.
    x = np.random.default_rng(0).normal(0.0, 2.0, 20000)
    safe = norm.logpdf(x, 0.0, 1.0) - norm.logpdf(x, 0.0, 2.0)
    assert _psislw(safe)[1] < 0.5
