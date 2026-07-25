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
    AUC_SPREAD_THRESHOLD,
    HeldoutLL,
    INIT_JITTER_LOG_SD,
    PSIS_KHAT_KEY,
    PSIS_SCOPE_NOTE,
    _psislw,
    convergence_block,
    elbo_convergence,
    heldout_predictive_ll,
    heldout_predictive_summary,
    holdout_split,
    jittered_init,
    khat_interpretation,
    mean_pairwise_jaccard,
    phi_centring_gradient,
    pooled_synthetic_auc,
    psis_khat,
    restart_record,
    spread,
    stability_block,
    top_k_wallets,
)
from src.data.preprocess import WalletIndex
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


def test_khat_bad_band_does_not_prescribe_enriching_the_family():
    """A bad khat is reported as a proposal fault, not a family-richness fault.

    Enriching the variational family is a refuted remedy here: a richer family
    centred at the same non-mode with the same expected-complete-data curvature
    scores the same khat. The bad-band text must send the reader to the
    centring diagnostic instead of prescribing that remedy.
    """
    bad = khat_interpretation(0.9)
    assert "centred at" in bad
    assert "phi_centring_gradient" in bad
    # The refuted prescription ("enrich the variational family") is gone.
    assert "enrich" not in bad.lower()


def test_psis_scope_note_states_the_theta_w_conditioning():
    """The stored scope string names the target as conditional on theta_w_hat.

    `theta_w` is pinned at its fitted value for every draw, so the PSIS target
    is `p(phi | Y, theta_w_hat)` and not a marginal over wallet propensities.
    The JSON key stays `psis_khat_laplace_vs_adf` for comparability with the
    committed artifacts, which makes the scope string the only place that
    limitation can be recorded.
    """
    assert "theta_w_hat" in PSIS_SCOPE_NOTE
    assert "CONDITIONAL" in PSIS_SCOPE_NOTE
    assert "necessary-not-sufficient" in PSIS_SCOPE_NOTE
    assert PSIS_KHAT_KEY == "psis_khat_laplace_vs_adf"


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


# ---------------- Multi-restart stability metrics (R6, KTD4) ----------------


def test_spread_known_values():
    """min/max/mean/sd/spread match hand-computed values on a known list."""
    s = spread([1.0, 2.0, 3.0, 4.0])
    assert s["min"] == 1.0
    assert s["max"] == 4.0
    assert s["mean"] == 2.5
    assert s["spread"] == 3.0
    # Population sd (ddof=0) of [1,2,3,4] is sqrt(1.25).
    assert s["sd"] == pytest.approx(math.sqrt(1.25))


def test_spread_drops_none_and_survives_all_none():
    """None entries are ignored; an all-None metric yields NaNs, not an error."""
    s = spread([None, 2.0, None, 4.0])
    assert s["min"] == 2.0 and s["max"] == 4.0 and s["spread"] == 2.0

    empty = spread([None, None])
    assert all(math.isnan(v) for v in empty.values())


def test_mean_pairwise_jaccard_known_cases():
    """Identical / disjoint / partially overlapping / empty sets, by hand."""
    assert mean_pairwise_jaccard([[1, 2, 3], [3, 2, 1]]) == 1.0
    assert mean_pairwise_jaccard([[1, 2], [3, 4]]) == 0.0
    # |{1,2} & {2,3}| / |{1,2} u {2,3}| = 1/3.
    assert mean_pairwise_jaccard([[1, 2], [2, 3]]) == pytest.approx(1.0 / 3.0)
    # Two empty sets are identical, not a 0/0 division.
    assert mean_pairwise_jaccard([[], []]) == 1.0
    # Three sets -> mean over the three unordered pairs: (1 + 0 + 0) / 3.
    assert mean_pairwise_jaccard([[1], [1], [2]]) == pytest.approx(1.0 / 3.0)
    # Undefined with fewer than two sets.
    assert math.isnan(mean_pairwise_jaccard([[1, 2]]))
    assert math.isnan(mean_pairwise_jaccard([]))


def test_top_k_wallets_ties_and_clamping():
    """Ties resolve to the lower wallet id and k is clamped to a valid size."""
    # Wallets 1 and 3 tie at 0.5; the cutoff k=2 must take the lower id (1).
    scores = np.array([0.1, 0.5, 0.9, 0.5])
    assert top_k_wallets(scores, 2) == [1, 2]
    # All-equal scores: the set is the first k ids in ascending order.
    assert top_k_wallets(np.ones(5), 3) == [0, 1, 2]
    # k above/below the valid range clamps rather than raising.
    assert top_k_wallets(scores, 99) == [0, 1, 2, 3]
    assert top_k_wallets(scores, 0) == [2]
    # The result is a set (sorted ids), not a ranking.
    assert top_k_wallets(scores, 3) == [1, 2, 3]


def _wallet_index(n_wallets: int) -> WalletIndex:
    """Synthetic wallet index whose ids mirror 0..n_wallets-1."""
    idx = WalletIndex()
    for w in range(n_wallets):
        idx.add(f"synthetic-{w:04d}")
    return idx


def test_pooled_synthetic_auc_perfect_and_inverted():
    """Scoring the true Z gives AUC 1.0; scoring its complement gives 0.0."""
    params = _true_params()
    mkt = generate_market(
        params,
        n_trades=200,
        n_wallets=10,
        n_insider_wallets=2,
        mean_inter_trade_time=1.0,
        rng=np.random.default_rng(9),
    )
    md = _market_data(mkt)
    z_true = mkt.Z.astype(float)
    assert 0 < z_true.sum() < z_true.size  # both classes present, AUC defined
    vo = _make_vem_output(params, mkt.theta_w, m_S=0.0, s_S=1.0)
    idx = _wallet_index(10)

    perfect = replace(vo, Z_prob=[z_true])
    assert pooled_synthetic_auc(perfect, [md], [mkt], idx) == pytest.approx(1.0)
    inverted = replace(vo, Z_prob=[1.0 - z_true])
    assert pooled_synthetic_auc(inverted, [md], [mkt], idx) == pytest.approx(0.0)


def test_jittered_init_is_seeded_and_stays_in_support():
    """Restart starts are reproducible, positive, and ordered as warm-started."""
    md = _linear_market(200)
    a, _ = jittered_init([md], 6, np.random.default_rng(0))
    b, _ = jittered_init([md], 6, np.random.default_rng(0))
    c, theta_c = jittered_init([md], 6, np.random.default_rng(1))

    assert a == b  # same seed -> same start point
    assert a != c  # different seed -> different start point
    for p in (a, c):
        assert p.sigma2_0 > 0 and p.sigma2_1 > 0
        assert p.tau2_0 > 0 and p.tau2_1 > 0
        # A 0.1 log-sd jitter cannot swap the decade-apart regime variances.
        assert p.sigma2_0 < p.sigma2_1
    assert theta_c.shape == (6,)
    assert np.all((theta_c > 0.0) & (theta_c < 1.0))


# ---------------- Convergence status (H1) ----------------


def test_elbo_convergence_matches_the_fit_stopping_rule():
    """The recorded verdict reproduces variational_em's own relative test."""
    # |(-100.0) - (-100.01)| / max(100.01, 1) = 9.999e-5 < 1e-4 -> converged.
    conv = elbo_convergence(
        np.array([-101.0, -100.01, -100.0]), 3, n_iter_max=50, tol=1e-4
    )
    assert conv["converged"] is True
    assert conv["hit_iter_cap"] is False
    assert conv["final_rel_elbo_change"] == pytest.approx(0.01 / 100.01)
    assert conv["final_elbo_gain"] == pytest.approx(0.01)


def test_elbo_convergence_flags_iteration_cap():
    """A run stopped by the cap with a change above tol is not converged."""
    trace = np.array([-2730.0, -2728.0, -2726.66])
    conv = elbo_convergence(trace, 3, n_iter_max=3, tol=1e-4)
    assert conv["hit_iter_cap"] is True
    assert conv["converged"] is False
    assert conv["final_rel_elbo_change"] > 1e-4

    # Too short to measure anything: no verdict is invented.
    short = elbo_convergence(np.array([-1.0]), 1, n_iter_max=50, tol=1e-4)
    assert short["converged"] is False
    assert math.isnan(short["final_rel_elbo_change"])


def _cap_records(terminal_elbos: list[float], gain: float) -> list[dict]:
    """Restart records that all stopped at a 50-iteration cap, still climbing."""
    return [
        {
            "seed": 40 + i,
            "terminal_elbo": e,
            "n_iter_run": 50,
            "hit_iter_cap": True,
            "converged": False,
            "final_rel_elbo_change": abs(gain) / abs(e - gain),
            "final_elbo_gain": gain,
            "pooled_auc": None,
            "beta_S_orig": 0.0,
            "beta_Z_orig": 0.0,
            "top_k_wallets": [0, 1],
        }
        for i, e in enumerate(terminal_elbos)
    ]


def test_convergence_block_reports_pre_convergence_and_guards_selection():
    """The committed-artifact pattern: capped restarts, unusable best-restart.

    Terminal ELBOs span 1.3 nats while the last iteration still gains ~1.4
    nats, so `argmax(terminal_elbo)` ranks trajectories by how far along they
    are, not by mode quality. Both facts must be flagged structurally, not just
    in prose.
    """
    records = _cap_records([-2726.66, -2725.97, -2726.84, -2725.54], gain=1.4)
    block = convergence_block(records, n_iter_max=50, tol=1e-4)

    assert block["converged"] is False
    assert block["n_restarts_at_iter_cap"] == 4
    assert block["n_restarts_converged"] == 0
    assert block["median_final_elbo_gain"] == pytest.approx(1.4)
    assert block["terminal_elbo_spread"] == pytest.approx(1.3, abs=1e-9)
    assert block["best_restart_selection_meaningful"] is False
    assert len(block["warnings"]) == 2
    assert any("PRE-CONVERGENCE" in w for w in block["warnings"])
    assert any("NOT MEANINGFUL" in w for w in block["warnings"])


def test_convergence_block_silent_when_converged_and_well_separated():
    """A converged ensemble with a real ELBO gap raises nothing."""
    records = _cap_records([-2700.0, -2600.0], gain=0.001)
    for r in records:
        r["hit_iter_cap"] = False
        r["converged"] = True
        r["n_iter_run"] = 12
    block = convergence_block(records, n_iter_max=50, tol=1e-4)
    assert block["converged"] is True
    assert block["best_restart_selection_meaningful"] is True
    assert block["warnings"] == []


def test_restart_record_carries_its_convergence_verdict():
    """restart_record embeds the per-restart convergence fields (H1a)."""
    vo = replace(
        _inert_vem_output(),
        elbo_trace=np.array([-10.0, -8.0, -6.0]),
        n_iter_run=3,
        theta_w=np.array([0.1, 0.9, 0.5, 0.2]),
    )
    rec = restart_record(
        vo, seed=42, top_k=2, pooled_auc=0.9, n_iter_max=3, tol=1e-4
    )
    assert rec["seed"] == 42
    assert rec["terminal_elbo"] == -6.0
    assert rec["pooled_auc"] == 0.9
    assert rec["top_k_wallets"] == [1, 2]
    assert rec["hit_iter_cap"] is True
    assert rec["converged"] is False
    assert rec["final_elbo_gain"] == pytest.approx(2.0)
    assert rec["elbo_trace"] == [-10.0, -8.0, -6.0]


# ---------------- Stability escalation (H4) ----------------


def test_stability_block_escalates_wide_auc_spread():
    """A pooled-AUC spread past the threshold sets the flag and a warning."""
    records = _cap_records([-2726.0, -2725.0], gain=1.4)
    records[0]["pooled_auc"] = 0.376
    records[1]["pooled_auc"] = 0.915
    block = stability_block(records, top_k=2)

    assert block["pooled_auc_spread_threshold"] == AUC_SPREAD_THRESHOLD
    assert block["pooled_auc_unstable"] is True
    assert len(block["warnings"]) == 1
    warning = block["warnings"][0]
    # The wording must pin this to initialization on fixed data, and must say
    # explicitly that it is not the data-seed protocol's sensitivity.
    assert "INITIALIZATION" in warning
    assert "NOT data-seed sensitivity" in warning
    assert str(INIT_JITTER_LOG_SD) in warning
    assert block["mean_pairwise_topk_jaccard"] == 1.0


def test_stability_block_silent_within_threshold():
    """A tight AUC spread records the spread without escalating."""
    records = _cap_records([-2726.0, -2725.0], gain=1.4)
    records[0]["pooled_auc"] = 0.885
    records[1]["pooled_auc"] = 0.899
    block = stability_block(records, top_k=2)
    assert block["pooled_auc"]["spread"] == pytest.approx(0.014)
    assert block["pooled_auc_unstable"] is False
    assert block["warnings"] == []


def test_stability_block_no_auc_does_not_escalate():
    """Real-data runs (pooled_auc None) leave the flag off rather than NaN-true."""
    block = stability_block(_cap_records([-2726.0, -2725.0], gain=1.4), top_k=2)
    assert math.isnan(block["pooled_auc"]["spread"])
    assert block["pooled_auc_unstable"] is False
    assert block["warnings"] == []


# ---------------- Proposal-centring gradient (H2) ----------------


def test_centring_gradient_is_zero_at_a_true_mode():
    """A proposal centred on its target's mode reports ~0 in every dimension.

    With no markets the PSIS target is the prior alone, and `_AnalyticPrior` is
    exactly a Gaussian on the unconstrained scale, so its mode is `_REF_MEAN_U`.
    Centring the proposal there must give a vanishing gradient — this is the
    precondition khat presupposes.
    """
    result = phi_centring_gradient(
        _inert_vem_output(),
        _posterior(_REF_MEAN_U, _REF_VAR_U),
        markets=[],
        prior=_AnalyticPrior(),
    )
    np.testing.assert_allclose(result.grad_sd_units, 0.0, atol=1e-6)
    assert result.max_abs < 1e-6
    assert result.dims[0] == "sigma2_0"


def test_centring_gradient_recovers_a_known_displacement():
    """Off-mode by d sds gives exactly -d sd-units of gradient in that dim.

    For a Gaussian target with sd s, the log-target gradient at `mean + d*s` is
    `-d/s`; scaled by the proposal's sd (equal to s here) that is exactly `-d`.
    The two shifted dimensions must read -2 and +3, the rest 0.
    """
    sd = np.sqrt(_REF_VAR_U)
    shift = np.zeros_like(_REF_MEAN_U)
    shift[0] = 2.0 * sd[0]
    shift[4] = -3.0 * sd[4]
    result = phi_centring_gradient(
        _inert_vem_output(),
        _posterior(_REF_MEAN_U + shift, _REF_VAR_U),
        markets=[],
        prior=_AnalyticPrior(),
    )
    expected = np.zeros_like(_REF_MEAN_U)
    expected[0] = -2.0
    expected[4] = 3.0
    np.testing.assert_allclose(result.grad_sd_units, expected, atol=1e-5)
    assert result.max_abs_dim == "beta_S"
    assert result.max_abs == pytest.approx(3.0, abs=1e-5)


def test_centring_gradient_serializes_for_the_artifact():
    """to_dict keys the gradient by parameter name and carries the caveat."""
    result = phi_centring_gradient(
        _inert_vem_output(),
        _posterior(_REF_MEAN_U, _REF_VAR_U),
        markets=[],
        prior=_AnalyticPrior(),
    )
    payload = result.to_dict()
    assert set(payload["centring_grad_sd_units"]) == set(result.dims)
    assert payload["centring_grad_max_abs_sd"] == result.max_abs
    assert payload["centring_grad_max_abs_dim"] in result.dims
    assert "mode" in payload["centring_note"]


def test_centring_gradient_finite_on_the_market_fixture():
    """The ADF path produces a finite gradient with n_jobs invariance."""
    vo, markets, phi_posterior = _psis_market_fixture()
    seq = phi_centring_gradient(vo, phi_posterior, markets)
    par = phi_centring_gradient(vo, phi_posterior, markets, n_jobs=2)
    assert np.all(np.isfinite(seq.grad_sd_units))
    np.testing.assert_array_equal(seq.grad_sd_units, par.grad_sd_units)
