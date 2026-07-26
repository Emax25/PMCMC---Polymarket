"""Tests for src.inference.adf_filter.

`ADFFilter` was extracted from `variational_em._vem_e_step` as a pure refactor
(plan 2026-07-23-004 U4), so these tests are dominated by identity gates: the
pinned pre-refactor fixture, batch-vs-stepwise equivalence, and the O(1)-per-
trade cost the extraction exists to provide.
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from config.default_params import ModelParams
from src.data.synthetic import generate_market
from src.inference.adf_filter import ADFFilter, ADFStep
from src.inference.particle_gibbs import MarketData
from src.inference.variational_em import _vem_e_step

FIXTURES = Path(__file__).parent / "fixtures"

# Pre-refactor identity fixture: `_vem_e_step` outputs pinned from the
# implementation that existed before `ADFFilter` was extracted. Regenerating it
# is only legitimate when the ADF *model* deliberately changes, never to make a
# refactor pass.
IDENTITY_FIXTURE = FIXTURES / "adf_filter_identity.npz"


def _identity_case(degenerate: bool):
    """Inputs for one pinned identity case; must not drift from the fixture.

    Uses non-zero `beta_S`/`beta_Z` and a spread of `theta_w` so the logistic
    predictor (not just the Kalman/collapse arithmetic) is exercised. The
    ``degenerate`` case zeroes `log_size_ratio`, driving ``s_S`` to exactly 0.0
    so the `S_STD_FLOOR` centering-only fallback is pinned too.

    Args:
        degenerate: Whether to flatten `log_size_ratio` to a constant column.

    Returns:
        `(md, theta_w, params, m_S, s_S, m_Z)` ready for `_vem_e_step`.
    """
    rng = np.random.default_rng(0)
    base = ModelParams.warm_start(rng.standard_normal(200))
    params = replace(base, beta_S=0.7, beta_Z=1.3)
    mkt = generate_market(
        params,
        n_trades=200,
        n_wallets=10,
        n_insider_wallets=2,
        mean_inter_trade_time=1.0,
        rng=np.random.default_rng(3),
    )
    log_size_ratio = np.log(mkt.S / mkt.S_bar)
    if degenerate:
        log_size_ratio = np.zeros_like(log_size_ratio)
    md = MarketData(
        Y=mkt.Y,
        delta=mkt.delta,
        log_size_ratio=log_size_ratio,
        wallet_ids=mkt.wallet_ids,
    )
    theta_w = np.linspace(0.05, 0.6, 10)
    m_S = float(md.log_size_ratio.mean())
    s_S = float(md.log_size_ratio.std())
    m_Z = 0.0 if degenerate else 0.3
    return md, theta_w, params, m_S, s_S, m_Z


def _accumulate(values):
    """Sum left-to-right in float, matching the batch E-step's accumulator.

    Not `sum()`: CPython 3.12+ gives `sum()` a compensated (Neumaier)
    float fast path, which is *more* accurate than the E-step's plain
    `log_marginal += ...` and so disagrees with it in the last bits —
    enough to break an exact-equality identity assertion.
    """
    total = 0.0
    for v in values:
        total += v
    return total


def _run_batch(md, theta_w, params, m_S, s_S, m_Z):
    """Run the batch E-step on an `_identity_case` tuple."""
    return _vem_e_step(
        md.Y,
        md.delta,
        md.log_size_ratio,
        md.wallet_ids,
        theta_w,
        params,
        m_S,
        s_S,
        m_Z,
    )


def _gate_scale_markets(K=10, T=2000, n_wallets=40):
    """Synthetic markets at the benchmark gate scale (results/_bench_queue.sh)."""
    rng = np.random.default_rng(0)
    params = ModelParams.warm_start(rng.standard_normal(200))
    mds = []
    for k in range(K):
        mkt = generate_market(
            params,
            n_trades=T,
            n_wallets=n_wallets,
            n_insider_wallets=3,
            mean_inter_trade_time=1.0,
            rng=np.random.default_rng(42 + k),
        )
        mds.append(
            MarketData(
                Y=mkt.Y,
                delta=mkt.delta,
                log_size_ratio=np.log(mkt.S / mkt.S_bar),
                wallet_ids=mkt.wallet_ids,
            )
        )
    return mds, params


@pytest.mark.parametrize("case,degenerate", [("std", False), ("degen", True)])
def test_batch_e_step_matches_prerefactor_fixture(case, degenerate):
    """The refactored batch E-step reproduces the pinned pre-refactor outputs.

    The `ADFFilter` extraction is a pure refactor, so this is asserted at
    exact floating-point equality (`assert_array_equal`), not a tolerance:
    the per-trade operation order was preserved deliberately, and any drift —
    even at 1e-16 — means the arithmetic changed and should be reviewed rather
    than absorbed by a tolerance.
    """
    fixture = np.load(IDENTITY_FIXTURE)
    inputs = _identity_case(degenerate)
    q_vz, mu_f, sigma2_f, lm = _run_batch(*inputs)

    np.testing.assert_array_equal(q_vz, fixture[f"{case}_q_vz"])
    np.testing.assert_array_equal(q_vz[:, 1] + q_vz[:, 3], fixture[f"{case}_Z_prob"])
    np.testing.assert_array_equal(q_vz[:, 2] + q_vz[:, 3], fixture[f"{case}_V_prob"])
    np.testing.assert_array_equal(mu_f, fixture[f"{case}_X_mean"])
    np.testing.assert_array_equal(sigma2_f, fixture[f"{case}_X_var"])
    assert lm == float(fixture[f"{case}_log_marginal"])


def test_degenerate_case_exercises_s_S_floor():
    """Guard the fixture's intent: the degenerate case really has s_S == 0.0."""
    _, _, _, _, s_S, _ = _identity_case(degenerate=True)
    assert s_S == 0.0


@pytest.mark.parametrize("degenerate", [False, True])
def test_stepwise_equals_batch(degenerate):
    """Feeding trades one at a time through `ADFFilter` equals the batch arrays."""
    md, theta_w, params, m_S, s_S, m_Z = _identity_case(degenerate)
    q_vz, mu_f, sigma2_f, lm = _run_batch(md, theta_w, params, m_S, s_S, m_Z)

    adf = ADFFilter(params, theta_w, m_S, s_S, m_Z)
    steps = [
        adf.step(md.Y[t], md.delta[t], md.log_size_ratio[t], md.wallet_ids[t])
        for t in range(len(md.Y))
    ]

    assert all(isinstance(s, ADFStep) for s in steps)
    np.testing.assert_array_equal(np.array([s.q_vz for s in steps]), q_vz)
    np.testing.assert_array_equal(np.array([s.X_mean for s in steps]), mu_f)
    np.testing.assert_array_equal(np.array([s.X_var for s in steps]), sigma2_f)
    np.testing.assert_array_equal(
        np.array([s.Z_prob for s in steps]), q_vz[:, 1] + q_vz[:, 3]
    )
    np.testing.assert_array_equal(
        np.array([s.V_prob for s in steps]), q_vz[:, 2] + q_vz[:, 3]
    )
    assert _accumulate(s.log_evidence for s in steps) == lm
    assert adf.t == len(md.Y)

    # Each step must hand back its own q_vz array. A shared buffer would make a
    # caller that retains ADFStep objects (the live-scoring path) silently see
    # only the last trade's assignment.
    assert len({id(s.q_vz) for s in steps}) == len(steps)


def test_interleaved_filters_do_not_share_state():
    """Two `ADFFilter` instances advanced in lockstep stay independent.

    The live-scoring path runs one filter per market concurrently; any state
    held on the class (rather than the instance) would cross-contaminate them.
    """
    md_a, theta_w, params, m_S, s_S, m_Z = _identity_case(degenerate=False)
    # A second market with genuinely different data, so cross-talk cannot be
    # masked by the two streams coinciding.
    md_b = MarketData(
        Y=md_a.Y[::-1].copy() + 0.5,
        delta=md_a.delta[::-1].copy(),
        log_size_ratio=md_a.log_size_ratio[::-1].copy(),
        wallet_ids=md_a.wallet_ids[::-1].copy(),
    )
    solo = {
        name: _run_batch(md, theta_w, params, m_S, s_S, m_Z)
        for name, md in (("a", md_a), ("b", md_b))
    }

    filters = {
        "a": ADFFilter(params, theta_w, m_S, s_S, m_Z),
        "b": ADFFilter(params, theta_w, m_S, s_S, m_Z),
    }
    got = {"a": [], "b": []}
    for t in range(len(md_a.Y)):
        for name, md in (("a", md_a), ("b", md_b)):
            got[name].append(
                filters[name].step(
                    md.Y[t], md.delta[t], md.log_size_ratio[t], md.wallet_ids[t]
                )
            )

    for name in ("a", "b"):
        np.testing.assert_array_equal(
            np.array([s.q_vz for s in got[name]]), solo[name][0]
        )
        np.testing.assert_array_equal(
            np.array([s.X_mean for s in got[name]]), solo[name][1]
        )
        assert _accumulate(s.log_evidence for s in got[name]) == solo[name][3]


def test_reset_rewinds_to_trade_zero():
    """`reset` restores the initial state, so a filter can be re-driven exactly."""
    md, theta_w, params, m_S, s_S, m_Z = _identity_case(degenerate=False)
    adf = ADFFilter(params, theta_w, m_S, s_S, m_Z)

    def drive():
        return [
            adf.step(md.Y[t], md.delta[t], md.log_size_ratio[t], md.wallet_ids[t])
            for t in range(len(md.Y))
        ]

    first = drive()
    adf.reset()
    assert adf.t == 0
    second = drive()
    np.testing.assert_array_equal(
        np.array([s.q_vz for s in first]), np.array([s.q_vz for s in second])
    )


def test_z0_prior_pins_first_trade_and_stationary_V():
    """Trade 0 uses Z_0 := 0 and the stationary V prior, not a transition."""
    md, theta_w, params, m_S, s_S, m_Z = _identity_case(degenerate=False)
    adf = ADFFilter(params, theta_w, m_S, s_S, m_Z)
    rho_V = params.q_01 / (params.q_01 + params.q_10)
    np.testing.assert_array_equal(adf.prev_q_V, np.array([1.0 - rho_V, rho_V]))

    first = adf.step(md.Y[0], md.delta[0], md.log_size_ratio[0], md.wallet_ids[0])
    assert first.Z_prob < 1e-100  # exp(-500) prior, numerically zero
    np.testing.assert_allclose(first.q_vz.sum(), 1.0, rtol=0, atol=1e-12)


@pytest.mark.slow
def test_batch_e_step_is_linear_and_fast_at_gate_scale():
    """The E-step stays O(1) per trade and well inside its wall-clock budget.

    Two complementary guards, both deliberately generous — this is a floor
    against an algorithmic regression, not the +/-5% refactor gate (that one is
    a machine-local interleaved A/B, recorded in the U4 commit message; a
    pinned absolute figure would only re-learn the 68.8s environment-drift
    lesson in ARCHITECTURE/STATUS).

      * Linearity: per-trade cost at T = 8000 must not exceed ~1.6x the cost at
        T = 2000. An accidentally quadratic filter (e.g. re-filtering history
        each step) would blow straight past that.
      * Budget: gate-scale (10 markets x 2000 trades) throughput must beat
        250 us/trade, ~4x the ~60 us/trade measured for this implementation.
    """

    def per_trade_seconds(md, params, theta_w, m_S, s_S, m_Z, reps=3):
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            _run_batch(md, theta_w, params, m_S, s_S, m_Z)
            best = min(best, time.perf_counter() - t0)
        return best / len(md.Y)

    small, params = _gate_scale_markets(K=1, T=2000)
    large, _ = _gate_scale_markets(K=1, T=8000)
    theta_w = np.full(40, params.a / (params.a + params.b))
    lsr = np.concatenate([m.log_size_ratio for m in small + large])
    m_S, s_S, m_Z = float(lsr.mean()), float(lsr.std()), 0.1

    t_small = per_trade_seconds(small[0], params, theta_w, m_S, s_S, m_Z)
    t_large = per_trade_seconds(large[0], params, theta_w, m_S, s_S, m_Z)
    assert t_large < 1.6 * t_small, (
        f"per-trade cost grew {t_large / t_small:.2f}x from T=2000 to T=8000; "
        "the ADF pass must stay O(1) per trade"
    )

    mds, _ = _gate_scale_markets()
    t0 = time.perf_counter()
    for md in mds:
        _run_batch(md, theta_w, params, m_S, s_S, m_Z)
    per_trade = (time.perf_counter() - t0) / sum(len(md.Y) for md in mds)
    assert per_trade < 250e-6, (
        f"gate-scale E-step at {per_trade * 1e6:.1f} us/trade, budget 250 us"
    )
