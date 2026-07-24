"""Tests for src.analysis.validation (held-out predictive log-likelihood)."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from config.default_params import ModelParams
from src.analysis.validation import (
    HeldoutLL,
    heldout_predictive_ll,
    heldout_predictive_summary,
    holdout_split,
)
from src.data.synthetic import generate_market
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
