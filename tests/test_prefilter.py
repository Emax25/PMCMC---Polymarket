"""Tests for src.analysis.prefilter (C4 microstructure prefilter)."""

from __future__ import annotations

import numpy as np
import pytest

from config.default_params import ModelParams
from src.analysis.prefilter import (
    PrefilterResult,
    prefilter_wallets,
    size_zscore_scores,
    subset_markets_to_wallets,
    vpin_scores,
    wash_trade_scores,
)
from src.data.synthetic import generate_dataset
from src.inference.particle_gibbs import MarketData


def _hand_market(
    *,
    Y: np.ndarray,
    delta: np.ndarray,
    log_size_ratio: np.ndarray,
    wallet_ids: np.ndarray,
) -> MarketData:
    return MarketData(
        Y=Y,
        delta=delta,
        log_size_ratio=log_size_ratio,
        wallet_ids=wallet_ids,
    )


def _synthetic_markets(*, seed: int = 42):
    rng = np.random.default_rng(0)
    params = ModelParams.warm_start(rng.standard_normal(200))
    synth = generate_dataset(
        params,
        n_markets=4,
        n_trades=300,
        n_wallets=20,
        n_insider_wallets=3,
        rng=np.random.default_rng(seed),
    )
    markets = [
        MarketData(
            Y=m.Y,
            delta=m.delta,
            log_size_ratio=np.log(m.S / m.S_bar),
            wallet_ids=m.wallet_ids,
        )
        for m in synth
    ]
    insider_ids = synth[0].insider_wallet_ids
    return markets, insider_ids


# ---------------- Shape / range sanity ----------------


def test_size_zscore_scores_shape_and_range():
    m = _hand_market(
        Y=np.array([0.0, 0.1, -0.1]),
        delta=np.array([0.0, 1.0, 2.0]),
        log_size_ratio=np.array([0.0, 1.0, -1.0]),
        wallet_ids=np.array([0, 1, 0]),
    )
    scores = size_zscore_scores([m])
    assert scores.shape == (2,)
    assert np.all(scores >= 0)
    assert scores[0] > 0
    assert scores[1] > 0


def test_vpin_scores_shape_and_range():
    rng = np.random.default_rng(1)
    T = 40
    m = _hand_market(
        Y=rng.standard_normal(T),
        delta=np.concatenate([[0.0], np.ones(T - 1)]),
        log_size_ratio=rng.normal(0, 0.5, T),
        wallet_ids=rng.integers(0, 5, T),
    )
    scores = vpin_scores([m], n_buckets=10)
    assert scores.shape == (5,)
    assert np.all((scores >= 0) & (scores <= 1))


def test_wash_trade_scores_shape_and_range():
    Y = np.array([0.0, 0.5, -0.5, 0.3, -0.2])
    delta = np.array([0.0, 10.0, 20.0, 5.0, 5.0])
    m = _hand_market(
        Y=Y,
        delta=delta,
        log_size_ratio=np.zeros(5),
        wallet_ids=np.array([0, 0, 0, 1, 1]),
    )
    scores = wash_trade_scores([m], window_seconds=30.0)
    assert scores.shape == (2,)
    assert np.all((scores >= 0) & (scores <= 1))


def test_prefilter_wallets_returns_dataclass():
    markets, _ = _synthetic_markets(seed=1)
    out = prefilter_wallets(markets, quantile=0.5)
    assert isinstance(out, PrefilterResult)
    assert out.scores.shape == (20,)
    assert out.flagged.shape == (20,)
    assert set(out.component_scores) == {"size_zscore", "vpin", "wash"}


# ---------------- Recall on synthetic insiders ----------------


def test_prefilter_recalls_all_insiders_at_half_flag_rate():
    markets, insider_ids = _synthetic_markets(seed=42)
    out = prefilter_wallets(markets, quantile=0.5)
    n_flagged = int(out.flagged.sum())
    assert n_flagged >= 10  # top 50% of 20 wallets
    assert np.all(out.flagged[insider_ids]), (
        f"missed insiders { [w for w in insider_ids if not out.flagged[w]] }; "
        f"component ranks: "
        f"{ {k: out.component_scores[k][insider_ids] for k in out.component_scores} }"
    )


def test_size_zscore_alone_discriminates_insiders():
    """Size z-score should rank insiders above median on synthetic data."""
    markets, insider_ids = _synthetic_markets(seed=42)
    sz = size_zscore_scores(markets)
    median = float(np.median(sz))
    assert np.all(
        sz[insider_ids] >= median
    ), f"insider sz={sz[insider_ids]}, median={median}"


# ---------------- Determinism ----------------


def test_prefilter_deterministic():
    markets, _ = _synthetic_markets(seed=7)
    a = prefilter_wallets(markets, quantile=0.5)
    b = prefilter_wallets(markets, quantile=0.5)
    np.testing.assert_array_equal(a.scores, b.scores)
    np.testing.assert_array_equal(a.flagged, b.flagged)
    for key in a.component_scores:
        np.testing.assert_array_equal(a.component_scores[key], b.component_scores[key])


# ---------------- Edge cases ----------------


def test_single_wallet_market():
    m = _hand_market(
        Y=np.array([0.0, 0.2]),
        delta=np.array([0.0, 5.0]),
        log_size_ratio=np.array([0.0, 0.5]),
        wallet_ids=np.array([0, 0]),
    )
    out = prefilter_wallets([m], quantile=0.5)
    assert out.flagged.shape == (1,)
    assert out.flagged[0]


def test_wallet_with_one_trade_scores_zero_wash():
    m = _hand_market(
        Y=np.array([0.0, 0.3]),
        delta=np.array([0.0, 1.0]),
        log_size_ratio=np.array([0.0, 0.0]),
        wallet_ids=np.array([0, 1]),
    )
    wash = wash_trade_scores([m])
    assert wash[1] == 0.0


def test_empty_markets_list():
    out = prefilter_wallets([])
    assert out.scores.size == 0
    assert out.flagged.size == 0


def test_subset_markets_drops_small_markets_and_preserves_delta():
    n = 21
    delta = np.concatenate([[0.0], np.ones(n - 1)])
    wallet_ids = np.zeros(n, dtype=int)
    wallet_ids[1::2] = 1  # wallet 0 has 11 trades (>= 10 minimum)
    m = _hand_market(
        Y=np.linspace(0, 1, n),
        delta=delta,
        log_size_ratio=np.zeros(n),
        wallet_ids=wallet_ids,
    )
    keep = np.zeros(2, dtype=bool)
    keep[0] = True
    subset, idx_maps = subset_markets_to_wallets([m], keep)
    assert len(subset) == 1
    assert len(idx_maps) == 1
    assert subset[0].Y.size == 11
    # Survivors at even indices; gap from index 0 to 2 sums delta[1:3]
    assert subset[0].delta[0] == 0.0
    assert subset[0].delta[1] == pytest.approx(2.0)


def test_subset_empty_overlap_drops_market():
    m = _hand_market(
        Y=np.arange(12, dtype=float),
        delta=np.concatenate([[0.0], np.ones(11)]),
        log_size_ratio=np.zeros(12),
        wallet_ids=np.zeros(12, dtype=int),
    )
    keep = np.array([False])
    subset, idx_maps = subset_markets_to_wallets([m], keep)
    assert subset == []
    assert idx_maps == []
