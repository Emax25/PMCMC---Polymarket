"""Tests for src.analysis.prefilter (C4 microstructure prefilter)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from config.default_params import ModelParams
from src.analysis.prefilter import (
    PrefilterResult,
    prefilter_wallets,
    side_labels_to_signs,
    size_zscore_scores,
    subset_markets_to_wallets,
    volume_controlled_scores,
    vpin_robustness,
    vpin_scores,
    wallet_volumes,
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


# ---------------- VPIN: native side vs price-change proxy ----------------


def _misclassified_market() -> tuple[MarketData, np.ndarray]:
    """Market where the price-change proxy reverses the true toxicity order.

    Wallet 0 fires six aggressive BUYs (native VPIN = 1) into a zigzagging
    print price, so half of them tick the price *down* and bulk-volume
    classification calls them sells. Wallet 1 splits three BUYs and three SELLs
    (native VPIN = 0) into a monotone uptick, so the proxy calls them all buys.
    Equal sizes and ``n_buckets=2`` put each wallet in exactly one bucket.
    """
    Y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    market = _hand_market(
        Y=Y,
        delta=np.concatenate([[0.0], np.ones(11)]),
        log_size_ratio=np.zeros(12),
        wallet_ids=np.array([0] * 6 + [1] * 6),
    )
    sides = np.array(["BUY"] * 6 + ["BUY", "SELL", "BUY", "SELL", "BUY", "SELL"])
    return market, sides


def _volume_artifact_market() -> tuple[MarketData, np.ndarray]:
    """High-volume / no-toxicity market: VPIN tracks size, not informed flow.

    Six "whale" wallets trade once each with a size above the bucket target, so
    every whale fills a bucket alone and scores VPIN = 1 purely because it has
    no counterparty flow to net against — the Andersen-Bondarenko artifact.
    Six small wallets trade five unit-size lots each with strictly alternating
    sides, so their shared buckets net to ~0. No wallet is systematically
    one-directional, so true toxicity is flat across all twelve.
    """
    log_size_ratio: list[float] = []
    wallet_ids: list[int] = []
    sides: list[str] = []
    for b in range(6):
        log_size_ratio.append(float(np.log(24.0 + 2.0 * b)))
        wallet_ids.append(6 + b)
        sides.append("BUY" if b % 2 == 0 else "SELL")
    for w in range(6):
        for _ in range(5):
            log_size_ratio.append(0.0)
            wallet_ids.append(w)
    sides += ["BUY" if i % 2 == 0 else "SELL" for i in range(len(wallet_ids) - 6)]

    T = len(wallet_ids)
    market = _hand_market(
        Y=np.cumsum(np.tile([0.3, -0.3], T)[:T]),
        delta=np.concatenate([[0.0], np.ones(T - 1)]),
        log_size_ratio=np.array(log_size_ratio),
        wallet_ids=np.array(wallet_ids),
    )
    return market, np.array(sides)


def test_native_side_and_proxy_disagree_when_proxy_misclassifies():
    market, sides = _misclassified_market()
    native = vpin_scores([market], 2, sides=[sides])
    proxy = vpin_scores([market], 2)

    # Native side is unambiguous: wallet 0's bucket is 100% buy, wallet 1's nets out.
    np.testing.assert_allclose(native, [1.0, 0.0])
    # The proxy not only differs, it inverts the ordering.
    assert not np.allclose(native, proxy)
    assert proxy[0] < proxy[1]
    assert native[0] > native[1]


def test_side_labels_and_signs_agree():
    market, sides = _misclassified_market()
    signs = np.where(sides == "BUY", 1.0, -1.0)
    np.testing.assert_array_equal(side_labels_to_signs(sides), signs)
    np.testing.assert_allclose(
        vpin_scores([market], 2, sides=[sides]),
        vpin_scores([market], 2, sides=[signs]),
    )


def test_unclassified_sides_fall_back_to_proxy_exactly():
    """Sign 0 means 'venue published no side'; those trades keep the proxy."""
    market, _ = _misclassified_market()
    unknown = np.zeros(market.T)
    np.testing.assert_array_equal(
        vpin_scores([market], 2, sides=[unknown]),
        vpin_scores([market], 2),
    )


def test_partial_sides_only_override_labelled_trades():
    market, sides = _misclassified_market()
    signs = np.where(sides == "BUY", 1.0, -1.0)
    partial = signs.copy()
    partial[6:] = 0.0  # wallet 1 unlabelled -> proxy; wallet 0 native
    scores = vpin_scores([market], 2, sides=[partial])
    assert scores[0] == pytest.approx(vpin_scores([market], 2, sides=[signs])[0])
    assert scores[1] == pytest.approx(vpin_scores([market], 2)[1])


def test_vpin_robustness_reports_labelled_comparison():
    market, sides = _misclassified_market()
    rb = vpin_robustness([market], [sides], n_buckets=2)

    np.testing.assert_allclose(rb.native, vpin_scores([market], 2, sides=[sides]))
    np.testing.assert_allclose(rb.proxy, vpin_scores([market], 2))
    assert rb.n_wallets_compared == 2
    # Proxy reverses the native ranking on this fixture: maximal disagreement.
    assert rb.rank_correlation == pytest.approx(-1.0)


def test_vpin_robustness_rank_correlation_is_nan_without_variation():
    """One traded wallet leaves the Spearman statistic undefined, not 0."""
    market = _hand_market(
        Y=np.array([0.0, 0.5, 1.0]),
        delta=np.array([0.0, 1.0, 1.0]),
        log_size_ratio=np.zeros(3),
        wallet_ids=np.zeros(3, dtype=int),
    )
    rb = vpin_robustness([market], [np.array(["BUY", "BUY", "SELL"])], n_buckets=2)
    assert rb.n_wallets_compared == 1
    assert np.isnan(rb.rank_correlation)


def test_vpin_rejects_mismatched_sides():
    market, sides = _misclassified_market()
    with pytest.raises(ValueError, match="one per market"):
        vpin_scores([market], 2, sides=[sides, sides])
    with pytest.raises(ValueError, match="expected"):
        vpin_scores([market], 2, sides=[sides[:5]])
    with pytest.raises(ValueError, match="unsupported dtype"):
        vpin_scores([market], 2, sides=[sides == "BUY"])


# ---------------- VPIN: volume controls ----------------


def test_wallet_volumes_sum_relative_size():
    market = _hand_market(
        Y=np.array([0.0, 0.1, 0.2]),
        delta=np.array([0.0, 1.0, 1.0]),
        log_size_ratio=np.array([0.0, np.log(2.0), np.log(3.0)]),
        wallet_ids=np.array([0, 0, 1]),
    )
    np.testing.assert_allclose(wallet_volumes([market]), [3.0, 3.0])


def test_volume_control_decorrelates_vpin_from_volume():
    market, sides = _volume_artifact_market()
    vpin = vpin_scores([market], 14, sides=[sides])
    volumes = wallet_volumes([market])
    residual = volume_controlled_scores(vpin, volumes)

    log_vol = np.log(volumes)
    raw_pearson = float(np.corrcoef(vpin, log_vol)[0, 1])
    res_pearson = float(np.corrcoef(residual, log_vol)[0, 1])
    raw_spearman = float(spearmanr(vpin, volumes).statistic)
    res_spearman = float(spearmanr(residual, volumes).statistic)

    # The artifact is present in the raw score...
    assert raw_pearson > 0.9
    assert raw_spearman > 0.9
    # ...and the control removes it (OLS residuals are orthogonal to log volume).
    assert abs(res_pearson) < 1e-8
    assert abs(res_spearman) < 0.5 * raw_spearman
    assert np.any(residual != 0.0)  # not a degenerate all-zero fit


def test_volume_controlled_scores_leaves_untraded_wallets_at_zero():
    scores = np.array([0.4, 0.6, 0.0])
    volumes = np.array([2.0, 8.0, 0.0])
    residual = volume_controlled_scores(scores, volumes)
    assert residual[2] == 0.0
    # Two points, two parameters: the fit is saturated, so nothing is left over.
    np.testing.assert_allclose(residual[:2], 0.0, atol=1e-12)


def test_volume_controlled_scores_single_wallet_is_degenerate_zero():
    residual = volume_controlled_scores(np.array([0.7]), np.array([5.0]))
    np.testing.assert_array_equal(residual, [0.0])


def test_volume_controlled_scores_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        volume_controlled_scores(np.zeros(3), np.zeros(2))


# ---------------- VPIN: edge cases with sides ----------------


def test_vpin_empty_market_with_sides():
    market = _hand_market(
        Y=np.array([]),
        delta=np.array([]),
        log_size_ratio=np.array([]),
        wallet_ids=np.array([], dtype=int),
    )
    scores = vpin_scores([market], 4, sides=[np.array([], dtype=float)])
    assert scores.size == 0
    assert wallet_volumes([market]).size == 0


def test_vpin_all_one_side_market_is_maximally_toxic():
    T = 8
    market = _hand_market(
        Y=np.linspace(0.0, 0.7, T),
        delta=np.concatenate([[0.0], np.ones(T - 1)]),
        log_size_ratio=np.zeros(T),
        wallet_ids=np.arange(T) % 2,
    )
    scores = vpin_scores([market], 4, sides=[np.array(["BUY"] * T)])
    np.testing.assert_allclose(scores, 1.0)


def test_vpin_tied_prices_give_neutral_proxy_but_sharp_native():
    """Flat prices leave the proxy with no information; native side still cuts."""
    T = 6
    market = _hand_market(
        Y=np.zeros(T),
        delta=np.concatenate([[0.0], np.ones(T - 1)]),
        log_size_ratio=np.zeros(T),
        wallet_ids=np.zeros(T, dtype=int),
    )
    proxy = vpin_scores([market], 2)
    native = vpin_scores([market], 2, sides=[np.array(["SELL"] * T)])
    np.testing.assert_allclose(proxy, 0.0)  # buy_frac == 0.5 everywhere
    np.testing.assert_allclose(native, 1.0)


def test_vpin_market_shorter_than_one_bucket():
    """T < n_buckets: trailing buckets stay empty and scores stay in [0, 1]."""
    market = _hand_market(
        Y=np.array([0.0, 0.4, 0.2]),
        delta=np.array([0.0, 1.0, 1.0]),
        log_size_ratio=np.zeros(3),
        wallet_ids=np.array([0, 1, 0]),
    )
    for sides in (None, [np.array(["BUY", "SELL", "BUY"])]):
        scores = vpin_scores([market], 50, sides=sides)
        assert scores.shape == (2,)
        assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_vpin_rejects_bad_n_buckets():
    market, sides = _misclassified_market()
    with pytest.raises(ValueError, match="n_buckets"):
        vpin_scores([market], 0, sides=[sides])


# ---------------- Backward compatibility (golden lock) ----------------

# Captured from the pre-side-support implementation (git HEAD before U3) on the
# seed-42 synthetic fixture. Adding the `sides` path must not move the default
# scores by even a ULP, since the whole C4 gate is calibrated on them.
_GOLDEN_VPIN = np.array(
    [
        0.21219000076570954,
        0.23950886769871094,
        0.33589094018758340,
        0.25726204241801615,
        0.37453437860210690,
        0.21081156934363277,
        0.32184846559634430,
        0.16725400180924980,
        0.12152948519785790,
        0.22801624202266200,
        0.29047526490813147,
        0.38189836985811630,
        0.19020749857524918,
        0.22137260247053966,
        0.21686187462375747,
        0.21604606970621348,
        0.17701486921514048,
        0.30977995128809990,
        0.15749090953727682,
        0.19487124905606854,
    ]
)
_GOLDEN_COMBINED = (
    np.array(
        [40, 41, 47, 28, 28, 18, 31, 22, 20, 19, 25, 33, 26, 27, 25, 37, 17, 32, 15, 39]
    )
    / 3.0
)
_GOLDEN_FLAGGED = np.array(
    [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1], dtype=bool
)


def test_vpin_default_path_matches_pre_side_support_golden():
    markets, _ = _synthetic_markets(seed=42)
    np.testing.assert_allclose(vpin_scores(markets), _GOLDEN_VPIN, rtol=0, atol=0)
    np.testing.assert_allclose(
        vpin_scores(markets, sides=None), _GOLDEN_VPIN, rtol=0, atol=0
    )


def test_prefilter_wallets_default_output_unchanged():
    markets, _ = _synthetic_markets(seed=42)
    out = prefilter_wallets(markets, quantile=0.5)
    np.testing.assert_allclose(out.scores, _GOLDEN_COMBINED, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(out.flagged, _GOLDEN_FLAGGED)
    np.testing.assert_allclose(
        out.component_scores["vpin"], _GOLDEN_VPIN, rtol=0, atol=0
    )


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
