"""C4 hybrid microstructure prefilter for wallet shortlisting (Stage 3).

Cheap O(n_trades) heuristics rank wallets before the expensive Bayesian core
(Particle Gibbs / variational EM) runs only on the flagged subset. Recall on
planted insiders is the gate metric; precision is secondary.

Three component scores (rank-combined in ``prefilter_wallets``):

  1. **Size z-score** — max |z| of ``log_size_ratio`` within each market.
     Strong on synthetic data because insiders trade ~3x more often, giving
     more draws from the size tail (sizes themselves are i.i.d. lognormal).
  2. **VPIN proxy** — volume-synchronized order-flow toxicity adapted from
     Easley, Lopez de Prado, and O'Hara (2012); see ``vpin_scores``.
     Motivated by real-market informed-flow patterns; weak on synthetic data
     where price moves are not size-linked.
  3. **Wash-trade heuristic** — rapid same-wallet round-trips with opposing
     price moves. Real-data-motivated; typically weak on synthetic data.

References:
  easley2012vpin — Easley, D., Lopez de Prado, M. M., & O'Hara, M. (2012).
      Flow toxicity and liquidity in a high-frequency world. *RFS*, 25(5).
  kyle1985insider — Kyle, A. S. (1985). Continuous auctions and insider trading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from src.inference.particle_gibbs import MarketData

_MIN_STD = 1e-12
_MIN_MARKET_TRADES = 10


@dataclass
class PrefilterResult:
    """Output of ``prefilter_wallets``.

    Attributes:
        scores: Combined weighted rank score per wallet; higher = more suspicious.
        flagged: Boolean mask; ``True`` for wallets passed to the Bayesian core.
        component_scores: Raw scores keyed by
            ``"size_zscore"``, ``"vpin"``, ``"wash"``.
    """

    scores: np.ndarray
    flagged: np.ndarray
    component_scores: dict[str, np.ndarray]


def _n_wallets_from_markets(markets: list[MarketData]) -> int:
    """Infer global wallet count as max wallet id + 1."""
    max_id = -1
    for market in markets:
        if market.wallet_ids.size:
            max_id = max(max_id, int(market.wallet_ids.max()))
    return max_id + 1


def size_zscore_scores(markets: list[MarketData]) -> np.ndarray:
    """Per-wallet max |z-score| of log trade size within each market.

    For market k with trades i = 1..T_k, let ``lsr_i = log(S_i / S_bar_k)``.
    Per-market z-score::

        z_i = (lsr_i - mean(lsr)) / std(lsr)

    Wallet w score::

        s_w = max_{i : wallet_i = w} |z_i|

    across all markets. Wallets with no trades score 0.

    Args:
        markets: Observed markets in ``MarketData`` format.

    Returns:
        Array of shape ``(n_wallets,)`` with non-negative scores.
    """
    n_wallets = _n_wallets_from_markets(markets)
    scores = np.zeros(n_wallets, dtype=float)

    for market in markets:
        lsr = market.log_size_ratio
        n_trades = lsr.size
        if n_trades == 0:
            continue
        std = float(np.std(lsr))
        if std < _MIN_STD:
            z_abs = np.zeros(n_trades, dtype=float)
        else:
            mean = float(np.mean(lsr))
            z_abs = np.abs((lsr - mean) / std)
        np.maximum.at(scores, market.wallet_ids, z_abs)

    return scores


def vpin_scores(markets: list[MarketData], n_buckets: int = 50) -> np.ndarray:
    """VPIN-style order-flow toxicity proxy for prediction markets.

    Adapted from Easley, Lopez de Prado, and O'Hara (2012). Signed volume is
    unavailable, so trade direction is inferred from the price change
    ``dY_i = Y_i - Y_{i-1}`` (bulk-volume classification): buy fraction
    ``Phi(dY_i / sigma_dY)`` where ``Phi`` is the standard normal CDF.
    Trade volume proxy is ``v_i = exp(log_size_ratio_i)`` (relative size).

    Per market, trades are bucketed chronologically into ``n_buckets`` equal-
    volume buckets (O(T) greedy fill). Bucket toxicity::

        VPIN_b = |V_buy_b - V_sell_b| / V_b

    where ``V_buy_b = sum_i Phi(...) * v_i`` over trades in bucket b.

    Per-wallet score is the volume-weighted mean VPIN of buckets the wallet
    trades in::

        s_w = sum_{i: w_i=w} VPIN_{b(i)} * v_i / sum_{i: w_i=w} v_i

    Approximations (documented for reviewers):
      - Direction from ``dY`` not true aggressor side.
      - First trade in each market gets neutral buy fraction 0.5 (no ``dY``).
      - Trades are assigned wholly to one bucket (no split at boundaries).
      - Relative ``exp(log_size_ratio)`` ranks toxicity; not USDC volume.

    Args:
        markets: Observed markets in ``MarketData`` format.
        n_buckets: Number of equal-volume buckets per market (>= 1).

    Returns:
        Array of shape ``(n_wallets,)`` with scores in [0, 1] where defined.
    """
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")

    n_wallets = _n_wallets_from_markets(markets)
    weighted = np.zeros(n_wallets, dtype=float)
    vol_tot = np.zeros(n_wallets, dtype=float)

    for market in markets:
        T = market.Y.size
        if T == 0:
            continue

        vol = np.exp(market.log_size_ratio)
        buy_frac = np.full(T, 0.5, dtype=float)
        if T > 1:
            dY = np.diff(market.Y)
            std_dY = float(np.std(dY))
            if std_dY >= _MIN_STD:
                buy_frac[1:] = norm.cdf(dY / std_dY)

        bucket_vpin, bucket_ids = _equal_volume_buckets(vol, buy_frac, n_buckets)
        trade_vpin = bucket_vpin[bucket_ids]
        wids = market.wallet_ids

        np.add.at(weighted, wids, trade_vpin * vol)
        np.add.at(vol_tot, wids, vol)

    scores = np.zeros(n_wallets, dtype=float)
    mask = vol_tot > 0
    scores[mask] = weighted[mask] / vol_tot[mask]
    return scores


def _equal_volume_buckets(
    vol: np.ndarray,
    buy_frac: np.ndarray,
    n_buckets: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy equal-volume bucketing in chronological order."""
    T = vol.size
    bucket_ids = np.zeros(T, dtype=int)
    bucket_vpin = np.zeros(n_buckets, dtype=float)

    total_vol = float(vol.sum())
    if total_vol < _MIN_STD:
        return bucket_vpin, bucket_ids

    target = total_vol / n_buckets
    b_idx = 0
    b_buy = 0.0
    b_sell = 0.0
    b_vol = 0.0

    for i in range(T):
        bucket_ids[i] = min(b_idx, n_buckets - 1)
        v_i = vol[i]
        b_buy += buy_frac[i] * v_i
        b_sell += (1.0 - buy_frac[i]) * v_i
        b_vol += v_i

        if b_vol >= target and b_idx < n_buckets - 1:
            bucket_vpin[b_idx] = abs(b_buy - b_sell) / b_vol
            b_idx += 1
            b_buy = 0.0
            b_sell = 0.0
            b_vol = 0.0

    if b_vol > 0:
        bucket_vpin[b_idx] = abs(b_buy - b_sell) / b_vol

    return bucket_vpin, bucket_ids


def wash_trade_scores(
    markets: list[MarketData],
    window_seconds: float = 60.0,
) -> np.ndarray:
    """Heuristic self-trading / circularity score per wallet.

    For wallet w in market k, let ``t_i = sum_{j<=i} delta_j`` and
    ``dY_i = Y_i - Y_{i-1}`` (``dY_0`` undefined). Among trades with
    ``i >= 1`` and ``dY_i != 0``, the fraction that have another trade j
    by the same wallet in the same market with ``|t_i - t_j| <= window`` and
    ``sign(dY_i) != sign(dY_j)``::

        s_w = #{i eligible : exists j} / #{i eligible}

    Wallets with fewer than 2 trades in a market contribute nothing there.
    Scores are pooled across markets by trade-count-weighted average.

    Args:
        markets: Observed markets in ``MarketData`` format.
        window_seconds: Pairing window in seconds (same units as ``delta``).

    Returns:
        Array of shape ``(n_wallets,)`` with scores in [0, 1].
    """
    n_wallets = _n_wallets_from_markets(markets)
    hit_sum = np.zeros(n_wallets, dtype=float)
    eligible_sum = np.zeros(n_wallets, dtype=float)

    for market in markets:
        T = market.Y.size
        if T < 2:
            continue

        times = np.cumsum(market.delta)
        dY = np.diff(market.Y)
        signs = np.zeros(T, dtype=float)
        signs[1:] = np.sign(dY)

        for wallet in np.unique(market.wallet_ids):
            idx = np.flatnonzero(market.wallet_ids == wallet)
            if idx.size < 2:
                continue

            s_w = signs[idx]
            eligible = idx[s_w != 0.0]
            if eligible.size == 0:
                continue

            t_elig = times[eligible]
            s_elig = signs[eligible]
            dt = np.abs(t_elig[:, None] - t_elig[None, :])
            opp = s_elig[:, None] * s_elig[None, :] < 0
            np.fill_diagonal(dt, np.inf)
            has_pair = np.any((dt <= window_seconds) & opp, axis=1)
            hit_sum[wallet] += float(has_pair.sum())
            eligible_sum[wallet] += float(eligible.size)

    scores = np.zeros(n_wallets, dtype=float)
    mask = eligible_sum > 0
    scores[mask] = hit_sum[mask] / eligible_sum[mask]
    return scores


def _ordinal_ranks(scores: np.ndarray) -> np.ndarray:
    """Return 0..n-1 ranks; higher score gets higher rank."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(scores.size, dtype=float)
    return ranks


def prefilter_wallets(
    markets: list[MarketData],
    *,
    quantile: float = 0.5,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> PrefilterResult:
    """Rank-combine microstructure scores and flag suspicious wallets.

    Each component score is converted to an ordinal rank (higher raw score ->
    higher rank). Combined score::

        s_w = (w1*r1_w + w2*r2_w + w3*r3_w) / (w1 + w2 + w3)

    Wallets with the top ``1 - quantile`` fraction by ``s_w`` are flagged.
    Always flags at least ``ceil(0.1 * n_wallets)`` wallets (recall gate).

    Args:
        markets: Observed markets in ``MarketData`` format.
        quantile: Fraction of wallets *not* flagged (e.g. 0.5 -> flag top 50%).
        weights: Non-negative weights for
            (size_zscore, vpin, wash) rank combination.

    Returns:
        ``PrefilterResult`` with combined scores, flag mask, and components.

    Raises:
        ValueError: If ``quantile`` is not in [0, 1) or weights are negative.
    """
    if not (0.0 <= quantile < 1.0):
        raise ValueError("quantile must be in [0, 1)")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")

    n_wallets = _n_wallets_from_markets(markets)
    if n_wallets == 0:
        empty = np.array([], dtype=float)
        return PrefilterResult(
            scores=empty,
            flagged=empty.astype(bool),
            component_scores={
                "size_zscore": empty,
                "vpin": empty,
                "wash": empty,
            },
        )

    comp = {
        "size_zscore": size_zscore_scores(markets),
        "vpin": vpin_scores(markets),
        "wash": wash_trade_scores(markets),
    }
    w1, w2, w3 = weights
    denom = w1 + w2 + w3
    if denom <= 0:
        raise ValueError("sum of weights must be positive")

    combined = (
        w1 * _ordinal_ranks(comp["size_zscore"])
        + w2 * _ordinal_ranks(comp["vpin"])
        + w3 * _ordinal_ranks(comp["wash"])
    ) / denom

    n_flag = max(
        int(math.ceil((1.0 - quantile) * n_wallets)),
        int(math.ceil(0.1 * n_wallets)),
    )
    n_flag = min(n_flag, n_wallets)
    top = np.argsort(-combined, kind="mergesort")[:n_flag]
    flagged = np.zeros(n_wallets, dtype=bool)
    flagged[top] = True

    return PrefilterResult(
        scores=combined,
        flagged=flagged,
        component_scores=comp,
    )


def subset_markets_to_wallets(
    markets: list[MarketData],
    keep: np.ndarray,
) -> tuple[list[MarketData], list[np.ndarray]]:
    """Drop trades from non-kept wallets; preserve elapsed time in ``delta``.

    Trades whose ``wallet_ids`` are not in ``keep`` are removed. For the
    surviving subsequence, ``delta`` is rebuilt so inter-arrival times match
    the original timeline: elapsed time from survivor ``s_{j-1}`` to ``s_j`` is
    ``sum(delta[s_{j-1}+1 : s_j+1])`` (with ``delta[0]=0`` on the first
    survivor). Markets with fewer than ``_MIN_MARKET_TRADES`` surviving trades
    are omitted.

    Args:
        markets: Full observed markets.
        keep: Boolean mask of length ``n_wallets``; ``True`` retains a wallet.

    Returns:
        Tuple of filtered markets and, for each retained market, the array of
        original trade indices that survived (for trace-back).
    """
    kept_wallets = np.flatnonzero(keep)
    out_markets: list[MarketData] = []
    index_maps: list[np.ndarray] = []

    for market in markets:
        trade_keep = np.isin(market.wallet_ids, kept_wallets)
        orig_idx = np.flatnonzero(trade_keep)
        if orig_idx.size < _MIN_MARKET_TRADES:
            continue

        new_delta = _rebuild_delta(market.delta, orig_idx)
        out_markets.append(
            MarketData(
                Y=market.Y[orig_idx],
                delta=new_delta,
                log_size_ratio=market.log_size_ratio[orig_idx],
                wallet_ids=market.wallet_ids[orig_idx],
            )
        )
        index_maps.append(orig_idx)

    return out_markets, index_maps


def _rebuild_delta(delta: np.ndarray, surviving: np.ndarray) -> np.ndarray:
    """Sum dropped inter-arrival gaps into the next surviving trade."""
    new_delta = np.zeros(surviving.size, dtype=delta.dtype)
    for j in range(1, surviving.size):
        prev_i = int(surviving[j - 1])
        curr_i = int(surviving[j])
        new_delta[j] = float(delta[prev_i + 1 : curr_i + 1].sum())
    return new_delta
