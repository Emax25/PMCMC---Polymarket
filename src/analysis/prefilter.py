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

**VPIN is a gating signal only — never a detector.** The filter-only detection
ablation is a recorded GATE FAIL (pooled AUC 0.524 at K=10/T=2000), so VPIN
must only ever narrow the candidate set handed to the Bayesian core, and its
score must never be reported as evidence that a wallet is informed. Two rules
follow, and both are enforced by the helpers in this module:

  * **Native side beats the proxy.** Andersen and Bondarenko (2014) show that
    VPIN's apparent toxicity signal is substantially an artifact of the
    bulk-volume classification (BVC) scheme rather than of informed flow:
    BVC's inferred buy fraction is driven by the volume and clustering of
    trades in a bucket, so large trades mechanically produce large bucket
    imbalances. Where the venue publishes a native taker side (Kalshi does on
    every trade), pass it via ``sides=`` and classify with it; the
    price-change proxy is then a *labelled robustness comparison*
    (``vpin_robustness``), not independent confirmation.
  * **Gating with volume controls.** Andersen and Bondarenko further find that
    VPIN's predictive power does not survive controls for trading volume. Any
    downstream analysis that uses these scores must include volume as a
    covariate/control; ``wallet_volumes`` and ``volume_controlled_scores``
    provide the per-wallet volume and the residualized score for that purpose.

References:
  easley2012vpin — Easley, D., Lopez de Prado, M. M., & O'Hara, M. (2012).
      Flow toxicity and liquidity in a high-frequency world. *RFS*, 25(5).
  andersen2014vpin — Andersen, T. G., & Bondarenko, O. (2014). VPIN and the
      flash crash. *Journal of Financial Markets*, 17, 1-46.
  kyle1985insider — Kyle, A. S. (1985). Continuous auctions and insider trading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, spearmanr

from src.inference.particle_gibbs import MarketData

_MIN_STD = 1e-12
_MIN_MARKET_TRADES = 10

# Repo-wide taker-side labels (``RawTrade.side``; the Kalshi adapter maps
# ``taker_side`` yes/no onto them) as aggressor signs. Anything else — a blank
# string, an unknown venue label — becomes 0.0, meaning "unclassified", and
# falls back to the bulk-volume price-change proxy for that trade only.
_SIDE_LABEL_TO_SIGN = {"BUY": 1.0, "SELL": -1.0}


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


def side_labels_to_signs(labels: np.ndarray) -> np.ndarray:
    """Map venue taker-side labels onto aggressor signs.

    ``"BUY" -> +1.0``, ``"SELL" -> -1.0``, anything else (blank, unknown venue
    label) ``-> 0.0`` meaning unclassified. Matches ``RawTrade.side``, which the
    Kalshi adapter populates from the native ``taker_side`` field.

    Args:
        labels: Per-trade side labels; compared case-insensitively.

    Returns:
        Float array of shape ``labels.shape`` with values in ``{-1, 0, +1}``.
    """
    upper = np.char.upper(np.asarray(labels, dtype=str))
    signs = np.zeros(upper.shape, dtype=float)
    for label, sign in _SIDE_LABEL_TO_SIGN.items():
        signs[upper == label] = sign
    return signs


def _as_side_signs(sides_k: np.ndarray, T: int) -> np.ndarray:
    """Coerce one market's side input to a length-``T`` array of signs."""
    arr = np.asarray(sides_k)
    # Accept the adapters' string labels directly so callers can hand over a
    # `side` column untouched; numeric input is taken as signs already. Anything
    # else (e.g. a bool mask) is rejected rather than silently read as
    # "unclassified", which would quietly downgrade the whole market to proxy.
    if arr.dtype.kind in "iuf":
        signs = arr.astype(float)
    elif arr.dtype.kind in "USO":
        signs = side_labels_to_signs(arr)
    else:
        raise ValueError(f"sides entry has unsupported dtype {arr.dtype}")
    if signs.shape != (T,):
        raise ValueError(f"sides entry has shape {signs.shape}, expected ({T},)")
    return np.sign(signs)


def _buy_fractions(market: MarketData, sides_k: np.ndarray | None) -> np.ndarray:
    """Per-trade buy fraction: native aggressor side where known, else proxy.

    The proxy is bulk-volume classification, ``Phi(dY_i / sigma_dY)`` with
    ``dY_i = Y_i - Y_{i-1}``; the first trade of a market has no ``dY`` and gets
    the neutral 0.5. Trades whose side sign is 0 (unclassified) keep the proxy
    value, so a partially-labelled feed degrades gracefully.
    """
    T = market.Y.size
    buy_frac = np.full(T, 0.5, dtype=float)
    if T > 1:
        dY = np.diff(market.Y)
        std_dY = float(np.std(dY))
        if std_dY >= _MIN_STD:
            buy_frac[1:] = norm.cdf(dY / std_dY)

    if sides_k is None:
        return buy_frac

    signs = _as_side_signs(sides_k, T)
    return np.where(signs > 0.0, 1.0, np.where(signs < 0.0, 0.0, buy_frac))


def vpin_scores(
    markets: list[MarketData],
    n_buckets: int = 50,
    *,
    sides: list[np.ndarray] | None = None,
) -> np.ndarray:
    """VPIN-style order-flow toxicity score per wallet (gating signal only).

    Adapted from Easley, Lopez de Prado, and O'Hara (2012). Trade direction
    comes from the native aggressor side when ``sides`` is given, and otherwise
    from the price change ``dY_i = Y_i - Y_{i-1}`` (bulk-volume classification):
    buy fraction ``Phi(dY_i / sigma_dY)`` where ``Phi`` is the standard normal
    CDF. Prefer the native side: Andersen and Bondarenko (2014) attribute much
    of VPIN's apparent toxicity signal to the classification scheme's volume
    artifacts rather than to informed flow. Trade volume proxy is
    ``v_i = exp(log_size_ratio_i)`` (relative size).

    Per market, trades are bucketed chronologically into ``n_buckets`` equal-
    volume buckets (O(T) greedy fill). Bucket toxicity::

        VPIN_b = |V_buy_b - V_sell_b| / V_b

    where ``V_buy_b = sum_i buy_frac_i * v_i`` over trades in bucket b.

    Per-wallet score is the volume-weighted mean VPIN of buckets the wallet
    trades in::

        s_w = sum_{i: w_i=w} VPIN_{b(i)} * v_i / sum_{i: w_i=w} v_i

    Approximations (documented for reviewers):
      - Without ``sides``, direction comes from ``dY``, not true aggressor side.
      - First trade in each market gets neutral buy fraction 0.5 (no ``dY``).
      - Trades are assigned wholly to one bucket (no split at boundaries).
      - Relative ``exp(log_size_ratio)`` ranks toxicity; not USDC volume.

    Note:
        The score gates which wallets reach the Bayesian core; it is not
        evidence of informed trading (filter-only detection is a recorded GATE
        FAIL at pooled AUC 0.524). Any analysis using it must control for
        volume — see ``volume_controlled_scores``.

    Args:
        markets: Observed markets in ``MarketData`` format.
        n_buckets: Number of equal-volume buckets per market (>= 1).
        sides: Optional per-market aggressor sides aligned with each market's
            trades: either signs (``+1`` buy, ``-1`` sell, ``0`` unknown) or
            ``"BUY"``/``"SELL"`` labels. One entry per market; ``None`` uses the
            price-change proxy everywhere. Keyword-only.

    Returns:
        Array of shape ``(n_wallets,)`` with scores in [0, 1] where defined.

    Raises:
        ValueError: If ``n_buckets < 1``, or ``sides`` does not have one
            correctly-sized entry per market.
    """
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")
    if sides is not None and len(sides) != len(markets):
        raise ValueError(
            f"sides has {len(sides)} entries, expected {len(markets)} (one per market)"
        )

    n_wallets = _n_wallets_from_markets(markets)
    weighted = np.zeros(n_wallets, dtype=float)
    vol_tot = np.zeros(n_wallets, dtype=float)

    for k, market in enumerate(markets):
        T = market.Y.size
        if T == 0:
            continue

        vol = np.exp(market.log_size_ratio)
        buy_frac = _buy_fractions(market, None if sides is None else sides[k])

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


@dataclass
class VpinRobustness:
    """Native-side vs price-change-proxy VPIN, for reporting as a robustness check.

    Attributes:
        native: ``vpin_scores`` computed from the venue's aggressor side.
        proxy: ``vpin_scores`` computed from the bulk-volume price-change proxy.
        rank_correlation: Spearman correlation between the two, over wallets
            with positive traded volume; ``nan`` when fewer than two such
            wallets exist or either score is constant across them.
        n_wallets_compared: Number of wallets entering the correlation.
    """

    native: np.ndarray
    proxy: np.ndarray
    rank_correlation: float
    n_wallets_compared: int


def wallet_volumes(markets: list[MarketData]) -> np.ndarray:
    """Total relative traded volume ``sum_i exp(log_size_ratio_i)`` per wallet.

    This is the volume covariate that every VPIN-based analysis must control
    for (Andersen & Bondarenko, 2014); see ``volume_controlled_scores``.

    Args:
        markets: Observed markets in ``MarketData`` format.

    Returns:
        Array of shape ``(n_wallets,)``; wallets with no trades get 0.
    """
    volumes = np.zeros(_n_wallets_from_markets(markets), dtype=float)
    for market in markets:
        if market.wallet_ids.size:
            np.add.at(volumes, market.wallet_ids, np.exp(market.log_size_ratio))
    return volumes


def volume_controlled_scores(
    scores: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Residualize per-wallet scores on log volume (the mandatory VPIN control).

    Fits ``s_w = a + b * log(v_w) + e_w`` by ordinary least squares over wallets
    with ``v_w > 0`` and returns ``e_w``. Because OLS residuals are orthogonal
    to the regressors, whatever ranking survives in ``e_w`` is the part of the
    score that raw volume does not explain — the only part Andersen and
    Bondarenko's critique leaves usable. Log volume (not raw) is the covariate
    because trade sizes are heavy-tailed lognormal.

    Wallets with zero volume have no score to explain and are returned as 0.

    Args:
        scores: Per-wallet scores, e.g. from ``vpin_scores``.
        volumes: Per-wallet volumes from ``wallet_volumes``, same shape.

    Returns:
        Residual scores of the same shape; higher = more toxic *than its volume
        alone predicts*.

    Raises:
        ValueError: If ``scores`` and ``volumes`` have different shapes.
    """
    if scores.shape != volumes.shape:
        raise ValueError(
            f"scores shape {scores.shape} != volumes shape {volumes.shape}"
        )

    residuals = np.zeros_like(scores, dtype=float)
    mask = volumes > 0.0
    if mask.sum() < 2:
        # One (or no) wallet: the fit is saturated, so nothing is explained
        # away and the residual is degenerate at 0. Keep the zeros.
        return residuals

    log_vol = np.log(volumes[mask])
    design = np.column_stack([np.ones(log_vol.size), log_vol])
    coef, *_ = np.linalg.lstsq(design, scores[mask], rcond=None)
    residuals[mask] = scores[mask] - design @ coef
    return residuals


def vpin_robustness(
    markets: list[MarketData],
    sides: list[np.ndarray],
    *,
    n_buckets: int = 50,
) -> VpinRobustness:
    """Compare native-side VPIN against the price-change proxy on one dataset.

    The two classifications share every other input, so they are *not*
    independent evidence: the proxy result is reported as a labelled robustness
    comparison to the native-side result, which is the one that gates. A low
    rank correlation means the bulk-volume classification is reordering wallets
    relative to the truth the venue publishes — the Andersen and Bondarenko
    (2014) failure mode — and the proxy-only result should be distrusted
    wherever native sides are unavailable.

    Args:
        markets: Observed markets in ``MarketData`` format.
        sides: Per-market aggressor sides (signs or ``"BUY"``/``"SELL"``
            labels), one entry per market, as accepted by ``vpin_scores``.
        n_buckets: Number of equal-volume buckets per market (>= 1).
            Keyword-only.

    Returns:
        ``VpinRobustness`` with both score vectors and their Spearman
        correlation over wallets that actually traded.

    Raises:
        ValueError: Propagated from ``vpin_scores`` for bad ``n_buckets`` or
            mis-sized ``sides``.
    """
    native = vpin_scores(markets, n_buckets, sides=sides)
    proxy = vpin_scores(markets, n_buckets)

    traded = wallet_volumes(markets) > 0.0
    n_compared = int(traded.sum())
    rho = float("nan")
    if n_compared >= 2:
        # spearmanr returns nan for a constant input (e.g. every wallet at
        # VPIN 1.0 in a one-bucket market); that nan is the honest answer.
        rho = float(spearmanr(native[traded], proxy[traded]).statistic)

    return VpinRobustness(
        native=native,
        proxy=proxy,
        rank_correlation=rho,
        n_wallets_compared=n_compared,
    )


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

    The VPIN component uses the price-change proxy here because ``MarketData``
    carries no side field; callers holding native taker sides should score with
    ``vpin_scores(..., sides=...)`` and combine themselves. Flags are a gate,
    not a verdict — see the module docstring.

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
