"""Validation metrics for the variational-EM fast path.

General validation-metrics module for the samplerless VEM defense (plan
2026-07-23-002). This unit provides the temporal train/tail split and the
held-out one-step predictive log-likelihood scored through the ADF forward
pass; a later unit adds the PSIS-khat diagnostic (`psis_khat`) to this same
module.

Held-out predictive log-likelihood (§12): the ADF E-step already accumulates
the one-step predictive log-marginal sum_t log p(Y_t | Y_{0:t-1}) for a market
(that quantity is `variational_em._vem_e_step`'s returned `log_marginal`).
Because that forward pass is *causal* — log p(Y_t | Y_{0:t-1}) depends only on
observations up to t — the head-portion terms are bit-identical whether or not
the held-out tail is appended, and are summed in the same order. Differencing
the full-market marginal against the head-only marginal therefore cancels the
head terms exactly and leaves precisely the sum of the tail's one-step
predictive densities. This reuses the fitted E-step verbatim rather than
reimplementing the filter mixture over the (V, Z) branches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.inference.particle_gibbs import MarketData
from src.inference.variational_em import VEMOutput, _vem_e_step


# ---------------- Temporal train/tail split ----------------


def holdout_split(
    markets: list[MarketData], h: float = 0.2
) -> tuple[list[MarketData], list[MarketData]]:
    """Split each market into a temporal training head and a held-out tail.

    The split preserves trade order (no shuffling — leakage discipline, KTD5):
    the head is the first `(1 - h)` fraction of trades and the tail is the
    final `h` fraction. The tail is scored as a *continuation* of the head, so
    its first `delta` keeps the true inter-trade gap from the last training
    trade rather than being reset to the `delta[0] == 0` fresh-market sentinel.
    Slicing the original `delta` array already yields that gap — `delta[n_head]`
    is by definition the time from trade `n_head - 1` (last training) to trade
    `n_head` (first held-out) — so no explicit recomputation is needed.

    Args:
        markets: Full markets to split.
        h: Fraction of each market's trades held out as the tail, in [0, 1].
            `h = 0` yields an empty tail (handled by the scorer); the default
            0.2 holds out the last 20% of trades.

    Returns:
        `(heads, tails)`: two lists of `MarketData`, aligned with `markets`.
        Each head + tail partitions its source market and concatenates back to
        it exactly. A tail may be empty (length 0) when `h` rounds to no trades.
    """
    heads: list[MarketData] = []
    tails: list[MarketData] = []
    for md in markets:
        T = len(md.Y)
        # Round to the nearest whole trade, then clamp so the tail never
        # exceeds the market (n_head stays >= 0).
        n_tail = min(int(round(h * T)), T)
        n_head = T - n_tail
        heads.append(
            MarketData(
                Y=md.Y[:n_head],
                delta=md.delta[:n_head],
                log_size_ratio=md.log_size_ratio[:n_head],
                wallet_ids=md.wallet_ids[:n_head],
            )
        )
        tails.append(
            MarketData(
                Y=md.Y[n_head:],
                delta=md.delta[n_head:],
                log_size_ratio=md.log_size_ratio[n_head:],
                wallet_ids=md.wallet_ids[n_head:],
            )
        )
    return heads, tails


# ---------------- Held-out one-step predictive log-likelihood ----------------


@dataclass
class HeldoutLL:
    """One market's held-out one-step predictive log-likelihood.

    Attributes:
        total: Sum over the held-out tail of the one-step predictive log
            densities log p(Y_t | Y_{0:t-1}) under the fitted ADF forward pass,
            conditioned on the training head.
        n_tail: Number of held-out trades scored.
        mean: Per-held-out-trade mean (`total / n_tail`); `nan` when
            `n_tail == 0`.
    """

    total: float
    n_tail: int
    mean: float


@dataclass
class HeldoutSummary:
    """Pooled and per-market held-out predictive log-likelihood (R4).

    Attributes:
        pooled_total: Sum of `total` over all markets (all held-out trades).
        pooled_n: Total number of held-out trades across all markets.
        pooled_mean: `pooled_total / pooled_n`; `nan` when no trades held out.
        per_market: Per-market `HeldoutLL`, aligned with the input markets.
    """

    pooled_total: float
    pooled_n: int
    pooled_mean: float
    per_market: list[HeldoutLL]


def _adf_log_marginal(md: MarketData, vem_output: VEMOutput) -> float:
    """Total ADF predictive log-marginal sum_t log p(Y_t | Y_{0:t-1}).

    Runs the fitted E-step forward pass over one market at the VEMOutput's
    parameters and standardization constants, returning only its accumulated
    one-step predictive log-marginal. An empty market contributes 0.0.
    """
    if len(md.Y) == 0:
        return 0.0
    _, _, _, log_marginal = _vem_e_step(
        md.Y,
        md.delta,
        md.log_size_ratio,
        md.wallet_ids,
        vem_output.theta_w,
        vem_output.params,
        vem_output.m_S,
        vem_output.s_S,
        vem_output.m_Z,
    )
    return float(log_marginal)


def _concat_markets(head: MarketData, tail: MarketData) -> MarketData:
    """Concatenate a head/tail split back into the contiguous full market."""
    return MarketData(
        Y=np.concatenate([head.Y, tail.Y]),
        delta=np.concatenate([head.delta, tail.delta]),
        log_size_ratio=np.concatenate([head.log_size_ratio, tail.log_size_ratio]),
        wallet_ids=np.concatenate([head.wallet_ids, tail.wallet_ids]),
    )


def heldout_predictive_ll(
    vem_output: VEMOutput, head: MarketData, tail: MarketData
) -> HeldoutLL:
    """Score one market's held-out tail by ADF one-step predictive densities.

    Accumulates sum_t log p(Y_t | Y_{0:t-1}) over the held-out tail, where the
    predictor is conditioned on the training head via the fitted ADF forward
    pass (the causal filter's state at the head boundary carries into the
    tail). The tail score is obtained as `log_marginal(head + tail) -
    log_marginal(head)`: the E-step's forward pass is causal, so the head-term
    partial sums are bit-identical in both passes and cancel exactly under the
    difference, leaving precisely the tail's one-step predictive densities
    (see the module docstring). This reuses `_vem_e_step` verbatim instead of
    reimplementing the mixture-over-(V, Z) filter math.

    `head` and `tail` must be a contiguous temporal split of one market — the
    output of `holdout_split` — so that concatenating them reconstructs the
    original market (in particular, `tail`'s first `delta` is the true gap from
    the last training trade, per KTD5).

    Args:
        vem_output: Fitted VEM output supplying the parameters, per-wallet
            propensities `theta_w`, and standardization constants
            `(m_S, s_S, m_Z)` that define the frozen ADF forward pass.
        head: Training head market (may be empty in degenerate splits).
        tail: Held-out tail market; an empty tail yields a zero-count result.

    Returns:
        The market's `HeldoutLL` (total tail log predictive density, held-out
        trade count, and per-trade mean).
    """
    n_tail = len(tail.Y)
    if n_tail == 0:
        return HeldoutLL(total=0.0, n_tail=0, mean=float("nan"))
    full = _concat_markets(head, tail)
    tail_total = _adf_log_marginal(full, vem_output) - _adf_log_marginal(
        head, vem_output
    )
    return HeldoutLL(total=tail_total, n_tail=n_tail, mean=tail_total / n_tail)


def heldout_predictive_summary(
    vem_output: VEMOutput,
    heads: list[MarketData],
    tails: list[MarketData],
) -> HeldoutSummary:
    """Pool held-out predictive log-likelihood over markets (R4).

    Scores each `(head, tail)` pair with `heldout_predictive_ll` and reports
    both the per-market values and the pooled aggregate over every held-out
    trade. Pooling by summed total (and dividing by the total held-out trade
    count for the mean) weights markets by their tail length rather than
    treating every market equally.

    Args:
        vem_output: Fitted VEM output defining the frozen ADF forward pass.
        heads: Training heads, aligned with `tails` (from `holdout_split`).
        tails: Held-out tails, aligned with `heads`.

    Returns:
        A `HeldoutSummary` with pooled total/count/mean and the per-market list.
    """
    per_market = [
        heldout_predictive_ll(vem_output, head, tail)
        for head, tail in zip(heads, tails)
    ]
    pooled_total = float(sum(m.total for m in per_market))
    pooled_n = int(sum(m.n_tail for m in per_market))
    pooled_mean = pooled_total / pooled_n if pooled_n > 0 else float("nan")
    return HeldoutSummary(
        pooled_total=pooled_total,
        pooled_n=pooled_n,
        pooled_mean=pooled_mean,
        per_market=per_market,
    )
