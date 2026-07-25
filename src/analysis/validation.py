"""Validation metrics for the variational-EM fast path.

General validation-metrics module for the samplerless VEM defense (plan
2026-07-23-002): the temporal train/tail split with the held-out one-step
predictive log-likelihood scored through the ADF forward pass, the multi-restart
stability metrics (R6), and the PSIS-khat importance-sampling diagnostic over the
parameter vector phi. `scripts/validate_vem.py` is a thin CLI over these
functions (KTD4), so downstream plans import them from here rather than reaching
into `scripts/`.

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

PSIS-khat (R5): draws phi_s from the Laplace posterior q and Pareto-smooths the
importance ratios against the ADF-implied *conditional* parameter posterior
`p(phi | Y, theta_w_hat)`,

    log w_s = log p(Y | phi_s, theta_w_hat) + log p(phi_s) - log q(phi_s),

where the likelihood term is the same ADF forward-pass log-marginal used above
(evaluated with `theta_w` pinned at its fitted value for every draw, so nothing
here integrates over wallet propensities), `p` is the model's own `PhiPrior`, and
`q` is `PhiPosterior`. Both densities are evaluated on the *constrained* scale:
`PhiPrior.log_prior` is natively constrained and `PhiPosterior.logpdf` adds the
change-of-variables Jacobian of the unconstrained reparameterization, so the
ratio compares two densities of the same variable (a missing Jacobian would
silently tilt the target by `|du/dphi|`). See the `psis_khat` docstring for the
scope of the resulting claim, and `phi_centring_gradient` for the precondition
that has to hold before khat says anything about the variational family at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from config.default_params import ModelParams, PhiPrior
from src.analysis.results import count_wallet_trades, evaluate_synthetic_gate
from src.data.preprocess import WalletIndex
from src.inference.laplace import PhiPosterior
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


# ---------------- Multi-restart stability (R6) ----------------

# Lognormal sd applied to each warm-start variance when perturbing a restart's
# start point. Large enough that restarts explore genuinely different basins,
# small enough to keep the regime ordering the warm start encodes
# (sigma2_0 = 0.1*var_Y vs sigma2_1 = var_Y are a decade apart, so a 0.1 log-sd
# jitter cannot swap them and hand VEM a label-switched initialization).
INIT_JITTER_LOG_SD = 0.1

# Variance fields perturbed per restart, in the order the jitter draw indexes.
_JITTERED_VARIANCES = ("sigma2_0", "sigma2_1", "tau2_0", "tau2_1")

# Pooled-AUC spread above which restart disagreement is escalated rather than
# merely recorded. 0.05 is the bar `tests/test_variational_em.py`'s multi-seed
# stability test already applies to exactly this quantity, so the two protocols
# are read on one scale — note they probe *different* sensitivities, see
# `stability_block`.
AUC_SPREAD_THRESHOLD = 0.05


def jittered_init(
    markets: list[MarketData],
    n_wallets: int,
    rng: np.random.Generator,
    jitter_log_sd: float = INIT_JITTER_LOG_SD,
) -> tuple[ModelParams, np.ndarray]:
    """Draw one restart's VEM start point around the deterministic warm start.

    VEM is a deterministic map from (data, initialization) to a fit, so a
    stability check has to move the initialization: the warm-start variances are
    scaled by ``exp(N(0, jitter_log_sd))`` (multiplicative, keeping every
    variance positive) and ``theta_w`` is drawn from its own Beta(a, b) prior
    instead of being pinned at the prior mean.

    Args:
        markets: Markets the restart will be fit on; supplies the warm start.
        n_wallets: Length of the ``theta_w`` vector to draw.
        rng: Restart-seeded generator (never the global RNG).
        jitter_log_sd: Lognormal sd of the multiplicative variance jitter.

    Returns:
        The ``(params_init, theta_w_init)`` pair for this restart.
    """
    Y_concat = np.concatenate([md.Y for md in markets])
    params = ModelParams.warm_start(Y_concat)
    jitter = np.exp(rng.normal(0.0, jitter_log_sd, size=len(_JITTERED_VARIANCES)))
    params = replace(
        params,
        **{
            name: float(getattr(params, name) * factor)
            for name, factor in zip(_JITTERED_VARIANCES, jitter)
        },
    )
    theta_w = rng.beta(params.a, params.b, size=n_wallets)
    return params, theta_w


def top_k_wallets(theta_w: np.ndarray, k: int) -> list[int]:
    """Return the k highest-scoring wallet ids, ties broken by ascending id.

    Args:
        theta_w: Per-wallet propensity scores, shape ``(n_wallets,)``.
        k: Requested set size; clamped to ``[1, n_wallets]``.

    Returns:
        The selected wallet ids, sorted ascending (a set, not a ranking). Ties
        at the cutoff resolve to the lower id: the sort is stable (mergesort)
        over descending scores, so equal scores keep their id order.
    """
    scores = np.asarray(theta_w, dtype=float)
    k_eff = min(max(1, k), scores.size)
    order = np.argsort(-scores, kind="mergesort")
    return sorted(int(w) for w in order[:k_eff])


def pooled_synthetic_auc(
    vem_output: VEMOutput,
    markets: list[MarketData],
    market_objs: list[Any],
    wallet_index: WalletIndex,
) -> float:
    """Pooled synthetic ROC AUC of one fit's per-trade insider scores.

    Delegates to the same ``evaluate_synthetic_gate`` path the benchmark gate
    uses, so a restart's AUC is directly comparable with the recorded gate
    numbers. Requires synthetic markets — real data carries no insider ground
    truth.

    Args:
        vem_output: Fitted VEM output supplying ``Z_prob`` and ``theta_w``.
        markets: The ``MarketData`` the fit consumed; supplies wallet ids for
            the trade-count annotation.
        market_objs: Aligned ``SyntheticMarket`` objects carrying ground truth.
        wallet_index: Wallet index used for the ranking table labels.

    Returns:
        The pooled ROC AUC over all trades of all synthetic markets.
    """
    n_trades = count_wallet_trades(
        [md.wallet_ids for md in markets],
        n_wallets=wallet_index.n_wallets,
    )
    gate = evaluate_synthetic_gate(
        vem_output.Z_prob,
        np.asarray(vem_output.theta_w, dtype=float),
        market_objs,
        wallet_index,
        n_trades_per_wallet=n_trades,
    )
    return float(gate["pooled_auc"])


def elbo_convergence(
    elbo_trace: np.ndarray,
    n_iter_run: int,
    *,
    n_iter_max: int,
    tol: float,
) -> dict[str, Any]:
    """Convergence verdict for one restart, from its ELBO trace alone.

    Reproduces ``variational_em``'s own stopping rule — relative change
    ``|L_last - L_prev| / max(|L_prev|, 1)`` compared against ``tol`` — so the
    recorded verdict is the same one the fit would have acted on rather than a
    second, differently-scaled criterion. Hitting the iteration cap is reported
    separately: a run can stop at the cap with a relative change still orders of
    magnitude above ``tol``, which is the pre-convergence case a reader must not
    mistake for a converged mode.

    Args:
        elbo_trace: Per-iteration ADF log-marginal, shape ``(n_iter_run,)``.
        n_iter_run: EM iterations actually completed.
        n_iter_max: The iteration cap the fit was given.
        tol: The relative-change tolerance the fit was given.

    Returns:
        Dict with ``converged``, ``hit_iter_cap``, ``final_rel_elbo_change`` and
        ``final_elbo_gain`` (the last iteration's raw ELBO increase). A trace
        shorter than two points leaves the change metrics NaN and
        ``converged`` False — nothing was measured.
    """
    trace = np.asarray(elbo_trace, dtype=float)
    hit_iter_cap = bool(int(n_iter_run) >= int(n_iter_max))
    if trace.size < 2:
        return {
            "converged": False,
            "hit_iter_cap": hit_iter_cap,
            "final_rel_elbo_change": float("nan"),
            "final_elbo_gain": float("nan"),
        }
    gain = float(trace[-1] - trace[-2])
    rel_change = abs(gain) / max(abs(float(trace[-2])), 1.0)
    return {
        "converged": bool(rel_change < tol),
        "hit_iter_cap": hit_iter_cap,
        "final_rel_elbo_change": rel_change,
        "final_elbo_gain": gain,
    }


def restart_record(
    vem_output: VEMOutput,
    *,
    seed: int,
    top_k: int,
    pooled_auc: float | None,
    n_iter_max: int,
    tol: float,
) -> dict[str, Any]:
    """Summarize one restart into its JSON record.

    Args:
        vem_output: The restart's fit.
        seed: Restart seed (drives the initialization draw only).
        top_k: Size of the suspicious-wallet set recorded for the Jaccard
            stability metric.
        pooled_auc: Pooled synthetic AUC, or None when there is no ground truth.
        n_iter_max: Iteration cap the restart was given (for the cap flag).
        tol: Relative-ELBO tolerance the restart was given.

    Returns:
        The restart's record, including its convergence verdict so a reader
        never has to re-derive whether the number is a converged quantity.
    """
    trace = np.asarray(vem_output.elbo_trace, dtype=float)
    record: dict[str, Any] = {
        "seed": seed,
        "terminal_elbo": float(trace[-1]) if trace.size else float("nan"),
        "n_iter_run": int(vem_output.n_iter_run),
        "pooled_auc": pooled_auc,
        "beta_S_orig": float(vem_output.beta_S_orig),
        "beta_Z_orig": float(vem_output.beta_Z_orig),
        "top_k_wallets": top_k_wallets(vem_output.theta_w, top_k),
        "elbo_trace": [float(v) for v in trace],
    }
    record.update(
        elbo_convergence(
            trace,
            vem_output.n_iter_run,
            n_iter_max=n_iter_max,
            tol=tol,
        )
    )
    return record


def spread(values: list[float | None]) -> dict[str, float]:
    """Min/max/mean/sd and max-min spread of a restart metric.

    ``None`` entries (a metric undefined for this run, e.g. AUC on real data)
    are dropped; an all-``None`` metric yields NaNs rather than raising.

    Args:
        values: One value per restart, possibly containing None.

    Returns:
        Dict with ``min``, ``max``, ``mean``, ``sd`` and ``spread`` keys.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        nan = float("nan")
        return {"min": nan, "max": nan, "mean": nan, "sd": nan, "spread": nan}
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        # Population sd (ddof=0): the restarts are the whole ensemble being
        # described, not a sample drawn from a larger population.
        "sd": float(arr.std()),
        "spread": float(arr.max() - arr.min()),
    }


def mean_pairwise_jaccard(wallet_sets: list[list[int]]) -> float:
    """Mean Jaccard overlap over all unordered pairs of top-K wallet sets.

    A value of 1.0 means every restart flagged exactly the same wallets — the
    stability claim the ranking rests on. Fewer than two restarts leaves the
    metric undefined (NaN); two empty sets count as identical (1.0) rather than
    dividing by zero.

    Args:
        wallet_sets: One top-K wallet id list per restart.

    Returns:
        Mean pairwise ``|A & B| / |A | B|``, or NaN with fewer than two sets.
    """
    if len(wallet_sets) < 2:
        return float("nan")
    sets = [set(s) for s in wallet_sets]
    overlaps = [
        len(a & b) / len(a | b) if (a | b) else 1.0
        for i, a in enumerate(sets)
        for b in sets[i + 1 :]
    ]
    return float(np.mean(overlaps))


def stability_block(
    records: list[dict[str, Any]],
    *,
    top_k: int,
    auc_spread_threshold: float = AUC_SPREAD_THRESHOLD,
) -> dict[str, Any]:
    """Aggregate per-restart records into the R6 stability summary.

    Escalates (rather than silently records) a pooled-AUC spread above
    ``auc_spread_threshold``: restarts here share one fixed dataset and differ
    only in their jittered start point, so a wide AUC spread is *initialization*
    sensitivity, not data-seed sensitivity. The two are separate findings and
    must not be conflated — the deterministic unjittered warm start that
    `scripts/benchmark.py` uses is stable across data seeds while these
    jittered restarts are not.

    Args:
        records: Per-restart records from `restart_record`.
        top_k: Wallet-set size the Jaccard metric was computed at.
        auc_spread_threshold: Pooled-AUC max-min spread above which the
            ``pooled_auc_unstable`` flag and its warning fire.

    Returns:
        The stability block: per-metric spreads, the mean pairwise top-K
        Jaccard, the threshold used, a boolean ``pooled_auc_unstable`` flag and
        a ``warnings`` list of reader-facing escalation strings (empty when
        nothing escalates).
    """
    auc_spread_stats = spread([r["pooled_auc"] for r in records])
    auc_spread = auc_spread_stats["spread"]
    unstable = bool(np.isfinite(auc_spread) and auc_spread > auc_spread_threshold)
    warnings_out: list[str] = []
    if unstable:
        warnings_out.append(
            f"UNSTABLE: pooled-AUC spread across restarts is {auc_spread:.3f} "
            f"(> {auc_spread_threshold:.2f}), ranging "
            f"{auc_spread_stats['min']:.3f}-{auc_spread_stats['max']:.3f}. "
            "Restarts share one FIXED dataset and differ only in their jittered "
            f"start point (init_jitter_log_sd={INIT_JITTER_LOG_SD}), so this is "
            "INITIALIZATION sensitivity, NOT data-seed sensitivity: the "
            "deterministic unjittered warm start used by scripts/benchmark.py "
            "is separately stable across data seeds. Both statements are true; "
            "do not report one as the other."
        )
    return {
        "n_restarts": len(records),
        "top_k": top_k,
        "terminal_elbo": spread([r["terminal_elbo"] for r in records]),
        "pooled_auc": auc_spread_stats,
        "beta_S_orig": spread([r["beta_S_orig"] for r in records]),
        "beta_Z_orig": spread([r["beta_Z_orig"] for r in records]),
        "mean_pairwise_topk_jaccard": mean_pairwise_jaccard(
            [r["top_k_wallets"] for r in records],
        ),
        "pooled_auc_spread_threshold": float(auc_spread_threshold),
        "pooled_auc_unstable": unstable,
        "warnings": warnings_out,
    }


def convergence_block(
    records: list[dict[str, Any]],
    *,
    n_iter_max: int,
    tol: float,
) -> dict[str, Any]:
    """Ensemble convergence status and the best-restart selection guard.

    Two things a reader of the artifact must not have to reconstruct:

    1. Whether the fits converged at all. Restarts that stop at the iteration
       cap with a final relative ELBO change above ``tol`` are pre-convergence
       trajectory snapshots; every downstream number (AUC, held-out LL, khat)
       inherits that status.
    2. Whether ``argmax terminal_elbo`` picked a better mode or just the
       trajectory that happens to be furthest along. If the across-restart
       terminal-ELBO spread is smaller than the median final per-iteration ELBO
       gain, one more iteration of any restart would reorder the ranking, so the
       selection carries no information about modes.

    Args:
        records: Per-restart records from `restart_record`.
        n_iter_max: Iteration cap the restarts were given.
        tol: Relative-ELBO tolerance the restarts were given.

    Returns:
        The convergence block, including a ``warnings`` list of reader-facing
        strings (empty when the ensemble converged and the selection is sound).
    """
    n = len(records)
    n_at_cap = sum(1 for r in records if r["hit_iter_cap"])
    rel_stats = spread([r["final_rel_elbo_change"] for r in records])
    gain_stats = spread([r["final_elbo_gain"] for r in records])
    elbo_stats = spread([r["terminal_elbo"] for r in records])
    gains = np.asarray([r["final_elbo_gain"] for r in records], dtype=float)
    median_gain = float(np.median(np.abs(gains))) if gains.size else float("nan")
    elbo_spread = elbo_stats["spread"]
    selection_meaningful = bool(
        n < 2
        or not (np.isfinite(elbo_spread) and np.isfinite(median_gain))
        or elbo_spread >= median_gain
    )
    converged = bool(n > 0 and all(r["converged"] for r in records))

    warnings_out: list[str] = []
    if n_at_cap:
        warnings_out.append(
            f"PRE-CONVERGENCE: {n_at_cap} of {n} restarts stopped at the "
            f"{n_iter_max}-iteration cap; the final relative ELBO change reached "
            f"{rel_stats['max']:.3g} against tol={tol:g} and the ELBO was still "
            f"climbing at ~{median_gain:.3g} nats/iteration. Terminal ELBO, "
            "pooled AUC, held-out LL and khat are trajectory snapshots, not "
            "converged-mode quantities."
        )
    if not selection_meaningful:
        warnings_out.append(
            f"BEST-RESTART SELECTION NOT MEANINGFUL: the across-restart "
            f"terminal-ELBO spread ({elbo_spread:.4g}) is smaller than the "
            f"median final per-iteration ELBO gain ({median_gain:.4g}), so "
            "argmax(terminal_elbo) selects whichever trajectory is furthest "
            "along, not a better mode. Do not read the chosen restart as the "
            "highest-ELBO mode."
        )
    return {
        "converged": converged,
        "n_restarts": n,
        "n_restarts_converged": sum(1 for r in records if r["converged"]),
        "n_restarts_at_iter_cap": n_at_cap,
        "vem_iters": int(n_iter_max),
        "vem_tol": float(tol),
        "final_rel_elbo_change": rel_stats,
        "final_elbo_gain": gain_stats,
        "median_final_elbo_gain": median_gain,
        "terminal_elbo_spread": elbo_spread,
        "best_restart_selection_meaningful": selection_meaningful,
        "warnings": warnings_out,
    }


# ---------------- PSIS-khat diagnostic ----------------

# Result key for the CLI's JSON artifact. The name states which two objects the
# diagnostic compares (the Laplace posterior against the ADF surface) so it can
# never be read as a check of ADF itself. Deliberately unchanged even though the
# target is now named precisely as a *conditional* posterior in
# `PSIS_SCOPE_NOTE`: renaming the key would break comparability with the
# committed artifacts — see R5 and the `psis_khat` docstring.
PSIS_KHAT_KEY = "psis_khat_laplace_vs_adf"

PSIS_SCOPE_NOTE = (
    "The target is the ADF-implied CONDITIONAL posterior p(phi | Y, "
    "theta_w_hat): theta_w is pinned at its fitted value for every draw, so "
    "nothing here integrates over wallet propensities and the diagnostic says "
    "nothing about theta_w uncertainty. Within that conditional, khat measures "
    "Laplace-shape adequacy for the ADF surface, not ADF's fidelity to the true "
    "posterior: q and the target are built on the same ADF surface, so any bias "
    "ADF itself carries is invisible here. khat is a statement about the "
    "variational family only once the proposal is centred at a mode of that "
    "conditional target — check `phi_centring_gradient` first, since a "
    "mis-centred proposal gives the same khat however rich the family is. A "
    "good khat is expected and is a necessary-not-sufficient check; "
    "simulation-based calibration (plan 2026-07-23-003) is the actual "
    "faithfulness test."
)

# Canonical constrained-parameter order this module assembles ModelParams from.
# Must agree with `PhiPosterior.dims`; `psis_khat` asserts it per call rather
# than importing laplace's private tuple, so a reordering there fails loudly.
_PHI_DIMS = (
    "sigma2_0",
    "sigma2_1",
    "q_01",
    "q_10",
    "beta_S",
    "beta_Z",
    "tau2_0",
    "tau2_1",
)

# Positions within `_PHI_DIMS` of the log-transformed (variance) and
# logit-transformed (transition) dimensions. Derived from the names rather than
# written out, so a reordering of `_PHI_DIMS` cannot silently mis-place the
# change-of-variables term in `_log_abs_dphi_du`.
_VAR_IDX = tuple(
    _PHI_DIMS.index(name) for name in ("sigma2_0", "sigma2_1", "tau2_0", "tau2_1")
)
_Q_IDX = tuple(_PHI_DIMS.index(name) for name in ("q_01", "q_10"))

# PSIS fits a generalized Pareto to the largest ~n/5 weights and needs at least
# 5 of them, so fewer than 25 draws makes arviz raise from deep inside its tail
# routine. Checked up front so the caller gets a message naming the knob.
_PSIS_MIN_DRAWS = 25


def khat_interpretation(khat: float) -> str:
    """Standard Yao et al. (2018) reading of a Pareto khat value.

    Args:
        khat: Estimated shape parameter of the generalized-Pareto fit to the
            importance-weight tail.

    Returns:
        A short human-readable band label ("good"/"ok"/"bad"/"undefined")
        followed by its meaning, for the CLI artifact and report text.
    """
    if not np.isfinite(khat):
        return (
            "undefined (khat is not finite): the weight tail could not be fit — "
            "check for degenerate or constant log-weights."
        )
    if khat < 0.5:
        return (
            "good (khat < 0.5): the importance-weight variance is finite and "
            "the Laplace proposal covers the target well."
        )
    if khat <= 0.7:
        return (
            "ok (0.5 <= khat <= 0.7): weights are heavy-tailed but PSIS "
            "smoothing remains reliable; convergence is slower."
        )
    return (
        "bad (khat > 0.7): the importance weights have unbounded variance — the "
        "Laplace proposal does not cover this target. Read that as a statement "
        "about the proposal, not about the richness of the variational family: "
        "khat is only a family-adequacy measure once the proposal is centred at "
        "a mode of its own target, because a richer family centred at the same "
        "non-mode with the same curvature scores the same khat. Read "
        "`phi_centring_gradient` before concluding anything about the family."
    )


def _psislw(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Pareto-smooth log importance weights, returning `(smoothed, khat)`.

    Implements Vehtari et al. (2024): normalize the log weights by their max,
    fit a generalized Pareto to the largest `n_tail` weights (on the natural
    scale), and replace that tail by the fitted quantiles. `khat` is the fitted
    shape parameter.

    arviz split its stats into `arviz-stats` at 1.0 and `arviz.psislw`
    disappeared, so both layouts are handled. The 1.x branch calls the array
    backend's tail routine directly instead of its two public wrappers, because
    as of arviz-stats 1.1.0 both wrappers are wrong and would silently invert
    the diagnostic:
      * `psislw` passes the *negated* log weights with `tail="right"`, fitting
        the smallest weights instead of the largest;
      * `pareto_khat` forgets to forward `log_weights=True`, fitting the Pareto
        to log weights rather than weights.
    Both return khat near 0 for reference samples with known khat of 0.7-1.0
    (verified against exact generalized-Pareto draws and against the analytic
    `1 - 1/S**2` value for a N(0, S) target under a N(0, 1) proposal); the call
    below reproduces those references. Revisit if upstream fixes them.

    The imports are at use site rather than module scope so the rest of this
    module still imports where arviz is absent.

    Args:
        log_weights: Raw log importance ratios, shape `(n_draws,)`.

    Returns:
        `(smoothed_log_weights, khat)`. Relative efficiency is 1 because the
        draws are i.i.d. from `q`, not an MCMC chain.
    """
    log_weights = np.asarray(log_weights, dtype=float)
    n_draws = log_weights.size
    try:
        from arviz_stats.base import array_stats
    except ImportError:  # pragma: no cover - arviz < 1.0 layout
        from arviz import psislw as _legacy_psislw

        smoothed, khat = _legacy_psislw(log_weights, reff=1.0)
        return np.asarray(smoothed, dtype=float), float(khat)

    n_tail = int(array_stats._get_ps_tails(n_draws, 1.0, tail="right"))
    smoothed, khat = array_stats._ps_tail(
        log_weights,
        n_draws,
        n_tail,
        smooth_draws=True,
        tail="right",
        log_weights=True,
    )
    return np.asarray(smoothed, dtype=float), float(khat)


@dataclass
class PSISResult:
    """Pareto-smoothed importance-sampling diagnostic for the Laplace posterior.

    Attributes:
        khat: Estimated generalized-Pareto tail shape of the importance weights.
        n_draws: Number of posterior draws scored.
        log_weights: Raw (unsmoothed) log importance ratios, shape `(n_draws,)`.
        log_weights_smoothed: PSIS-smoothed log weights, same shape, normalized
            to sum to one on the natural scale.
        interpretation: `khat_interpretation(khat)`, carried so a report never
            has to re-derive the band.
    """

    khat: float
    n_draws: int
    log_weights: np.ndarray
    log_weights_smoothed: np.ndarray
    interpretation: str

    def to_dict(self) -> dict[str, float | int | str]:
        """Serializable summary for the validation CLI's JSON artifact.

        Returns:
            The scalar fields keyed for the artifact — `PSIS_KHAT_KEY` for khat
            itself, plus the draw count, band interpretation, and the
            scope-of-claim caveat. The log-weight arrays are omitted (they are a
            diagnostic detail, not a headline number).
        """
        return {
            PSIS_KHAT_KEY: self.khat,
            "psis_n_draws": self.n_draws,
            "psis_khat_interpretation": self.interpretation,
            "psis_scope_note": PSIS_SCOPE_NOTE,
        }


def _params_from_phi(base: ModelParams, phi: np.ndarray) -> ModelParams:
    """Overlay one constrained phi draw onto the fitted parameter object.

    Only the eight `_PHI_DIMS` entries move; the fixed hyperparameters the
    Laplace layer does not model (`gamma`, `s0_2`, the theta_w Beta prior) are
    carried over from `base` so each draw is scored under the same model.
    """
    return replace(
        base,
        sigma2_0=float(phi[0]),
        sigma2_1=float(phi[1]),
        q_01=float(phi[2]),
        q_10=float(phi[3]),
        beta_S=float(phi[4]),
        beta_Z=float(phi[5]),
        tau2_0=float(phi[6]),
        tau2_1=float(phi[7]),
    )


def _adf_log_lik(
    vem_output: VEMOutput, markets: list[MarketData], phi: np.ndarray
) -> float:
    """Total ADF log-marginal `log p(Y | phi)` over all markets for one draw.

    Runs the batch E-step (a forward pass only — no EM iteration) per market at
    the draw's parameters, holding `theta_w` and the standardization constants
    at their fitted values, and sums the per-market log-marginals. Module-level
    (not a closure) so joblib can ship it to process workers.
    """
    draw_output = replace(vem_output, params=_params_from_phi(vem_output.params, phi))
    return float(sum(_adf_log_marginal(md, draw_output) for md in markets))


def psis_khat(
    vem_output: VEMOutput,
    phi_posterior: PhiPosterior,
    markets: list[MarketData],
    rng: np.random.Generator,
    n_draws: int = 1000,
    n_jobs: int = 1,
    prior: PhiPrior | None = None,
) -> PSISResult:
    """PSIS-khat between the Laplace posterior and the conditional ADF target.

    Draws `phi_s ~ q` from the Laplace posterior and Pareto-smooths the
    importance ratios against the ADF-implied *conditional* parameter posterior
    `p(phi | Y, theta_w_hat)`:

        log w_s = log p(Y | phi_s, theta_w_hat) + log p(phi_s) - log q(phi_s)

    with `log p(Y | phi_s, theta_w_hat)` the ADF forward-pass log-marginal summed
    over markets at the fitted `theta_w`, `log p(phi_s)` the model's own prior
    (`PhiPrior`, the same spec the M-step optimizes against), and `log q(phi_s)`
    the Laplace density. Both density terms are evaluated on the **constrained**
    scale — `log_prior` is natively constrained and `PhiPosterior.logpdf` adds
    the unconstrained -> constrained Jacobian — so the ratio compares densities
    of the same variable.

    **Scope of the claim.** `theta_w` is pinned for every draw, so nothing here
    integrates over wallet propensities and the diagnostic says nothing about
    `theta_w` uncertainty. Within that conditional, because `q` and the target
    are both built on the ADF surface, khat measures *Laplace-shape adequacy for
    the ADF surface* and cannot detect ADF's own bias relative to the true
    posterior: a good khat is expected, and is a necessary-not-sufficient check.
    Simulation-based calibration (plan 2026-07-23-003) is the actual faithfulness
    test. Report the value under `PSIS_KHAT_KEY` alongside `PSIS_SCOPE_NOTE`, and
    read it with the standard bands (< 0.5 good, 0.5-0.7 ok, > 0.7 bad). A khat
    > 0.7 indicts the *proposal*; it becomes a statement about the variational
    family only once `phi_centring_gradient` shows the proposal is centred at a
    mode of this conditional target. It is not a test assertion.

    Args:
        vem_output: Fitted VEM output supplying `theta_w`, the standardization
            constants, and the base `ModelParams` each draw is overlaid on.
        phi_posterior: Laplace posterior over phi (`laplace_from_vem`), the
            proposal `q`.
        markets: Markets the VEM was fit on. An empty list makes the likelihood
            term identically zero, so the target reduces to the prior — used by
            the tests as an analytically known control.
        rng: Explicit generator for the `q` draws (never the global RNG). All
            `n_draws` are sampled up front in one call, so the scoring loop is a
            deterministic map and results do not depend on `n_jobs`.
        n_draws: Number of posterior draws (one ADF pass per market per draw);
            at least `_PSIS_MIN_DRAWS`, below which the Pareto tail fit has too
            few points to be defined.
        n_jobs: joblib workers over draws (embarrassingly parallel); 1 runs
            in-process and is bit-exact with any other value.
        prior: The prior spec VEM was fit with; `None` uses `PhiPrior()`
            defaults. Must match, or the ratio is taken against a density the
            estimator never used.

    Returns:
        A `PSISResult` with khat, the raw and smoothed log-weights, and the band
        interpretation.

    Raises:
        ValueError: If `phi_posterior.dims` is not this module's canonical
            parameter order, or if `n_draws` is below `_PSIS_MIN_DRAWS`.
    """
    if tuple(phi_posterior.dims) != _PHI_DIMS:
        raise ValueError(
            f"phi_posterior.dims {tuple(phi_posterior.dims)} does not match the "
            f"canonical parameter order {_PHI_DIMS}"
        )
    if n_draws < _PSIS_MIN_DRAWS:
        raise ValueError(
            f"n_draws must be at least {_PSIS_MIN_DRAWS} for the PSIS tail fit, "
            f"got {n_draws}"
        )
    if prior is None:
        prior = PhiPrior()

    draws = phi_posterior.sample(rng, n_draws)
    if markets:
        log_lik = np.asarray(
            Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_adf_log_lik)(vem_output, markets, phi) for phi in draws
            ),
            dtype=float,
        )
    else:
        # No data: log p(Y | phi) is empty-sum zero for every draw. Short-circuit
        # so the likelihood-free controls do not pay for an empty joblib loop.
        log_lik = np.zeros(n_draws)

    log_weights = log_lik + prior.log_prior(draws) - phi_posterior.logpdf(draws)
    smoothed, khat = _psislw(log_weights)
    return PSISResult(
        khat=khat,
        n_draws=n_draws,
        log_weights=log_weights,
        log_weights_smoothed=smoothed,
        interpretation=khat_interpretation(khat),
    )


# ---------------- Proposal-centring diagnostic ----------------

# Central finite-difference step, as a fraction of each dimension's Laplace sd.
# Stepping in posterior-sd units makes the probe scale-free across the eight
# very differently scaled dimensions; 1e-2 sd is small enough that the quadratic
# term cancels in the central difference and large enough that the resulting
# log-target difference is ~1e-4 of the log-marginal's magnitude, far above
# double-precision noise.
_CENTRING_FD_STEP_SD = 1e-2

CENTRING_NOTE = (
    "Gradient of the log PSIS target at the proposal centre, in units of nats "
    "per Laplace sd. The Laplace Gaussian is by construction stationary at its "
    "own centre, so the proposal is centred at a mode of its target only if "
    "every entry here is ~0. A large entry means khat is diagnosing the "
    "centring of the proposal, not the adequacy of the variational family."
)


def _log_abs_dphi_du(phi: np.ndarray) -> np.ndarray:
    """log|dphi/du| for the unconstrained -> constrained reparameterization.

    ``d exp(u)/du = phi`` on the log-transformed variance dimensions and
    ``d sigmoid(u)/du = q(1 - q)`` on the logit-transformed transitions; the
    beta dimensions are the identity and contribute nothing.

    Args:
        phi: Constrained parameter vector(s), shape ``(..., 8)``.

    Returns:
        The log absolute Jacobian determinant, shape ``phi.shape[:-1]``.
    """
    phi = np.asarray(phi, dtype=float)
    q = phi[..., _Q_IDX]
    return np.log(phi[..., _VAR_IDX]).sum(axis=-1) + (
        np.log(q) + np.log1p(-q)
    ).sum(axis=-1)


@dataclass
class CentringDiagnostic:
    """Whether the Laplace proposal sits at a mode of the PSIS target.

    Attributes:
        grad_sd_units: Per-dimension gradient of the log target at the proposal
            centre, scaled by that dimension's Laplace sd, shape ``(8,)``.
        dims: Parameter names aligned with `grad_sd_units`.
        max_abs: Largest absolute entry of `grad_sd_units`.
        max_abs_dim: Name of the dimension attaining `max_abs`.
        fd_step_sd: Central-difference step used, in Laplace-sd units.
    """

    grad_sd_units: np.ndarray
    dims: tuple[str, ...]
    max_abs: float
    max_abs_dim: str
    fd_step_sd: float

    def to_dict(self) -> dict[str, Any]:
        """Serializable summary for the validation CLI's JSON artifact.

        Returns:
            The per-dimension gradient keyed by parameter name, the worst
            dimension and its magnitude, the finite-difference step, and the
            reader-facing `CENTRING_NOTE`.
        """
        return {
            "centring_grad_sd_units": {
                name: float(g) for name, g in zip(self.dims, self.grad_sd_units)
            },
            "centring_grad_max_abs_sd": self.max_abs,
            "centring_grad_max_abs_dim": self.max_abs_dim,
            "centring_fd_step_sd": self.fd_step_sd,
            "centring_note": CENTRING_NOTE,
        }


def phi_centring_gradient(
    vem_output: VEMOutput,
    phi_posterior: PhiPosterior,
    markets: list[MarketData],
    prior: PhiPrior | None = None,
    n_jobs: int = 1,
    fd_step_sd: float = _CENTRING_FD_STEP_SD,
) -> CentringDiagnostic:
    """Log-target gradient at the Laplace centre, in Laplace-sd units.

    khat only measures whether the variational *family* is rich enough once the
    proposal is centred at a mode of its own target: a richer family centred at
    the same non-mode with the same curvature produces the same khat. This is
    the precondition check. It differentiates the PSIS target

        log p(Y | phi, theta_w_hat) + log p(phi)

    at the proposal centre by central finite differences along each
    unconstrained coordinate — the coordinates the Laplace Gaussian actually
    lives on, where that Gaussian is stationary at its centre by construction —
    and reports the result in units of one Laplace sd. A mode gives ~0 in every
    dimension; an entry of, say, -13 says the target rises steeply 13 posterior
    sds' worth away from where q is centred, and khat is then measuring that
    displacement rather than the family.

    The target is carried onto the unconstrained scale with its change-of-
    variables term (`_log_abs_dphi_du`), matching how `PhiPosterior.logpdf`
    carries the Gaussian onto the constrained scale; the importance ratio is
    reparameterization-invariant, so the two conventions locate the same
    stationary point.

    Cost is ``2 * 8`` ADF forward passes over the markets, independent of
    `n_draws` — under 2% of a 1000-draw `psis_khat` call at gate scale.

    Args:
        vem_output: Fitted VEM output supplying `theta_w`, the standardization
            constants, and the base `ModelParams` each probe point overlays.
        phi_posterior: The Laplace proposal whose centring is being checked.
        markets: Markets the VEM was fit on. An empty list drops the likelihood
            term, leaving the prior as the target (used by the tests as an
            analytically known control).
        prior: The prior spec VEM was fit with; `None` uses `PhiPrior()`.
        n_jobs: joblib workers over the probe points; results are identical for
            any value.
        fd_step_sd: Central-difference step in Laplace-sd units.

    Returns:
        A `CentringDiagnostic` with the per-dimension sd-scaled gradient and
        its worst entry.

    Raises:
        ValueError: If `phi_posterior.dims` is not this module's canonical
            parameter order.
    """
    if tuple(phi_posterior.dims) != _PHI_DIMS:
        raise ValueError(
            f"phi_posterior.dims {tuple(phi_posterior.dims)} does not match the "
            f"canonical parameter order {_PHI_DIMS}"
        )
    if prior is None:
        prior = PhiPrior()

    mean_u = np.asarray(phi_posterior.mean_u, dtype=float)
    sd_u = np.sqrt(np.diag(np.asarray(phi_posterior.cov_u, dtype=float)))
    k = mean_u.size
    # Probe points ordered (dim 0 forward, dim 0 backward, dim 1 forward, ...)
    # so the difference below reshapes to (k, 2) without an index map.
    offsets = np.repeat(np.eye(k) * (fd_step_sd * sd_u), 2, axis=0)
    offsets[1::2] *= -1.0
    probes = phi_posterior.to_constrained(mean_u + offsets)

    if markets:
        log_lik = np.asarray(
            Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_adf_log_lik)(vem_output, markets, phi) for phi in probes
            ),
            dtype=float,
        )
    else:
        log_lik = np.zeros(2 * k)

    log_target = log_lik + prior.log_prior(probes) + _log_abs_dphi_du(probes)
    paired = log_target.reshape(k, 2)
    # grad_i * sd_i = (f(u + c*sd_i) - f(u - c*sd_i)) / (2c): the sd factor is
    # already carried by the step, so no separate rescaling is needed.
    grad_sd_units = (paired[:, 0] - paired[:, 1]) / (2.0 * fd_step_sd)

    worst = int(np.argmax(np.abs(grad_sd_units)))
    return CentringDiagnostic(
        grad_sd_units=grad_sd_units,
        dims=tuple(phi_posterior.dims),
        max_abs=float(abs(grad_sd_units[worst])),
        max_abs_dim=str(phi_posterior.dims[worst]),
        fd_step_sd=float(fd_step_sd),
    )
