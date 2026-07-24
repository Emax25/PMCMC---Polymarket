"""Validation metrics for the variational-EM fast path.

General validation-metrics module for the samplerless VEM defense (plan
2026-07-23-002): the temporal train/tail split with the held-out one-step
predictive log-likelihood scored through the ADF forward pass, and the PSIS-khat
importance-sampling diagnostic over the parameter vector phi.

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
importance ratios against the ADF-implied parameter posterior,

    log w_s = log p(Y | phi_s) + log p(phi_s) - log q(phi_s),

where the likelihood term is the same ADF forward-pass log-marginal used above,
`p` is the model's own `PhiPrior`, and `q` is `PhiPosterior`. Both densities are
evaluated on the *constrained* scale: `PhiPrior.log_prior` is natively
constrained and `PhiPosterior.logpdf` adds the change-of-variables Jacobian of
the unconstrained reparameterization, so the ratio compares two densities of the
same variable (a missing Jacobian would silently tilt the target by
`|du/dphi|`). See the `psis_khat` docstring for the scope of the resulting claim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from joblib import Parallel, delayed

from config.default_params import ModelParams, PhiPrior
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


# ---------------- PSIS-khat diagnostic ----------------

# Result key for the CLI's JSON artifact. The name states what the diagnostic
# actually compares (the Laplace posterior against the ADF-implied parameter
# marginal) so it can never be read as a check of ADF itself — see R5 and the
# `psis_khat` docstring.
PSIS_KHAT_KEY = "psis_khat_laplace_vs_adf"

PSIS_SCOPE_NOTE = (
    "khat measures Laplace-shape adequacy for the ADF-implied parameter "
    "marginal, not ADF's fidelity to the true posterior: q and the target are "
    "built on the same ADF surface, so any bias ADF itself carries is invisible "
    "here. A good khat is expected and is a necessary-not-sufficient check; "
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
        "bad (khat > 0.7): the Laplace proposal does not cover the target's "
        "tails; enrich the variational family before trusting the estimate."
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
    """PSIS-khat between the Laplace posterior and the ADF parameter marginal.

    Draws `phi_s ~ q` from the Laplace posterior and Pareto-smooths the
    importance ratios against the ADF-implied parameter posterior:

        log w_s = log p(Y | phi_s) + log p(phi_s) - log q(phi_s)

    with `log p(Y | phi_s)` the ADF forward-pass log-marginal summed over
    markets, `log p(phi_s)` the model's own prior (`PhiPrior`, the same spec the
    M-step optimizes against), and `log q(phi_s)` the Laplace density. Both
    density terms are evaluated on the **constrained** scale — `log_prior` is
    natively constrained and `PhiPosterior.logpdf` adds the unconstrained ->
    constrained Jacobian — so the ratio compares densities of the same variable.

    **Scope of the claim.** Because `q` and the target are both built on the ADF
    surface, this khat measures *Laplace-shape adequacy for the ADF-implied
    parameter marginal*. It cannot detect ADF's own bias relative to the true
    posterior: a good khat is expected, and is a necessary-not-sufficient check.
    Simulation-based calibration (plan 2026-07-23-003) is the actual faithfulness
    test. Report the value under `PSIS_KHAT_KEY` alongside `PSIS_SCOPE_NOTE`, and
    read it with the standard bands (< 0.5 good, 0.5-0.7 ok, > 0.7 bad); khat
    > 0.7 is the plan's stop condition to enrich the variational family, not a
    test assertion.

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
