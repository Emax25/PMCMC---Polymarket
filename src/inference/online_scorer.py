"""Online insider scoring: ADF filtering with continuously adapted parameters.

Wraps `src.inference.adf_filter.ADFFilter` — the O(1)-per-trade E-step — in a
Cappé & Moulines (2009) online-EM loop, so a live trade stream is scored and the
model is re-fit in the same pass. Batch VEM (`variational_em`) re-reads the whole
dataset every EM iteration; this module reads each trade exactly once:

    score   q(Z_t = 1) from `ADFFilter.step` under the *current* parameters
    stats   S_t = (1 - rho_t) * S_{t-1} + s(trade_t)      (decayed sufficient
            statistics; `s(.)` is the batch M-step's per-trade contribution)
    M-step  phi_t = map(S_t)                              (unchanged batch maps)

Every map is the batch one — `PhiPrior.sigma2_map` / `tau2_map` / `q_map` for
the variance and transition blocks, `variational_em._update_beta_irls` for the
logistic block, and the Beta-count posterior mean for `theta_w`. Nothing here
invents an estimator; the online part is only *which statistics* the maps see.

Design notes, in the order they bite:

  * **Decayed sums, not decayed averages.** The recursion above accumulates the
    per-trade statistic with weight 1 (equivalently, the averaged form
    `s_bar <- (1 - rho) s_bar + rho s_t` scaled by the effective window
    `1 / rho`). The batch MAP maps consume *sum*-scale statistics, so this is
    what gives the prior the same weight relative to an effective window of
    `1 / (1 - lambda)` trades that it has relative to `T` trades in batch. Feeding
    the averaged form straight into the maps would let the prior swamp the data.

  * **Seeded from the incoming fit.** The statistics start at the values whose
    map reproduces the supplied `params` / `theta_w`, weighted by
    `OnlineScorerConfig.effective_window` pseudo-trades. Adaptation therefore
    starts *at* the batch fit and drifts away from it, rather than jumping to
    the prior on trade 1. Where the inversion would need a negative statistic
    (a variance below what the Inverse-Gamma prior alone implies) it is clamped
    at zero and the seed is only approximate.

  * **Frozen limit.** `forgetting = 1.0` gives `rho_t == 0`; combined with
    `n_refresh = None` the adaptation block is skipped entirely and the scorer
    is a bare, frozen-parameter `ADFFilter`, bit-for-bit. This is the regression
    anchor `tests/test_online_scorer.py` pins.

  * **Per-wallet `theta_w`.** Decayed Beta counts `(s_w, n_w)` with posterior
    mean `(a + s_w) / (a + b + n_w)` — the exact `beta_S = beta_Z = 0` reduction
    of the batch `_update_theta_w` Newton block. Counts live in a growable dict,
    so a wallet the batch fit never saw simply has no entry and scores at the
    `Beta(a, b)` prior mean `a / (a + b)`: cold start is the absence of a
    special case, not one. Each wallet's counts decay on *its own* trade clock,
    which keeps the update O(1) per trade instead of O(n_wallets).

  * **`delta = 0` exclusion.** Same-second trades are dropped from the process-
    variance statistic, mirroring the batch fix (ARCHITECTURE.md §6.1); the
    `1/delta` in `SS_v` would otherwise be a division by zero and poison every
    later parameter through the decayed state.

  * **Beta refresh.** `beta_S`/`beta_Z` cannot be updated by a scalar recursion
    (their M-step is an IRLS solve), so they are refit every `n_refresh` trades
    by calling the batch `_update_beta_irls` on the most recent `beta_window`
    trades, warm-started at the current betas. Off by default, matching
    `variational_em(estimate_betas=False)` and for the same reason (§6.2).

Reference: Cappé, O. & Moulines, E. (2009) "On-line expectation-maximization
algorithm for latent data models", JRSS-B 71(3) — the decayed-sufficient-
statistic recursion and its Robbins-Monro / forgetting-factor rates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from config.default_params import ModelParams, OnlineScorerConfig, PhiPrior
from src.inference.adf_filter import ADFFilter
from src.inference.particle_gibbs import MarketData
from src.inference.variational_em import _update_beta_irls
from src.utils.transforms import logit

# Minimum usable rows before a beta refresh is attempted. `_update_beta_irls`
# fits two coefficients; on a handful of near-identical rows the data curvature
# is rank-deficient and the estimate is pure Cauchy prior, so refreshing that
# early only injects noise into an otherwise warm-started coefficient.
_MIN_BETA_ROWS = 8

# Floor/ceiling reproducing `_vem_m_step`'s degenerate-regime guards exactly:
# transition probabilities are held off {0, 1} (an absorbing V chain kills the
# regime's sufficient statistics forever) and variances off 0.
_Q_CLIP = 1e-6
_VAR_FLOOR = 1e-6

# Initial per-trade split of the q(V) / q(Z) mass used when seeding the decayed
# statistics: neutral 0.5/0.5, since the seed encodes the *parameters* of the
# incoming fit and carries no information about which regime the stream is in.
_SEED_MASS = 0.5


class OnlineScore(NamedTuple):
    """One trade's online score and the state that produced it.

    Mirrors `ADFStep` and adds the adaptive state, which a caller monitoring a
    live stream needs alongside the score itself.

    Attributes:
        q_vz: (4,) soft assignment ``q(V_t=v, Z_t=z)`` indexed by ``k = 2*v + z``.
        Z_prob: ``q(Z_t = 1)`` — the insider score for this trade.
        V_prob: ``q(V_t = 1)`` — the high-volatility regime probability.
        X_mean: Collapsed ``E[X_t | Y_{0:t}]``.
        X_var: Collapsed ``Var[X_t | Y_{0:t}]``.
        log_evidence: ``log p(Y_t | Y_{0:t-1})`` under the parameters in force
            for this trade.
        theta_w: The per-wallet propensity the trade was scored *under*, i.e.
            before this trade's own count update — the prior-mean
            ``a / (a + b)`` for a wallet never seen before.
        t: 0-based position of this trade in the stream.
    """

    q_vz: np.ndarray
    Z_prob: float
    V_prob: float
    X_mean: float
    X_var: float
    log_evidence: float
    theta_w: float
    t: int


@dataclass
class _DecayedStats:
    """Exponentially-decayed copies of the batch M-step sufficient statistics.

    Field-for-field the sum-scale counterparts of
    `variational_em._mstep_sufficient_stats`, so the batch MAP maps consume them
    unchanged. `n_trans[i, j]` is the batch ``n_ij`` expected transition count.
    """

    SS_v: np.ndarray
    N_v: np.ndarray
    n_trans: np.ndarray
    SS_z: np.ndarray
    N_z: np.ndarray

    def decay(self, rho: float) -> None:
        """Scale every statistic by ``1 - rho`` in place (one trade of ageing)."""
        keep = 1.0 - rho
        self.SS_v *= keep
        self.N_v *= keep
        self.n_trans *= keep
        self.SS_z *= keep
        self.N_z *= keep


def _seed_stats(params: ModelParams, prior: PhiPrior, window: float) -> _DecayedStats:
    """Build decayed statistics whose MAP maps reproduce ``params``.

    Inverts each closed-form M-step map at a total mass of ``window``
    pseudo-trades, so the scorer's first parameter refit lands on (or very near)
    the fit it was handed instead of on the prior mode. The inversions are:

        sigma2_map(SS, N) = (b + SS/2) / (a + N/2 + 1)
                            ->  SS = 2 (sigma2 (a + N/2 + 1) - b)
        tau2_map(SS, N)   = (SS + 2b) / (N + 2a)
                            ->  SS = tau2 (N + 2a) - 2b
        q_map(n_sw, n_st) = (1 + n_sw) / (2 + n_sw + n_st)
                            ->  n_sw = q (2 + M) - 1

    with ``N = _SEED_MASS * window`` per regime and ``M`` the row's total
    transition mass. `sigma2` and `q` can require a negative statistic when the
    supplied value sits below what the prior alone implies (a very small
    variance, or a transition rarer than the Beta(1, 1) pseudo-counts allow);
    those are clamped at zero, leaving the seed prior-pulled rather than exact.
    That degrades the seed, never the recursion.

    Args:
        params: The fit to seed from — typically a batch `VEMOutput.params`.
        prior: The prior spec whose maps the scorer will apply.
        window: Effective number of pseudo-trades the seed is worth.

    Returns:
        Statistics positioned at ``params``.
    """
    mass = _SEED_MASS * window
    SS_v = np.array(
        [
            max(
                2.0
                * (
                    s * (prior.sigma2_ig_alpha + mass / 2.0 + 1.0)
                    - prior.sigma2_ig_beta
                ),
                0.0,
            )
            for s in (params.sigma2_0, params.sigma2_1)
        ]
    )
    SS_z = np.array(
        [
            max(t * (mass + 2.0 * prior.tau2_ig_alpha) - 2.0 * prior.tau2_ig_beta, 0.0)
            for t in (params.tau2_0, params.tau2_1)
        ]
    )

    # Split the transition mass by the V chain's stationary law, so the seeded
    # rows are the ones the chain would actually visit.
    denom_q = params.q_01 + params.q_10
    rho_V = params.q_01 / denom_q if denom_q > 0 else 0.5
    n_trans = np.empty((2, 2))
    for i, (q_switch, row_mass) in enumerate(
        ((params.q_01, (1.0 - rho_V) * window), (params.q_10, rho_V * window))
    ):
        n_switch = max(
            q_switch * (prior.q_beta_a + prior.q_beta_b + row_mass) - prior.q_beta_a,
            0.0,
        )
        n_trans[i, 1 - i] = n_switch
        n_trans[i, i] = max(row_mass - n_switch, 0.0)

    return _DecayedStats(
        SS_v=SS_v,
        N_v=np.full(2, mass),
        n_trans=n_trans,
        SS_z=SS_z,
        N_z=np.full(2, mass),
    )


class OnlineScorer:
    """Streaming ADF scorer with online-EM parameter and `theta_w` adaptation.

    One instance scores one trade stream (a market, or a live feed). Call
    `step` per trade; read `params` / `theta_w` at any point for the current
    fit. Instances are independent and hold no global state.

    The scorer owns an `ADFFilter` and pushes freshly adapted parameters into it
    between trades — the filter itself treats parameters as fixed for its
    lifetime, which is correct for the batch driver and exactly what the online
    path must override.
    """

    def __init__(
        self,
        params: ModelParams,
        theta_w: np.ndarray,
        m_S: float,
        s_S: float,
        m_Z: float,
        *,
        config: OnlineScorerConfig | None = None,
        prior: PhiPrior | None = None,
    ) -> None:
        """Initialize at trade 0 of a fresh stream.

        Args:
            params: Starting parameters, normally a batch `VEMOutput.params`;
                ``beta_S``/``beta_Z`` are on the internal (standardized) scale.
            theta_w: (n_wallets,) per-wallet propensities on the probability
                scale. May be empty — every wallet then cold-starts at the
                `Beta(a, b)` prior mean. Copied, never mutated.
            m_S: Pooled mean of log_size_ratio (standardization), held fixed for
                the stream's lifetime: re-standardizing mid-stream would move
                `beta_S`'s scale under the estimate that is tracking it.
            s_S: Pooled std of log_size_ratio; see `adf_filter.S_STD_FLOOR`.
            m_Z: Pooled mean of ``E[Z_prev]`` (centering), likewise fixed.
            config: Forgetting / learning-rate schedule; ``None`` uses
                `OnlineScorerConfig` defaults (lambda = 0.98, no beta refresh).
            prior: The MAP prior spec; ``None`` uses `PhiPrior` defaults, i.e.
                the same priors the batch M-step optimizes against.
        """
        self._config = config if config is not None else OnlineScorerConfig()
        self._prior = prior if prior is not None else PhiPrior()
        self._params_init = params
        self._theta_init = np.asarray(theta_w, dtype=float).copy()
        self.m_S = m_S
        self.s_S = s_S
        self.m_Z = m_Z

        self._prior_mean = params.a / (params.a + params.b)
        self._prior_mean_logit = float(logit(self._prior_mean))
        self.reset()

    # ---------------- Public surface ----------------

    @property
    def params(self) -> ModelParams:
        """Current adapted model parameters."""
        return self._params

    @property
    def theta_w(self) -> np.ndarray:
        """Current per-wallet propensities, indexed by wallet id (a live view)."""
        return self._theta

    @property
    def t(self) -> int:
        """Number of trades consumed so far."""
        return self._t

    def reset(self) -> None:
        """Rewind to trade 0, re-seeding statistics from the initial fit."""
        self._params = self._params_init
        self._t = 0
        window = self._config.effective_window
        self._stats = _seed_stats(self._params, self._prior, window)

        # theta_w state: `_theta` / `_logit_theta` are the dense arrays the
        # filter indexes (grown geometrically as new wallet ids appear), while
        # `_theta_counts` is the growable decayed-count dict. A wallet absent
        # from the dict has no online evidence and sits at the prior mean.
        n_init = self._theta_init.size
        self._theta = np.full(max(n_init, 1), self._prior_mean)
        self._theta[:n_init] = self._theta_init
        self._logit_theta = logit(self._theta)
        self._theta_counts: dict[int, tuple[float, float]] = {}
        a, b = self._params.a, self._params.b
        for w in range(n_init):
            # Inverse of the Beta posterior mean at `window` pseudo-trades:
            # (a + s) / (a + b + n) = theta. Clamped into [0, n] so an extreme
            # incoming theta cannot ask for impossible counts.
            s_w = float(
                np.clip(self._theta_init[w] * (a + b + window) - a, 0.0, window)
            )
            self._theta_counts[w] = (s_w, window)

        self._adf = ADFFilter(self._params, self._theta, self.m_S, self.s_S, self.m_Z)
        # `ADFFilter` caches `logit(theta_w)` at construction; hand it the
        # scorer's own array so per-wallet updates are visible without a resync.
        self._adf._logit_theta = self._logit_theta

        self._prev: OnlineScore | None = None
        self._buffer = self._new_buffer()

    def step(
        self,
        y: float,
        delta: float,
        log_size_ratio: float,
        wallet_id: int,
    ) -> OnlineScore:
        """Score one trade, then adapt the parameters on it.

        The trade is filtered under the parameters in force *before* it arrives
        (a genuine one-step-ahead score, never peeking at its own contribution),
        and only afterwards folded into the decayed statistics.

        Args:
            y: Logit-price observation ``Y_t``.
            delta: Seconds since the previous trade; ``0.0`` at trade 0, and
                legitimately ``0.0`` for same-second trades thereafter.
            log_size_ratio: ``log(S_t / S_bar)`` for this trade.
            wallet_id: Integer wallet index; ids beyond anything seen so far are
                admitted and cold-start at the `Beta(a, b)` prior mean.

        Returns:
            The `OnlineScore` for this trade.
        """
        w = int(wallet_id)
        self._ensure_wallet(w)
        theta_used = float(self._theta[w])

        out = self._adf.step(y, delta, log_size_ratio, w)
        score = OnlineScore(
            q_vz=out.q_vz,
            Z_prob=out.Z_prob,
            V_prob=out.V_prob,
            X_mean=out.X_mean,
            X_var=out.X_var,
            log_evidence=out.log_evidence,
            theta_w=theta_used,
            t=self._t,
        )

        rho = self._config.rho(self._t)
        if rho > 0.0:
            self._accumulate(score, float(y), float(delta), float(log_size_ratio), rho)
            self._refit_params()
            self._update_theta(w, score.Z_prob, rho)
            self._sync_params()

        self._t += 1
        self._prev = score
        self._push_buffer(w, float(log_size_ratio), score.Z_prob)
        return score

    # ---------------- Online sufficient statistics ----------------

    def _accumulate(
        self,
        score: OnlineScore,
        y: float,
        delta: float,
        log_size_ratio: float,
        rho: float,
    ) -> None:
        """Age the statistics by one trade and fold in this trade's contribution.

        The per-trade contributions are term-for-term the ones
        `variational_em._mstep_sufficient_stats` sums over a market: observation
        residuals from every trade, process increments and transition products
        from every consecutive pair.

        Args:
            score: This trade's filtered posterior.
            y: Logit-price observation.
            delta: Seconds since the previous trade.
            log_size_ratio: ``log(S_t / S_bar)`` for this trade.
            rho: Learning rate for this trade.
        """
        stats = self._stats
        stats.decay(rho)
        q_vz = score.q_vz

        # Observation-variance statistics: every trade contributes, trade 0
        # included (it has an observation even though it has no predecessor).
        denom_t = max(1.0 + log_size_ratio * self._params.gamma, 0.1)
        resid2_obs = (y - score.X_mean) ** 2
        for z in (0, 1):
            q_Z_z = q_vz[z] + q_vz[2 + z]
            stats.SS_z[z] += q_Z_z * (resid2_obs + score.X_var) * denom_t
            stats.N_z[z] += q_Z_z

        prev = self._prev
        if prev is None:
            return

        q_V = np.array([q_vz[0] + q_vz[1], q_vz[2] + q_vz[3]])
        prev_q_V = np.array(
            [prev.q_vz[0] + prev.q_vz[1], prev.q_vz[2] + prev.q_vz[3]]
        )
        stats.n_trans += np.outer(prev_q_V, q_V)

        # Process-variance statistics. Same-second trades (delta == 0) are
        # dropped: the statistic divides by delta, and one such trade would send
        # SS_v to inf and every downstream parameter to NaN — permanently, since
        # the decayed state carries it forward (ARCHITECTURE.md §6.1).
        if delta > 0.0:
            increment = (score.X_mean - prev.X_mean) ** 2 + score.X_var + prev.X_var
            stats.SS_v += q_V * increment / delta
            stats.N_v += q_V

    def _refit_params(self) -> None:
        """Re-apply the batch closed-form M-step maps to the decayed statistics.

        Block-for-block identical to `_vem_m_step`'s variance/transition block,
        including its degenerate-regime clips and the ``sigma2_1 >= sigma2_0`` /
        ``tau2_1 <= tau2_0`` order constraints that identify the regimes.
        """
        stats, prior = self._stats, self._prior
        q_01 = float(
            np.clip(
                prior.q_map(stats.n_trans[0, 1], stats.n_trans[0, 0]),
                _Q_CLIP,
                1.0 - _Q_CLIP,
            )
        )
        q_10 = float(
            np.clip(
                prior.q_map(stats.n_trans[1, 0], stats.n_trans[1, 1]),
                _Q_CLIP,
                1.0 - _Q_CLIP,
            )
        )
        sigma2_0 = max(prior.sigma2_map(stats.SS_v[0], stats.N_v[0]), _VAR_FLOOR)
        sigma2_1 = max(
            prior.sigma2_map(stats.SS_v[1], stats.N_v[1]), _VAR_FLOOR, sigma2_0
        )
        tau2_0 = max(prior.tau2_map(stats.SS_z[0], stats.N_z[0]), _VAR_FLOOR)
        tau2_1 = min(
            max(prior.tau2_map(stats.SS_z[1], stats.N_z[1]), _VAR_FLOOR), tau2_0
        )
        self._params = replace(
            self._params,
            q_01=q_01,
            q_10=q_10,
            sigma2_0=sigma2_0,
            sigma2_1=sigma2_1,
            tau2_0=tau2_0,
            tau2_1=tau2_1,
        )

    def _sync_params(self) -> None:
        """Push the adapted parameters into the wrapped `ADFFilter`.

        `ADFFilter` derives `_q_01`, `_q_10` and the stationary `_rho_V` from
        `params` once, at construction, because the batch E-step freezes
        parameters for a whole pass. The online path is precisely the case that
        invalidates that assumption, so the derived caches are refreshed here
        rather than rebuilding the filter and transplanting its carried Kalman
        state (which would touch strictly more of its internals).
        """
        params = self._params
        adf = self._adf
        adf.params = params
        adf._q_01 = params.q_01
        adf._q_10 = params.q_10
        denom_q = params.q_01 + params.q_10
        adf._rho_V = params.q_01 / denom_q if denom_q > 0 else 0.5

    # ---------------- Per-wallet theta_w ----------------

    def _ensure_wallet(self, wallet_id: int) -> None:
        """Grow the dense theta arrays so ``wallet_id`` is addressable.

        Doubling keeps the amortized cost O(1) per trade, which matters because
        a live stream meets new wallets indefinitely. New slots hold the
        `Beta(a, b)` prior mean, which is exactly what an evidence-free wallet
        scores at.

        Args:
            wallet_id: Integer wallet index of the incoming trade.
        """
        size = self._theta.size
        if wallet_id < size:
            return
        new_size = max(wallet_id + 1, 2 * size)
        theta = np.full(new_size, self._prior_mean)
        theta[:size] = self._theta
        logit_theta = np.full(new_size, self._prior_mean_logit)
        logit_theta[:size] = self._logit_theta
        self._theta = theta
        self._logit_theta = logit_theta
        # Re-point the filter: the old array object it shared is now stale.
        self._adf._logit_theta = logit_theta

    def _update_theta(self, wallet_id: int, Z_prob: float, rho: float) -> None:
        """Fold one trade into its wallet's decayed Beta counts.

        Applies ``(s, n) <- ((1 - rho) s + q(Z), (1 - rho) n + 1)`` and the
        Beta posterior mean ``(a + s) / (a + b + n)``. The decay uses the
        *wallet's* trade clock, not the stream's: decaying every wallet on every
        trade would cost O(n_wallets) per trade for no statistical gain, since
        an idle wallet has acquired no new evidence to age against.

        Args:
            wallet_id: Integer wallet index of this trade's trader.
            Z_prob: This trade's ``q(Z_t = 1)``, the fractional Bernoulli target.
            rho: Learning rate for this trade.
        """
        s_w, n_w = self._theta_counts.get(wallet_id, (0.0, 0.0))
        keep = 1.0 - rho
        s_w = keep * s_w + Z_prob
        n_w = keep * n_w + 1.0
        self._theta_counts[wallet_id] = (s_w, n_w)
        params = self._params
        theta = (params.a + s_w) / (params.a + params.b + n_w)
        self._theta[wallet_id] = theta
        self._logit_theta[wallet_id] = float(logit(theta))

    # ---------------- Periodic beta refresh ----------------

    def _new_buffer(self) -> deque | None:
        """Rolling ``(wallet_id, log_size_ratio, q(Z))`` window, or None if unused.

        Sized one longer than `beta_window` because the batch covariate builder
        spends the window's first trade as the lag supplying ``x_Z~`` for the
        second (the ``Z_0 := 0`` convention drops trade 0 of any block).
        """
        n_refresh = self._config.n_refresh
        if n_refresh is None or n_refresh <= 0:
            return None
        window = self._config.beta_window
        if window is None:
            window = int(round(self._config.effective_window))
        return deque(maxlen=max(window, _MIN_BETA_ROWS) + 1)

    def _push_buffer(
        self, wallet_id: int, log_size_ratio: float, Z_prob: float
    ) -> None:
        """Record this trade for the beta refresh and fire one when due.

        Args:
            wallet_id: Integer wallet index of this trade's trader.
            log_size_ratio: ``log(S_t / S_bar)`` for this trade.
            Z_prob: This trade's ``q(Z_t = 1)``.
        """
        if self._buffer is None:
            return
        self._buffer.append((wallet_id, log_size_ratio, Z_prob))
        if self._t % self._config.n_refresh == 0:
            self._refresh_betas()

    def _refresh_betas(self) -> None:
        """Refit ``beta_S``/``beta_Z`` on the recent window via the batch IRLS.

        Replays the buffered window as a one-market `MarketData` plus the
        matching ``q_vz`` and hands it to `variational_em._update_beta_irls`, so
        the Cauchy(0, 2.5) penalized IRLS — including its step-halving and
        separation handling — is the batch code, not a copy of it. Warm-starting
        at the current coefficients is what keeps successive refreshes smooth:
        each is a few Newton steps from the last, not an independent fit.

        Only ``Y``/``delta`` are unused by the covariate builder and passed as
        placeholders; the window's own ``log_size_ratio``, ``wallet_ids`` and
        ``q(Z)`` are the real inputs. The window is rectangular rather than
        exponentially weighted (the batch IRLS takes no per-row weights); its
        length is tied to `effective_window`, so the beta block forgets on the
        same timescale as the recursive blocks.
        """
        n = len(self._buffer)
        if n - 1 < _MIN_BETA_ROWS:
            return
        wallet_ids = np.fromiter((row[0] for row in self._buffer), np.int64, n)
        log_size_ratio = np.fromiter((row[1] for row in self._buffer), float, n)
        Z_prob = np.fromiter((row[2] for row in self._buffer), float, n)

        # `_pooled_zj_covariates` reads E[Z] as q_vz[:, 1] + q_vz[:, 3]; putting
        # the mass in columns 3 and 0 reproduces the window's q(Z) exactly.
        q_vz = np.zeros((n, 4))
        q_vz[:, 3] = Z_prob
        q_vz[:, 0] = 1.0 - Z_prob
        md = MarketData(
            Y=np.zeros(n),
            delta=np.ones(n),
            log_size_ratio=log_size_ratio,
            wallet_ids=wallet_ids,
        )
        beta_S, beta_Z, _ = _update_beta_irls(
            [md],
            [q_vz],
            self._theta,
            self.m_S,
            self.s_S,
            self.m_Z,
            self._params.beta_S,
            self._params.beta_Z,
            cauchy_scale=self._prior.beta_cauchy_scale,
        )
        self._params = replace(self._params, beta_S=beta_S, beta_Z=beta_Z)
        self._sync_params()
