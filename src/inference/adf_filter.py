"""Stepwise assumed-density filter (ADF) for the switching SSM.

The per-trade kernel behind the VEM E-step, exposed as an object that carries
its own state so it can be driven either in batch (``variational_em._vem_e_step``
loops over an entire market) or one trade at a time (live scoring, where the
next trade is not yet known).

At each trade the filter collapses the 4-way ``(V_t, Z_t)`` mixture back onto a
single Gaussian, which is what makes the pass O(1) per trade — O(4T) per market
— rather than exponential in T:

    prior   P(V_t) = P(V_{t-1}) @ [[1-q_01, q_01], [q_10, 1-q_10]]
            logit P(Z_t=1) = level_t + beta_S * x_S~_t + beta_Z * x_Z~_t
            level_t        = logit(theta_w[w_t])   (wallet mode)
                           = alpha                 (anonymous mode)
    update  one Kalman predict+update per (v, z) combo off the *shared*
            incoming (mu, sigma2) — see `kalman._kalman_step_all_combos`
    collapse q_t(v, z) proportional to P(V_t=v) P(Z_t=z) p(Y_t | v, z)
            mu_t     = sum_k q_t[k] mu_t[k]
            sigma2_t = sum_k q_t[k] (sigma2_t[k] + (mu_t[k] - mu_t)^2)

Collapsing to one mode discards the path structure (a full mixture Kalman
filter would keep 4^t components); this is the standard assumed-density
approximation of Ghahramani & Hinton (2000).

Conventions carried over from the batch E-step, all load-bearing for
output identity:
  * ``Z_0 := 0`` — trade 0 gets ``log_p_Z = [0, -500]``, i.e. q(Z_0=1) ~ 0,
    and its V prior is the chain's stationary distribution
    ``rho_V = q_01 / (q_01 + q_10)`` rather than a transition.
  * ``E[Z_{t-1}]`` (the filtered value, not the true latent) feeds the next
    trade's logistic predictor — a plug-in that attenuates ``beta_Z``
    (see `variational_em`'s module docstring on regression dilution).
  * Covariates are centered/standardized (Gelman et al. 2008). In wallet mode
    there is no free intercept — the theta_w Beta hierarchy carries the level.
    In anonymous mode (Kalshi's feed has no per-account identifier, so no
    theta_w exists) the per-market intercept ``alpha`` is that level, and
    ``wallet_id`` is ignored entirely.

Reference: Ghahramani, Z. & Hinton, G.E. (2000) "Variational Learning for
Switching State-Space Models", Neural Computation 12(4).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from config.default_params import ModelParams
from src.inference.kalman import _kalman_step_all_combos
from src.utils.transforms import log1pexp, logit

# Below this pooled std of log_size_ratio, treat the size covariate as
# degenerate (all trades effectively the same size) and skip the 0.5/s_S
# scale factor. A truly constant column has std ~1e-16 from floating-point
# rounding, not exactly 0.0, so the guard uses a floor well above that noise
# floor rather than an exact `s_S > 0` check. Shared with `variational_em`,
# whose pooled M-step design matrix must standardize identically.
S_STD_FLOOR = 1e-8

# Trade 0's insider prior. The model fixes Z_0 := 0, imposed here as a
# log-prior of -500 on Z_0 = 1 (exp(-500) ~ 7e-218, numerically zero in the
# normalized q) rather than -inf, which would make the logsumexp NaN-prone.
_LOG_P_Z0 = -500.0

# Floor inside the log of the V prior. Once the data decisively rules out one
# regime the collapsed q(V) component underflows to exactly 0.0, and log(0.0)
# = -inf would propagate into `log_prior_joint`; if both V components ever hit
# it the step's logsumexp goes to -inf and the normalized q becomes NaN. 1e-300
# is far below any probability the model can act on, so the floor is inert.
_PROB_FLOOR = 1e-300


def _logsumexp4(log_joint: np.ndarray) -> float:
    """`scipy.special.logsumexp` for a length-4 array, bit-for-bit.

    scipy's array-API dispatch costs ~70 us per call on an array this small —
    two thirds of a trade step, ~45x the njit Kalman kernel next to it — so the
    algorithm is reproduced here operation-for-operation instead: the maximum is
    separated out of the sum, the residual sum ``s`` is divided by the tie count
    ``m``, and the result is recomposed as ``log1p(s) + log(m) + a_max`` in that
    association. The cases where scipy discards that form and falls back to a
    direct ``log(sum(exp(a)))`` — a non-finite maximum, i.e. any ``+inf``, all
    ``-inf``, or any NaN (which makes ``m`` zero) — delegate to numpy here for
    the same reason, off the hot path.

    Bit-identity with scipy is a contract, not an approximation: verified by
    exact bit comparison (`struct`, not `isclose`) on 570k random 4-vectors
    covering exact tied maxima, the ``[-500, x, y, z]`` trade-0 pattern,
    ``-inf`` and NaN entries, all-equal vectors, and 1-ulp near-ties — zero
    mismatches. Changing the arithmetic or its association breaks the identity
    fixtures in `tests/test_adf_filter.py`.

    Args:
        log_joint: (4,) array of log-weights.

    Returns:
        ``log(sum(exp(log_joint)))``.
    """
    v0, v1, v2, v3 = log_joint.tolist()
    a_max = v0
    if v1 > a_max:
        a_max = v1
    if v2 > a_max:
        a_max = v2
    if v3 > a_max:
        a_max = v3
    if not (math.isfinite(a_max) and v0 == v0 and v1 == v1 and v2 == v2 and v3 == v3):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return float(np.log(np.exp(log_joint).sum()))

    # The tied maxima contribute exp(0) each and are carried by `m`; the rest
    # are summed in index order, matching scipy's masked sum term for term.
    m = 0.0
    s = 0.0
    for v in (v0, v1, v2, v3):
        if v == a_max:
            m += 1.0
        else:
            s += math.exp(v - a_max)
    if s != 0.0:
        s = s / m
    return math.log1p(s) + math.log(m) + a_max


class ADFStep(NamedTuple):
    """One trade's filtered posterior from `ADFFilter.step`.

    Attributes:
        q_vz: (4,) soft assignment ``q(V_t=v, Z_t=z)`` indexed by ``k = 2*v + z``.
        Z_prob: ``q(Z_t = 1)`` — the insider score for this trade.
        V_prob: ``q(V_t = 1)`` — the high-volatility regime probability.
        X_mean: Collapsed ``E[X_t | Y_{0:t}]``.
        X_var: Collapsed ``Var[X_t | Y_{0:t}]``.
        log_evidence: ``log p(Y_t | Y_{0:t-1})``; summing these over a market
            gives the approximate log-marginal the EM trace reports.
    """

    q_vz: np.ndarray
    Z_prob: float
    V_prob: float
    X_mean: float
    X_var: float
    log_evidence: float


class ADFFilter:
    """Stateful O(1)-per-trade assumed-density filter for one market.

    Holds exactly the state the forward recursion carries between trades — the
    collapsed Kalman moments ``(mu, sigma2)``, the previous V-marginal, and the
    previous ``E[Z]`` — plus the read-only parameters/centering constants. One
    instance filters one market; instances are fully independent, so several
    markets (or live streams) can be advanced interleaved.

    The parameters and centering constants are fixed for the filter's lifetime
    by default, matching the batch E-step where they are frozen at the values
    the current EM iteration started from; `set_params` and `set_theta_logits`
    are the supported way for a streaming driver
    (`online_scorer.OnlineScorer`) to replace them between trades.
    """

    def __init__(
        self,
        params: ModelParams,
        theta_w: np.ndarray,
        m_S: float,
        s_S: float,
        m_Z: float,
    ) -> None:
        """Initialize a filter at trade 0 of a fresh market.

        Args:
            params: Model parameters; ``beta_S``/``beta_Z`` are on the
                *internal* (standardized) covariate scale.
            theta_w: (n_wallets,) per-wallet insider propensities on the
                probability scale; converted to logits once here since the
                predictor needs them every trade. May be empty in anonymous
                mode, where the predictor never consults it.
            m_S: Pooled mean of log_size_ratio, for centering the size covariate.
            s_S: Pooled std of log_size_ratio; at or below `S_STD_FLOOR`
                (degenerate constant-size data) the 0.5/s_S scale factor is
                skipped and only centering is applied, avoiding an unstable or
                divide-by-zero scale factor.
            m_Z: Pooled mean of ``E[Z_prev]``, for centering the persistence
                covariate.
        """
        self.m_S = m_S
        self.s_S = s_S
        self.m_Z = m_Z
        self._logit_theta = logit(theta_w)
        self.set_params(params)
        self.reset()

    def set_params(self, params: ModelParams) -> None:
        """Install parameters and refresh the derived transition cache.

        `_q_01`, `_q_10` and the stationary `_rho_V` are derived from `params`
        rather than read per trade, which is only valid while the parameters
        hold still. The online driver adapts them between trades and calls this
        to keep the cache consistent, instead of rebuilding the filter and
        transplanting its carried Kalman state.

        The carried state is deliberately untouched: the new parameters take
        effect from the next `step` onwards.

        Args:
            params: Model parameters to filter under from now on.
        """
        self.params = params
        self._q_01 = params.q_01
        self._q_10 = params.q_10
        # Cached hot-path form of `ModelParams.z_logit_level` — the one place
        # the wallet/anonymous mode switch is *defined*. Hoisted out of the
        # per-trade predictor because it is a per-parameter-set constant, and
        # because in anonymous mode there is no wallet index to look up at all:
        # `_logit_theta` is legitimately empty there. `test_adf_filter.py` pins
        # the two against each other so the cache cannot drift from the method.
        self._anonymous = params.anonymous
        self._alpha = params.alpha

        # Stationary V-marginal of the 2-state chain, used as trade 0's prior
        # (there is no previous trade to transition from). A degenerate chain
        # with q_01 = q_10 = 0 has no unique stationary law; fall back to the
        # uninformative 0.5 rather than dividing by zero.
        denom_q = self._q_01 + self._q_10
        self._rho_V = self._q_01 / denom_q if denom_q > 0 else 0.5

    def set_theta_logits(self, logit_theta: np.ndarray) -> None:
        """Re-point the cached per-wallet insider logits.

        `__init__` converts `theta_w` to logits once, since the predictor needs
        them every trade. A driver that maintains its own logit array — the
        streaming scorer, which grows it as new wallets appear and rewrites
        entries per trade — hands the array over here; it is kept by reference,
        so in-place updates need no further resync.

        Args:
            logit_theta: (n_wallets,) logits of the per-wallet propensities,
                indexed by wallet id. Not copied.
        """
        self._logit_theta = logit_theta

    def reset(self) -> None:
        """Rewind the filter to trade 0, keeping parameters and constants."""
        self.t = 0
        # Length-1 buffers, written in place each step and handed straight to
        # the njit Kalman kernel: reallocating them per trade cost ~2 array
        # allocations on a loop that runs T x n_EM_iter times. The kernel only
        # reads them, so in-place reuse is safe and bit-identical.
        self._mu = np.zeros(1)
        self._sigma2 = np.array([self.params.s0_2])
        # Previous V-marginal, kept as two scalars rather than an array for the
        # same reason. Unused at t = 0 (trade 0 uses the stationary prior) but
        # defined so the carried state is always well-formed.
        self._prev_q_V0 = 1.0 - self._rho_V
        self._prev_q_V1 = self._rho_V
        self._prev_E_Z = 0.0

    def _log_priors(
        self, log_size_ratio: float, wallet_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Log-priors ``(log P(V_t), log P(Z_t))`` for the incoming trade.

        Args:
            log_size_ratio: ``log(S_t / S_bar)`` for this trade.
            wallet_id: Integer wallet index of this trade's trader; ignored (and
                never indexed) in anonymous mode.

        Returns:
            ``(log_p_V, log_p_Z)``, each a length-2 array over ``{0, 1}``.
        """
        if self.t == 0:
            log_p_V = np.array(
                [
                    np.log(max(1.0 - self._rho_V, _PROB_FLOOR)),
                    np.log(max(self._rho_V, _PROB_FLOOR)),
                ]
            )
            return log_p_V, np.array([0.0, _LOG_P_Z0])

        prev_q_V0 = self._prev_q_V0
        prev_q_V1 = self._prev_q_V1
        p_V0 = prev_q_V0 * (1.0 - self._q_01) + prev_q_V1 * self._q_10
        p_V1 = prev_q_V0 * self._q_01 + prev_q_V1 * (1.0 - self._q_10)
        log_p_V = np.array(
            [np.log(max(p_V0, _PROB_FLOOR)), np.log(max(p_V1, _PROB_FLOOR))]
        )

        # Standardize/center covariates (Gelman et al. 2008) before the
        # logistic predictor. The *level* the covariates tilt is mode-dependent
        # (see `set_params`): wallet mode has no free intercept because
        # theta_w's Beta hierarchy carries it; anonymous mode has no theta_w and
        # uses the estimated per-market intercept. Guard s_S below a small floor
        # (degenerate/near-constant-size data) rather than exactly 0: a
        # constant column's std is only ~machine-epsilon due to floating-point
        # rounding, not exactly zero, and dividing by that residual noise would
        # amplify it into an arbitrary value.
        x_S_centered = float(log_size_ratio) - self.m_S
        x_S_tilde = (
            x_S_centered * 0.5 / self.s_S if self.s_S > S_STD_FLOOR else x_S_centered
        )
        x_Z_tilde = self._prev_E_Z - self.m_Z
        params = self.params
        level = (
            self._alpha if self._anonymous else float(self._logit_theta[int(wallet_id)])
        )
        logit_pi = level + params.beta_S * x_S_tilde + params.beta_Z * x_Z_tilde
        # log(1 + exp(x)) in the stable form: log_p_Z = [-lp, logit_pi - lp]
        # is exactly [log(1 - pi), log(pi)] without ever forming pi itself.
        lp = float(log1pexp(logit_pi))
        return log_p_V, np.array([-lp, logit_pi - lp])

    def step(
        self,
        y: float,
        delta: float,
        log_size_ratio: float,
        wallet_id: int,
    ) -> ADFStep:
        """Advance the filter by one trade and return its collapsed posterior.

        Args:
            y: Logit-price observation ``Y_t``.
            delta: Inter-trade time since the previous trade; ``0.0`` at trade 0.
            log_size_ratio: ``log(S_t / S_bar)`` for this trade.
            wallet_id: Integer wallet index of this trade's trader; ignored in
                anonymous mode, where any value (including 0 against an empty
                `theta_w`) is accepted.

        Returns:
            The `ADFStep` for this trade. The filter's state is advanced, so
            the next call consumes trade ``t + 1``.
        """
        params = self.params
        log_p_V, log_p_Z = self._log_priors(log_size_ratio, wallet_id)
        log_prior_joint = (log_p_V[:, None] + log_p_Z[None, :]).reshape(4)

        # All four (v, z) combos share the one incoming Gaussian — that shared
        # state is exactly the assumed-density collapse this filter implements.
        mu_combos, sigma2_combos, log_lik = _kalman_step_all_combos(
            self._mu,
            self._sigma2,
            float(y),
            float(delta),
            float(log_size_ratio),
            params.sigma2_0,
            params.sigma2_1,
            params.tau2_0,
            params.tau2_1,
            params.gamma,
        )

        log_joint = log_prior_joint + log_lik[0]
        log_Z_t = _logsumexp4(log_joint)
        q_t = np.exp(log_joint - log_Z_t)

        # Moment-match the 4-component mixture back to one Gaussian: the
        # variance picks up the between-component spread, not just the
        # within-component average.
        mu_c = mu_combos[0]
        sigma2_c = sigma2_combos[0]
        mu_mixed = float(q_t @ mu_c)
        sigma2_mixed = float(q_t @ (sigma2_c + (mu_c - mu_mixed) ** 2))

        V_prob = q_t[2] + q_t[3]
        Z_prob = float(q_t[1] + q_t[3])
        self._mu[0] = mu_mixed
        self._sigma2[0] = sigma2_mixed
        self._prev_q_V0 = q_t[0] + q_t[1]
        self._prev_q_V1 = V_prob
        self._prev_E_Z = Z_prob
        self.t += 1

        return ADFStep(
            q_vz=q_t,
            Z_prob=Z_prob,
            V_prob=float(V_prob),
            X_mean=mu_mixed,
            X_var=sigma2_mixed,
            log_evidence=log_Z_t,
        )
