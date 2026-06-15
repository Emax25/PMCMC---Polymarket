"""Numerical utility transforms used throughout the PMCMC inference pipeline.

Implements the logit/sigmoid pair, the numerically stable log1pexp (softplus),
log-normalisation of particle weights, effective sample size, and systematic
resampling — the building blocks for log-space SMC arithmetic (§6 of
ARCHITECTURE.md).
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp
from scipy.special import logit as _scipy_logit

# Clamp prices away from 0/1 before logit so that boundary trades (which do
# occur in the Polymarket feed due to rounding) don't produce ±inf observations.
_LOGIT_EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    """Clipped logit so boundary prices (0 or 1) don't produce ±inf.

    Args:
        p: Probabilities in [0, 1]; values outside are clipped to [ε, 1-ε]
            before the transform.

    Returns:
        log(p / (1-p)) elementwise, guaranteed finite.
    """
    return _scipy_logit(np.clip(p, _LOGIT_EPS, 1.0 - _LOGIT_EPS))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid (inverse logit) via scipy.special.expit for numerical stability.

    Args:
        x: Real-valued array.

    Returns:
        1 / (1 + exp(-x)) elementwise, in (0, 1).
    """
    return expit(x)


def log1pexp(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x)).

    For x > 35 the naive form exp(x) overflows to inf; the identity
    log(1 + exp(x)) ≈ x for large x gives the exact limit without overflow.

    Args:
        x: Real-valued array.

    Returns:
        log(1 + exp(x)) elementwise.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x > 35.0, x, np.log1p(np.exp(np.minimum(x, 35.0))))


def log_normalize(log_w: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize log-weights with a log-sum-exp shift.

    Args:
        log_w: Unnormalized log-weights over particles, typically shape ``(N,)``.

    Returns:
        A pair ``(log_w_norm, log_z)`` where ``log_w_norm`` sums to one after
        exponentiation and ``log_z`` is the normalizing constant
        ``log(sum(exp(log_w)))``.
    """
    log_z = float(logsumexp(log_w))
    return log_w - log_z, log_z


def log_ess(log_w: np.ndarray) -> float:
    """Compute log effective sample size from log-weights.

    Args:
        log_w: Log-weights, normalized or unnormalized, over a particle set.

    Returns:
        ``log(ESS)`` where ``ESS`` is the effective sample size induced by the
        normalized weights.
    """
    log_w_norm, _ = log_normalize(log_w)
    return float(-logsumexp(2.0 * log_w_norm))


def ess(log_w: np.ndarray) -> float:
    """Effective sample size ESS = exp(log_ess(log_w)).

    Args:
        log_w: Log weights, possibly unnormalized; shape (N,).

    Returns:
        ESS in [1, N].
    """
    return float(np.exp(log_ess(log_w)))


def systematic_resample(log_w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw particle ancestor indices via systematic resampling.

    Args:
        log_w: Log-weights over particles, shape ``(N,)``.
        rng: Random generator used for the single systematic offset draw.

    Returns:
        Integer ancestor indices of shape ``(N,)`` for multinomial-equivalent,
        low-variance resampling.
    """
    N = len(log_w)
    log_w_norm, _ = log_normalize(log_w)
    cumsum = np.cumsum(np.exp(log_w_norm))
    cumsum[-1] = 1.0  # guard against floating-point overshoot
    u = (rng.random() + np.arange(N)) / N
    return np.searchsorted(cumsum, u)


def bounded_indicator_probability(weights: np.ndarray, indicator: np.ndarray) -> float:
    """Compute a clipped weighted indicator probability.

    Clips ``weights @ 1{indicator == 1}`` into ``[0, 1]`` to absorb tiny
    floating-point roundoff (typically around 1e-16) after log-weight
    normalization.

    Args:
        weights: Normalized particle weights over the current ensemble.
        indicator: Binary particle state indicator (for example ``V`` or ``Z``).

    Returns:
        The weighted indicator probability clipped to the closed interval
        ``[0, 1]``.
    """
    return min(1.0, max(0.0, float(weights @ indicator.astype(float))))
