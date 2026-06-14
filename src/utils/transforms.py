import numpy as np
from scipy.special import expit, logit as _scipy_logit, logsumexp

_LOGIT_EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    return _scipy_logit(np.clip(p, _LOGIT_EPS, 1.0 - _LOGIT_EPS))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return expit(x)


def log1pexp(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x > 35.0, x, np.log1p(np.exp(np.minimum(x, 35.0))))


def log_normalize(log_w: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (normalized log weights, log sum of weights)."""
    log_z = float(logsumexp(log_w))
    return log_w - log_z, log_z


def log_ess(log_w: np.ndarray) -> float:
    """Log effective sample size from (possibly unnormalized) log weights."""
    log_w_norm, _ = log_normalize(log_w)
    return float(-logsumexp(2.0 * log_w_norm))


def ess(log_w: np.ndarray) -> float:
    return float(np.exp(log_ess(log_w)))


def systematic_resample(log_w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling. Returns integer indices of length N."""
    N = len(log_w)
    log_w_norm, _ = log_normalize(log_w)
    cumsum = np.cumsum(np.exp(log_w_norm))
    cumsum[-1] = 1.0  # guard against floating-point overshoot
    u = (rng.random() + np.arange(N)) / N
    return np.searchsorted(cumsum, u)
