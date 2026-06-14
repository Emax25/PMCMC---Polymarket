"""Convergence and particle-filter diagnostics (decision #13).

Per the README: R-hat is computed across the P conditional chains of an
iPMCMC run (or across multiple independent PG runs concatenated); the
headline number is the minimum bulk-ESS across phi components. Particle
degeneracy flags markets where more than 10% of SMC steps have particle
ESS < N/4.

Sample arrays are stored in `(n_iter, n_chains, *extra)` order throughout
the codebase. `arviz` expects `(chain, draw, *extra)`, so we swap the
leading two axes inside each wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np

from src.inference.ipmcmc import iPMCMCOutput
from src.inference.smc import SMCOutput

PHI_PARAM_NAMES = (
    "sigma2_0", "sigma2_1",
    "q_01", "q_10",
    "beta_S", "beta_Z",
    "tau2_0", "tau2_1",
)

RHAT_FLAG_THRESHOLD = 1.01
PARTICLE_ESS_FRACTION = 0.25     # threshold = N/4 (decision #13)
DEGENERACY_FLAG_RATE = 0.10      # >10% steps below threshold flags the market


# ---------------- Building blocks ----------------

def _ensure_2d_or_more(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim < 2:
        raise ValueError(
            f"Need at least (n_iter, n_chains); got shape {samples.shape}."
        )
    return samples


def compute_rhat(samples: np.ndarray) -> np.ndarray | float:
    """Rank-normalized R-hat for samples of shape (n_iter, n_chains, *extra).

    Returns a scalar if `samples` is 2D, an ndarray shaped like `extra`
    otherwise. With fewer than two chains, returns NaN(s) — R-hat requires
    multiple chains by construction.
    """
    samples = _ensure_2d_or_more(samples)
    extra_shape = samples.shape[2:]
    if samples.shape[1] < 2:
        return float("nan") if not extra_shape else np.full(extra_shape, np.nan)
    chain_first = np.swapaxes(samples, 0, 1)
    if samples.ndim == 2:
        return float(az.rhat(chain_first, method="rank"))
    out = np.empty(extra_shape)
    for idx in np.ndindex(*extra_shape):
        slc = chain_first[(slice(None), slice(None)) + idx]
        out[idx] = float(az.rhat(slc, method="rank"))
    return out


def compute_ess(samples: np.ndarray) -> np.ndarray | float:
    """Bulk ESS for samples of shape (n_iter, n_chains, *extra) or (n_iter,).

    Single-chain inputs are supported (arviz computes autocorrelation-based
    ESS within a single chain).
    """
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    samples = _ensure_2d_or_more(samples)
    chain_first = np.swapaxes(samples, 0, 1)
    if samples.ndim == 2:
        return float(az.ess(chain_first, method="bulk"))
    extra_shape = samples.shape[2:]
    out = np.empty(extra_shape)
    for idx in np.ndindex(*extra_shape):
        slc = chain_first[(slice(None), slice(None)) + idx]
        out[idx] = float(az.ess(slc, method="bulk"))
    return out


def particle_degeneracy_rate(
    ess: np.ndarray | float,
    N: int,
    *,
    threshold_fraction: float = PARTICLE_ESS_FRACTION,
) -> float:
    """Fraction of entries in `ess` below `threshold_fraction * N`.

    Accepts a (T,) ESS-per-step vector from a single SMC pass, or any
    multi-dim aggregation across iterations / chains — every entry counts
    equally.
    """
    ess = np.asarray(ess)
    threshold = threshold_fraction * N
    return float((ess < threshold).mean())


def smc_particle_degeneracy(
    out: SMCOutput,
    N: int,
    *,
    threshold_fraction: float = PARTICLE_ESS_FRACTION,
) -> float:
    """Convenience wrapper: particle-degeneracy rate for one SMC pass."""
    return particle_degeneracy_rate(
        out.ess_per_step, N, threshold_fraction=threshold_fraction,
    )


# ---------------- iPMCMC report ----------------

@dataclass
class iPMCMCDiagnostics:
    """Convergence summary for an iPMCMCOutput (post-burn-in)."""
    rhat: dict[str, float]              # R-hat per phi component
    ess_bulk: dict[str, float]          # bulk ESS per phi component
    rhat_max: float                     # worst R-hat across phi
    ess_bulk_min: float                 # headline min ESS (decision #13)
    rhat_flagged: list[str]             # phi components with R-hat > 1.01
    # θ_w aggregated across wallets
    rhat_theta_w_max: float
    ess_bulk_theta_w_min: float


def diagnose_ipmcmc(
    output: iPMCMCOutput,
    n_burnin: int = 0,
) -> iPMCMCDiagnostics:
    """Compute R-hat and bulk ESS across the P conditional chains of an
    iPMCMC run, after dropping `n_burnin` iterations."""
    rhat: dict[str, float] = {}
    ess_bulk: dict[str, float] = {}
    for name in PHI_PARAM_NAMES:
        samples = getattr(output, name)[n_burnin:]   # (n_iter, P)
        rhat[name] = float(compute_rhat(samples))
        ess_bulk[name] = float(compute_ess(samples))

    finite_rhats = [v for v in rhat.values() if np.isfinite(v)]
    rhat_max = float(max(finite_rhats)) if finite_rhats else float("nan")
    ess_bulk_min = float(min(ess_bulk.values()))
    rhat_flagged = sorted(
        k for k, v in rhat.items()
        if np.isfinite(v) and v > RHAT_FLAG_THRESHOLD
    )

    theta = output.theta_w[n_burnin:]               # (n_iter, P, n_wallets)
    rhat_theta = compute_rhat(theta)
    ess_theta = compute_ess(theta)
    rhat_theta_w_max = float(np.nanmax(rhat_theta))
    ess_bulk_theta_w_min = float(np.nanmin(ess_theta))

    return iPMCMCDiagnostics(
        rhat=rhat, ess_bulk=ess_bulk,
        rhat_max=rhat_max, ess_bulk_min=ess_bulk_min,
        rhat_flagged=rhat_flagged,
        rhat_theta_w_max=rhat_theta_w_max,
        ess_bulk_theta_w_min=ess_bulk_theta_w_min,
    )
