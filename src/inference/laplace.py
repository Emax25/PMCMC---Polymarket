"""Laplace / curvature Gaussian over the unconstrained VEM parameter vector phi.

VEM's M-step is a *point* estimator, but the samplerless validation ladder
(PSIS, SBC, coverage — plans 2 and 3) needs a *distribution* over parameters.
This module supplies the cheapest defensible such object: a block-diagonal
Gaussian on the unconstrained scale, centred at the VEM estimate, whose
covariance is the inverse curvature of the M-step's own expected complete-data
objective at that estimate (plan 2026-07-23-002 KTD1/R1/R2).

Parameter vector (canonical order, the constrained/natural scale):

    phi = (sigma2_0, sigma2_1, q_01, q_10, beta_S, beta_Z, tau2_0, tau2_1)

Unconstrained reparameterization ``u = g(phi)`` (so a Gaussian on ``u`` respects
the parameters' support):
    * variances (sigma2_*, tau2_*):  u = log(phi),        phi = exp(u)
    * transitions (q_*):             u = logit(phi),       phi = sigmoid(u)
    * standardized betas:            u = phi               (identity)

Block curvature at the VEM optimum (all as *precisions*, i.e. negative Hessians
of the M-step's expected complete-data log-posterior, delta-method-mapped to the
unconstrained scale and evaluated at the returned estimate):

    * log(sigma2_v):  (SS_v/2 + prior.sigma2_ig_beta) / sigma2_v
                      (= alpha + N_v/2 + 1 at the interior Inverse-Gamma mode)
    * log(tau2_z):    (SS_z/2 + prior.tau2_ig_beta) / tau2_z
                      (= N_z/2 + prior.tau2_ig_alpha at the pseudo-count mode)
    * logit(q):       (a + b - 2 + n_switch + n_stay) * q * (1 - q)   [binomial]
    * (beta_S,beta_Z): the 2x2 IRLS Fisher information already on VEMOutput
                       (data + Cauchy-prior curvature, internal scale = identity)

Cross-block covariance is set to zero (block-diagonal). This is an honest,
documented simplification — the M-step is an ECM sweep that updates the blocks
in sequence, not jointly, so it never forms the cross-curvature; whether the
missing correlations matter is exactly what SBC/coverage (plan 3) test.

The variance/transition precisions depend only on the E-step sufficient
statistics and the returned parameter values, so the builder re-runs one ADF
E-step pass per market at the fitted (params, theta_w) to recover them — this is
a cold analysis-side path, so correctness is preferred over reusing the M-step's
internal buffers. The block algebra mirrors ``_vem_m_step`` exactly (see the
per-block comments).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit
from scipy.special import logit as _logit

from config.default_params import PhiPrior
from src.inference.particle_gibbs import MarketData
from src.inference.variational_em import VEMOutput, _vem_e_step

# Canonical constrained-parameter order and the per-dimension transform groups.
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
_VAR_IDX = [0, 1, 6, 7]  # log-transformed (positive variances)
_Q_IDX = [2, 3]          # logit-transformed (probabilities in (0, 1))
_BETA_IDX = [4, 5]       # identity (already unconstrained, internal scale)

# Curvature-fallback constants (R3). Jitter is added relative to a block's own
# scale; the per-dimension precision floor yields a wide-but-finite variance so
# a degenerate block never crashes sampling/logpdf.
_JITTER = 1e-10
_MIN_PRECISION = 1e-8


def _to_unconstrained(phi: np.ndarray) -> np.ndarray:
    """Map constrained phi to the unconstrained scale (log / logit / identity).

    Args:
        phi: Constrained parameter vector(s), shape ``(..., 8)`` in ``_PHI_DIMS``
            order.

    Returns:
        The unconstrained vector(s) ``u``, same shape.
    """
    phi = np.asarray(phi, dtype=float)
    u = phi.copy()
    u[..., _VAR_IDX] = np.log(phi[..., _VAR_IDX])
    u[..., _Q_IDX] = _logit(phi[..., _Q_IDX])
    return u


def _to_constrained(u: np.ndarray) -> np.ndarray:
    """Invert ``_to_unconstrained`` (exp / sigmoid / identity).

    Args:
        u: Unconstrained vector(s), shape ``(..., 8)`` in ``_PHI_DIMS`` order.

    Returns:
        The constrained parameter vector(s) ``phi``, same shape.
    """
    u = np.asarray(u, dtype=float)
    phi = u.copy()
    phi[..., _VAR_IDX] = np.exp(u[..., _VAR_IDX])
    phi[..., _Q_IDX] = expit(u[..., _Q_IDX])
    return phi


def _log_abs_du_dphi(phi: np.ndarray) -> np.ndarray:
    """Log absolute Jacobian ``sum_i log|du_i/dphi_i|`` of the forward transform.

    Turns the unconstrained-scale Gaussian density into a proper density on the
    constrained scale, so ``logpdf`` (constrained input) is consistent with
    ``sample`` (constrained output) and with ``PhiPrior.log_prior`` (also
    constrained) for PSIS ratios.

    Args:
        phi: Constrained parameter vector(s), shape ``(..., 8)``.

    Returns:
        The summed log-Jacobian, shape ``phi.shape[:-1]``.
    """
    phi = np.asarray(phi, dtype=float)
    terms = np.zeros(phi.shape)
    # d log(x)/dx = 1/x  ->  log|.| = -log(x)
    terms[..., _VAR_IDX] = -np.log(phi[..., _VAR_IDX])
    # d logit(q)/dq = 1/(q(1-q))  ->  log|.| = -log(q) - log(1-q)
    q = phi[..., _Q_IDX]
    terms[..., _Q_IDX] = -np.log(q) - np.log1p(-q)
    # betas: identity -> 0
    return terms.sum(axis=-1)


def _is_pd(mat: np.ndarray) -> bool:
    """Return True if ``mat`` is symmetric positive-definite (Cholesky test)."""
    try:
        np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False


def _block_cov_from_precision(
    prec: np.ndarray, fallback_prec_diag: float
) -> tuple[np.ndarray, bool]:
    """Invert a precision block to a covariance, with the R3 fallback ladder.

    Ladder: (1) invert directly if positive-definite; (2) else add scale-relative
    jitter to the diagonal and invert; (3) else replace with a per-dimension
    diagonal precision floored at ``fallback_prec_diag`` (a wide-but-finite,
    prior-informed curvature) and invert. Steps 2-3 flag the block.

    Args:
        prec: Symmetric precision (negative-Hessian) block, shape ``(d, d)``.
        fallback_prec_diag: Per-dimension precision floor for the last resort —
            for the beta block this is the Cauchy prior curvature ``2/scale**2``;
            for scalar blocks the generic ``_MIN_PRECISION``.

    Returns:
        ``(cov, used_fallback)``: the covariance block and whether a fallback ran.
    """
    if _is_pd(prec):
        return np.linalg.inv(prec), False
    # (2) jitter — but only when the block carries real curvature to stabilize;
    # an all-(near-)zero block (e.g. betas when estimate_betas=False) should skip
    # straight to the informed per-dimension floor rather than invert pure jitter.
    if np.any(np.abs(prec) > _JITTER):
        scale = max(float(np.trace(prec)) / prec.shape[0], 1.0)
        jittered = prec + _JITTER * scale * np.eye(prec.shape[0])
        if _is_pd(jittered):
            return np.linalg.inv(jittered), True
    # (3) per-dimension curvature floor.
    prec_diag = np.maximum(np.abs(np.diag(prec)), fallback_prec_diag)
    return np.diag(1.0 / prec_diag), True


@dataclass
class PhiPosterior:
    """Block-diagonal Laplace Gaussian over the unconstrained parameter phi.

    Attributes:
        mean_u: Unconstrained-scale mean, shape ``(8,)``, in ``dims`` order.
        cov_u: Unconstrained-scale covariance, shape ``(8, 8)``, block-diagonal.
        dims: Names of the 8 parameters in canonical order.
        curvature_fallback: True if any block needed the R3 jitter/per-dimension
            fallback (its curvature was non-PD or degenerate).
        fallback_dims: Names of the parameters whose block used a fallback.
    """

    mean_u: np.ndarray
    cov_u: np.ndarray
    dims: tuple[str, ...] = _PHI_DIMS
    curvature_fallback: bool = False
    fallback_dims: tuple[str, ...] = ()
    _chol: np.ndarray = field(default=None, repr=False, compare=False)
    _prec: np.ndarray = field(default=None, repr=False, compare=False)
    _logdet: float = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Cache the Cholesky, precision, and log-determinant for reuse."""
        self.mean_u = np.asarray(self.mean_u, dtype=float)
        self.cov_u = np.asarray(self.cov_u, dtype=float)
        self._chol = np.linalg.cholesky(self.cov_u)
        self._prec = np.linalg.inv(self.cov_u)
        _, self._logdet = np.linalg.slogdet(self.cov_u)

    def to_unconstrained(self, phi: np.ndarray) -> np.ndarray:
        """Transform constrained phi to the unconstrained scale."""
        return _to_unconstrained(phi)

    def to_constrained(self, u: np.ndarray) -> np.ndarray:
        """Transform unconstrained u back to the constrained scale."""
        return _to_constrained(u)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` constrained parameter vectors from the Laplace posterior.

        Draws ``u ~ N(mean_u, cov_u)`` and back-transforms, so every returned
        variance is positive and every ``q`` lies in ``(0, 1)`` by construction.

        Args:
            rng: Explicit NumPy generator (never the global RNG).
            n: Number of draws.

        Returns:
            Constrained draws, shape ``(n, 8)`` in ``dims`` order.
        """
        z = rng.standard_normal((n, self.mean_u.size))
        u = self.mean_u + z @ self._chol.T
        return _to_constrained(u)

    def logpdf(self, phi: np.ndarray) -> np.ndarray:
        """Log density of the posterior at constrained phi (constrained scale).

        Evaluates the unconstrained Gaussian at ``u = g(phi)`` and adds the
        change-of-variables Jacobian, giving a proper density on the constrained
        space consistent with ``sample`` and with ``PhiPrior.log_prior``.

        Args:
            phi: Constrained parameter vector(s), shape ``(..., 8)``.

        Returns:
            ``log q(phi)``, shape ``phi.shape[:-1]``.
        """
        phi = np.asarray(phi, dtype=float)
        u = _to_unconstrained(phi)
        d = u - self.mean_u
        k = self.mean_u.size
        quad = np.einsum("...i,ij,...j->...", d, self._prec, d)
        gauss = -0.5 * (k * np.log(2.0 * np.pi) + self._logdet + quad)
        return gauss + _log_abs_du_dphi(phi)


def _recompute_mstep_stats(
    markets: list[MarketData],
    q_vz_list: list[np.ndarray],
    mu_filt_list: list[np.ndarray],
    sigma2_filt_list: list[np.ndarray],
    gamma: float,
) -> dict[str, float | list[float]]:
    """Recompute the M-step sufficient statistics from an ADF E-step pass.

    Mirrors the three stat sweeps in ``_vem_m_step`` exactly (process-variance
    ``SS_v/N_v``, transition counts ``n_ij``, observation-variance ``SS_z/N_z``)
    so the Laplace curvature is evaluated against the very objective the M-step
    optimized. Recomputed here rather than reused to keep ``_vem_m_step`` free of
    Laplace-specific bookkeeping.

    Args:
        markets: Input market data.
        q_vz_list: Per-market soft (V, Z) assignments from the re-run E-step.
        mu_filt_list: Per-market mixed Kalman means from the re-run E-step.
        sigma2_filt_list: Per-market mixed Kalman variances from the re-run E-step.
        gamma: Size-informativeness scaling (from the fitted params).

    Returns:
        Dict with ``SS_v``/``N_v`` (length-2 lists over regimes), the four
        transition counts ``n_00/n_01/n_10/n_11``, and ``SS_z``/``N_z``.
    """
    SS_v = [0.0, 0.0]
    N_v = [0.0, 0.0]
    n_00 = n_01 = n_10 = n_11 = 0.0
    SS_z = [0.0, 0.0]
    N_z = [0.0, 0.0]
    for md, q_vz, mu_f, sigma2_f in zip(
        markets, q_vz_list, mu_filt_list, sigma2_filt_list
    ):
        # Process-variance stats (trades j >= 1 with a positive time gap).
        dt = md.delta[1:]
        valid = dt > 0
        if valid.any():
            resid2 = (mu_f[1:] - mu_f[:-1]) ** 2
            extra_var = sigma2_f[1:] + sigma2_f[:-1]
            for v in (0, 1):
                q_V_v = (q_vz[:, 2 * v] + q_vz[:, 2 * v + 1])[1:]
                contrib = (
                    q_V_v[valid] * (resid2[valid] + extra_var[valid]) / dt[valid]
                )
                SS_v[v] += float(contrib.sum())
                N_v[v] += float(q_V_v[valid].sum())

        # Transition counts from the product of consecutive V-marginals.
        q_V0 = q_vz[:, 0] + q_vz[:, 1]
        q_V1 = q_vz[:, 2] + q_vz[:, 3]
        n_00 += float((q_V0[:-1] * q_V0[1:]).sum())
        n_01 += float((q_V0[:-1] * q_V1[1:]).sum())
        n_10 += float((q_V1[:-1] * q_V0[1:]).sum())
        n_11 += float((q_V1[:-1] * q_V1[1:]).sum())

        # Observation-variance stats over all trades (size-scaled residuals).
        denom_t = np.maximum(1.0 + md.log_size_ratio * gamma, 0.1)
        resid2_obs = (md.Y - mu_f) ** 2
        for z in (0, 1):
            q_Z_z = q_vz[:, z] + q_vz[:, 2 + z]
            SS_z[z] += float((q_Z_z * (resid2_obs + sigma2_f) * denom_t).sum())
            N_z[z] += float(q_Z_z.sum())

    return {
        "SS_v": SS_v,
        "N_v": N_v,
        "n_00": n_00,
        "n_01": n_01,
        "n_10": n_10,
        "n_11": n_11,
        "SS_z": SS_z,
        "N_z": N_z,
    }


def laplace_from_vem(
    vem_output: VEMOutput,
    markets: list[MarketData],
    prior: PhiPrior | None = None,
) -> PhiPosterior:
    """Assemble the block-diagonal Laplace posterior over phi at the VEM optimum.

    Re-runs one ADF E-step per market at the fitted ``(params, theta_w)`` to
    recover the M-step sufficient statistics, then builds each block's
    unconstrained-scale precision (see the module docstring for the algebra) and
    inverts it with the R3 fallback ladder. The beta block reuses the IRLS Fisher
    information carried on ``vem_output`` (which already blends data and
    Cauchy-prior curvature); when ``estimate_betas`` was False that Fisher is
    zero and the fallback substitutes the Cauchy prior curvature ``2/scale**2``.

    Args:
        vem_output: A fitted ``VEMOutput`` (params, theta_w, standardization
            constants, and ``beta_fisher_info``).
        markets: The markets the VEM was fit on, for the E-step re-run.
        prior: The M-step prior spec used to fit ``vem_output``; ``None`` uses
            ``PhiPrior()`` defaults. Must match the prior VEM used, so the
            curvature is taken against the same objective.

    Returns:
        A ``PhiPosterior`` over the 8-vector phi.
    """
    if prior is None:
        prior = PhiPrior()
    params = vem_output.params

    # Re-run the E-step at the fitted point to recover q(V, Z) and the filtered
    # moments feeding the M-step sufficient statistics.
    q_vz_list: list[np.ndarray] = []
    mu_filt_list: list[np.ndarray] = []
    sigma2_filt_list: list[np.ndarray] = []
    for md in markets:
        q_vz, mu_f, sigma2_f, _ = _vem_e_step(
            md.Y,
            md.delta,
            md.log_size_ratio,
            md.wallet_ids,
            vem_output.theta_w,
            params,
            vem_output.m_S,
            vem_output.s_S,
            vem_output.m_Z,
        )
        q_vz_list.append(q_vz)
        mu_filt_list.append(mu_f)
        sigma2_filt_list.append(sigma2_f)

    stats = _recompute_mstep_stats(
        markets, q_vz_list, mu_filt_list, sigma2_filt_list, params.gamma
    )
    SS_v, N_v = stats["SS_v"], stats["N_v"]
    SS_z, N_z = stats["SS_z"], stats["N_z"]

    # Per-block unconstrained-scale precisions (negative Hessians at the VEM
    # estimate). See the module docstring for each derivation.
    prec_sigma2_0 = (0.5 * SS_v[0] + prior.sigma2_ig_beta) / params.sigma2_0
    prec_sigma2_1 = (0.5 * SS_v[1] + prior.sigma2_ig_beta) / params.sigma2_1
    prec_tau2_0 = (0.5 * SS_z[0] + prior.tau2_ig_beta) / params.tau2_0
    prec_tau2_1 = (0.5 * SS_z[1] + prior.tau2_ig_beta) / params.tau2_1
    # Binomial info on the logit scale: (effective count) * q(1-q). The Beta(a,b)
    # prior contributes a-1 / b-1 pseudo-counts (zero for the Beta(1,1) default).
    q_count = prior.q_beta_a + prior.q_beta_b - 2.0
    prec_q_01 = (q_count + stats["n_01"] + stats["n_00"]) * params.q_01 * (
        1.0 - params.q_01
    )
    prec_q_10 = (q_count + stats["n_10"] + stats["n_11"]) * params.q_10 * (
        1.0 - params.q_10
    )

    cov = np.zeros((8, 8))
    fallback_dims: list[str] = []
    scalar_blocks = {
        0: prec_sigma2_0,
        1: prec_sigma2_1,
        2: prec_q_01,
        3: prec_q_10,
        6: prec_tau2_0,
        7: prec_tau2_1,
    }
    for idx, prec in scalar_blocks.items():
        block_cov, used_fb = _block_cov_from_precision(
            np.array([[prec]]), _MIN_PRECISION
        )
        cov[idx, idx] = block_cov[0, 0]
        if used_fb:
            fallback_dims.append(_PHI_DIMS[idx])

    # Beta block: 2x2 IRLS Fisher information (identity = unconstrained scale).
    fisher = np.asarray(vem_output.beta_fisher_info, dtype=float)
    beta_cov, used_fb = _block_cov_from_precision(
        fisher, 2.0 / prior.beta_cauchy_scale**2
    )
    cov[np.ix_(_BETA_IDX, _BETA_IDX)] = beta_cov
    if used_fb:
        fallback_dims.extend(_PHI_DIMS[i] for i in _BETA_IDX)

    mean_u = _to_unconstrained(
        np.array(
            [
                params.sigma2_0,
                params.sigma2_1,
                params.q_01,
                params.q_10,
                params.beta_S,
                params.beta_Z,
                params.tau2_0,
                params.tau2_1,
            ]
        )
    )

    return PhiPosterior(
        mean_u=mean_u,
        cov_u=cov,
        curvature_fallback=bool(fallback_dims),
        fallback_dims=tuple(fallback_dims),
    )
