"""Curvature Gaussian over the unconstrained VEM parameter vector phi.

VEM's M-step is a *point* estimator, but the samplerless validation ladder
(PSIS, SBC, coverage — plans 2 and 3) needs a *distribution* over parameters.
This module supplies the cheapest such object: a block-diagonal Gaussian on the
unconstrained scale, centred at the VEM/ECM fixed point, whose covariance is the
inverse *expected complete-data* (ECM) curvature at that point
(plan 2026-07-23-002 KTD1/R1/R2).

What this object is NOT
-----------------------
Despite the module name, this is **not** a Laplace approximation to the
posterior ``p(phi | Y)``. Two distinct gaps, both measured at dev scale:

  1. *Wrong information matrix.* The blocks below are expected complete-data
     information, not observed information. By the Louis (1982) missing-
     information identity, observed = complete minus missing, so the complete-
     data curvature systematically *over*-states precision. Measured on the
     standard fixture: at ``log sigma2_0`` the observed information is 2.24
     against this object's 252.3 — 113x over-precise.
  2. *Wrong centre.* A VEM fixed point is stationary for the variational
     objective, not for the target ``log p(Y|phi) + log p(phi)`` that PSIS and
     SBC weight against. Run to convergence (1500 iterations, relative change
     1.3e-8) the target's gradient at the centre is still ~10 Laplace sd along
     ``log sigma2_0``, and the observed information at ``tau2_1`` is *negative*
     (-3.96) — i.e. the centre is a local *minimum* along that axis, which no
     Laplace approximation can be.

Consequences: PSIS khat computed against this proposal diagnoses the *proposal*,
not the model, and a poor khat is expected rather than evidence of model
misfit; SBC coverage built on it inherits both the centre offset and the
over-precision. Any such claim must be qualified accordingly. Fixing this needs
either Louis-identity observed information or a direct optimum of the marginal
target — neither is in the current layer.

Known limitation: the order constraint binds
--------------------------------------------
The M-step enforces ``sigma2_1 = max(sigma2_1, sigma2_0)``. On the standard
fixture that constraint is *active at every fitted point* — ``sigma2_0`` equals
``sigma2_1`` to the last bit on 5/5 dev restarts, and still at convergence. The
sigma2 blocks' curvature is therefore evaluated on a constraint boundary, where
the unconstrained quadratic below is simply the wrong local model, and ~75% of
draws from the resulting Gaussian violate the estimator's own order constraints.
Relatedly the V regime is non-identified at that point: the ADF log-marginal
moves less than 5e-13 over +-4 sd in both ``logit q_01`` and ``logit q_10``,
while this object assigns those blocks precisions of ~106 and ~19. Both are
known limitations of this layer, not of the fallback ladder.

Parameter vector (canonical order, the constrained/natural scale):

    phi = (sigma2_0, sigma2_1, q_01, q_10, beta_S, beta_Z, tau2_0, tau2_1)

Unconstrained reparameterization ``u = g(phi)`` (so a Gaussian on ``u`` respects
the parameters' support):
    * variances (sigma2_*, tau2_*):  u = log(phi),        phi = exp(u)
    * transitions (q_*):             u = logit(phi),       phi = sigmoid(u)
    * standardized betas:            u = phi               (identity)

Block curvature at the ECM fixed point (all as *precisions*, i.e. negative
Hessians of the M-step's expected complete-data log-posterior — see the caveats
above — delta-method-mapped to the unconstrained scale and evaluated at the
returned estimate):

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
internal buffers. The statistics themselves come from
``variational_em._mstep_sufficient_stats``, the same helper the M-step calls, so
the block algebra below is guaranteed to be curvature of the objective the
estimator actually optimized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit
from scipy.special import logit as _logit

from config.default_params import PhiPrior
from src.inference.particle_gibbs import MarketData
from src.inference.variational_em import (
    VEMOutput,
    _mstep_sufficient_stats,
    _vem_e_step,
)

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

# Width cap for the *scalar* blocks (log-variances and logit-probabilities).
# A 1x1 precision is positive-definite for any positive value, so the sign test
# alone cannot catch a block whose data-driven curvature has vanished; the
# variance itself has to be bounded. 4.0 on the log/logit scale is already an
# extreme width: on log(sigma2)/log(tau2) a +-2 sd interval spans a factor of
# exp(16) ~ 9e6 in the variance, and on logit(q) it puts over 99.9% of the mass
# outside q in (3e-4, 1 - 3e-4). It is also ~7x the widest well-conditioned
# scalar sd observed on the dev fixture (~0.55 for logit q_10 at T=300), so
# well-identified blocks never touch it. Anything wider carries no information
# and only risks exp() overflow when draws are back-transformed.
_MAX_SCALAR_SD_U = 4.0
_MAX_SCALAR_VAR_U = _MAX_SCALAR_SD_U**2


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
    """Invert a multivariate precision block to a covariance (R3 fallback ladder).

    Ladder: (1) invert directly if positive-definite; (2) else add scale-relative
    jitter to the diagonal and invert; (3) else replace with a per-dimension
    diagonal precision floored at ``fallback_prec_diag`` (a wide-but-finite,
    prior-informed curvature) and invert. Steps 2-3 flag the block.

    Used for the 2x2 beta block only. Scalar blocks go through
    ``_scalar_block_variance`` instead, because positive-definiteness of a 1x1
    precision is a vacuous test (see that function's docstring).

    Args:
        prec: Symmetric precision (negative-Hessian) block, shape ``(d, d)``.
        fallback_prec_diag: Per-dimension precision floor for the last resort —
            for the beta block this is the Cauchy prior curvature ``2/scale**2``.

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


def _scalar_block_variance(prec: float) -> tuple[float, bool]:
    """Convert a scalar curvature to a sane unconstrained-scale variance (R3).

    Scalar blocks need a stronger guard than the positive-definiteness test used
    for the 2x2 beta block: a 1x1 precision is "PD" for *any* positive value,
    including the purely prior-driven remnant left when a regime is empty. With
    ``SS_z = N_z = 0`` the tau2 curvature collapses to ``tau2_ig_beta / tau2``
    ~ 1e-9, which passes a sign test unflagged but implies a log-scale sd of
    ~3e4; back-transforming such draws overflows ``exp`` to 0/inf and sends both
    ``PhiPosterior.logpdf`` and ``PhiPrior.log_prior`` non-finite. So the
    *returned variance* is capped at ``_MAX_SCALAR_VAR_U`` (see that constant
    for the width justification), not merely checked for sign, and any block
    that hits the cap is reported through ``fallback_dims``.

    Args:
        prec: Scalar unconstrained-scale precision (negative Hessian) for one
            of the log-variance or logit-transition blocks.

    Returns:
        ``(var, used_fallback)``: the unconstrained-scale variance and whether
        the sign floor or the width cap had to intervene.
    """
    prec = float(prec)
    if not np.isfinite(prec) or prec <= 0.0:
        # Non-PD (or undefined) curvature: keep its magnitude as an abs-Hessian
        # proxy, floored so the inverse stays finite. Mirrors step (3) of
        # `_block_cov_from_precision` for the 1x1 case.
        magnitude = abs(prec) if np.isfinite(prec) else 0.0
        var = 1.0 / max(magnitude, _MIN_PRECISION)
        return min(var, _MAX_SCALAR_VAR_U), True
    var = 1.0 / prec
    if var > _MAX_SCALAR_VAR_U:
        return _MAX_SCALAR_VAR_U, True
    return var, False


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

    # Same helper the M-step itself calls, so the curvature is evaluated against
    # the very statistics the estimator optimized rather than a second copy of
    # the algebra that could drift from it.
    stats = _mstep_sufficient_stats(
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
        cov[idx, idx], used_fb = _scalar_block_variance(prec)
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
