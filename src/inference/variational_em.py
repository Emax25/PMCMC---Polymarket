"""Variational EM (ADF + moment-matched/IRLS M-step) for the switching SSM.

Replaces CSMC Monte Carlo with deterministic assumed-density filtering:
one forward pass per market per EM iteration, O(4T) per market vs O(NT)
for CSMC with N particles. Suitable for fast wallet-ranking when exact
posterior uncertainty is not required.

E-step: single-mode ADF forward pass - 4 (V,Z) combos share one incoming
        Kalman state (the mixture mean/variance from the previous step).
        This collapses the path structure but keeps the algorithm O(T).

M-step is an ECM sweep (Meng & Rubin 1993) of three ordered blocks; each
block maximizes the expected complete-data log-posterior holding the other
two fixed, so — *provided the objective is the same one the E-step scored*
(it is not; see the m_Z caveat below) — the (approximate) EM objective is
monotone *blockwise* within an iteration even though the blocks are not
updated jointly:
  (a) theta_w  - offset-adjusted per-wallet penalized Newton on
      logit(theta_w[w]), using the *previous* iteration's (beta_S, beta_Z)
      as a fixed per-trade offset. The Beta(a, b) prior is placed on
      logit(theta_w) via the exact change-of-variables Jacobian, which is
      equivalent to "a prior successes / b prior failures" pseudo-counts in
      logit space. With the offset uniformly zero (beta_S = beta_Z = 0)
      this reduces exactly to the closed-form Beta-Bernoulli conjugate mean
      alpha_w / (alpha_w + beta_w) of the original count-based update — the
      new update is a strict generalization, not a parallel code path.
  (b) q_01, q_10, sigma2_*, tau2_*  - unchanged moment-matched/IG updates.
  (c) beta_S, beta_Z  - pooled IRLS over all markets (trades j >= 1 only;
      Z_0 := 0 excludes trade 0), offset by logit(theta_w) at the freshly
      updated block-(a) posterior means, with a Cauchy(0, 2.5) prior on each
      standardized coefficient via the Gelman et al. (2008, §3)
      approximate-EM modification: at each IRLS step the prior's curvature
      is evaluated at the *current* beta, which keeps the estimate finite
      even under complete separation of q(Z) on a covariate.

Block (c) is *opt-in*: `variational_em` defaults to `estimate_betas=False`
(betas held fixed). On the current synthetic generator the ADF E-step cannot
identify Z (q(Z) is near-flat — Z modulates only the observation variance
tau2_Z), so default-on beta estimation fits a spurious size-correlated beta_S
(~-0.40) that drops gate AUC from ~0.89 to ~0.68. Beta estimation is enabled
explicitly (Laplace layer, real-data runs, Z-identifiability work) until the
E-step identifies Z.

Caveat (blockwise monotonicity is conditional, and currently latent): the
guarantee above holds only while the E-step and the M-step score the *same*
objective, i.e. with the covariate standardization constants held fixed
across the iteration. `m_Z` is not: it is refreshed from the fresh q(Z)
*between* the E-step and the same iteration's M-step call, so `q_vz_list`
was produced under the old `m_Z` while block (c) re-centers `x_Z~` under the
new one — the objective is redefined mid-iteration and the log-marginal
trace therefore carries no monotonicity guarantee. The effect is inert on
every committed artifact because block (c) is opt-in and off by default
(`estimate_betas=False` pins `beta_Z = 0`): measured induced level shift
1.7e-5, with `m_Z` drifting 0.117 -> 0.055 over 15 iterations (largest
per-iteration change 0.0124). At the oracle `beta_Z = 1.5` that same drift
would move the logistic predictor by ~0.019 logit per iteration, absorbed by
`theta_w` as a level; and with `estimate_betas=True` the log-marginal trace
was measured non-monotone (-1065.6 -> -1082.1 -> -1126.3, then rising).
Moving the refresh after the M-step is an open scope decision, not an
oversight — nothing here reorders it.

Model modes. Everything above describes the wallet-anchored (Polymarket)
model. The **anonymous** mode (`ModelParams.anonymous`, for feeds like Kalshi's
that publish no per-account identifier) drops block (a) entirely — there is no
theta_w — and replaces the predictor's `logit(theta_w[w])` level with a single
per-market intercept `alpha`, fitted as a third IRLS column alongside the
slopes. The intercept is identified precisely because theta_w is gone; the
price is that it is far more weakly shrunk than Beta(1, 19) ever shrank
theta_w, so short markets carry an intercept bias (see
`PhiPrior.alpha_cauchy_scale`). Because `alpha` rides on block (c), the
`estimate_betas=False` default freezes it too — an anonymous run that wants its
level fitted must opt in.

Predictor covariates are centered/standardized (Gelman et al. 2008) before
entering the logistic Z predictor, with no free intercept in wallet mode — the
theta_w Beta hierarchy already carries the level:
    x_S~ = (log_size_ratio - m_S) * 0.5 / s_S   (mean 0, sd 0.5)
    x_Z~ = E[Z_prev] - m_Z                       (centered only)
`(m_S, s_S, m_Z)` are fit once per `variational_em` call (m_S, s_S pooled
over all markets' log_size_ratio; m_Z a running pooled mean of E[Z_prev]
updated after each E-step) and stored on `VEMOutput`. Standardizing keeps
`beta_S`/`beta_Z` on a common, interpretable scale, but the resulting slopes
are only *approximately* centering-invariant: wallet-specific theta_w
shrinkage is heavy below ~20 trades (ARCHITECTURE.md §9.5), so the logistic
predictor and the Beta hierarchy interact rather than factoring cleanly.

Caveat (regression dilution): x_Z~ plugs in the *filtered* E[Z_prev] rather
than the true (unobserved) Z_prev. This mean-field/plug-in treatment of the
lagged covariate is a known source of attenuation bias in errors-in-variables
regression — `beta_Z` should be expected to systematically *underestimate*
the data-generating slope, worse at shorter T where E[Z_prev] is noisier.

`ModelParams.beta_S` / `beta_Z` are on the *internal* (standardized) scale
consumed directly by the E-step's logistic predictor above. `VEMOutput`
additionally reports `beta_S_orig` / `beta_Z_orig`, back-transformed to the
original `log_size_ratio` units, for interpretation.

Reference: Ghahramani & Hinton (2000) "Variational Learning for Switching
State-Space Models"; also known as Assumed Density Filtering for SSMs.
Gelman, A. (2008) "Scaling regression inputs by dividing by two standard
deviations", Statistics in Medicine 27(15).
Gelman, A., Jakulin, A., Pittau, M.G., Su, Y.S. (2008) "A weakly informative
default prior distribution for logistic and other regression models",
Annals of Applied Statistics 2(4) — §3's approximate-EM IRLS modification
for a Cauchy prior.
Meng, X.L. & Rubin, D.B. (1993) "Maximum likelihood estimation via the ECM
algorithm: A general framework", Biometrika 80(2).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from config.default_params import InferenceConfig, ModelParams, PhiPrior
from src.inference.adf_filter import S_STD_FLOOR, ADFFilter
from src.inference.particle_gibbs import MarketData
from src.utils.transforms import log1pexp, logit, sigmoid

# The Cauchy(0, scale) weakly-informative prior on each standardized beta
# (Gelman et al. 2008) now lives in `config.default_params.PhiPrior`
# (`beta_cauchy_scale`, default 2.5) so every consumer — the IRLS M-step here,
# the Laplace layer, and PSIS — shares one definition (plan 2026-07-23-002 R8).
# The approximate-EM IRLS modification below still evaluates the prior's
# curvature/gradient contribution at the current coefficient each step.
_IRLS_MAX_ITER = 25
_IRLS_REL_TOL = 1e-6

# theta_w's per-wallet update is a strictly concave 1-D problem (the Beta
# prior's pseudo-count curvature (a+b)*sigmoid(phi)(1-sigmoid(phi)) never
# vanishes), so it has a unique finite mode. Strict concavity does NOT make
# an *undamped* Newton step globally convergent, though: from a cold phi_init
# a full step overshoots on a high-count wallet and can diverge to a NaN, so
# each step is backtracked (halved) until the objective does not decrease. A
# tight tolerance is cheap and lets the beta=0 reduction to the closed-form
# conjugate mean hold to high precision.
_THETA_W_MAX_ITER = 25
_THETA_W_REL_TOL = 1e-10
_THETA_W_MAX_HALVINGS = 40


@dataclass
class VEMOutput:
    """Output of variational_em: deterministic posterior summaries + fitted params."""

    params: ModelParams
    theta_w: np.ndarray          # (n_wallets,) posterior mean of theta_w (probability scale)
    Z_prob: list[np.ndarray]     # per-market (T_k,) q(Z_t=1) = filter-marginal P(Z_t=1|Y)
    V_prob: list[np.ndarray]     # per-market (T_k,) q(V_t=1) = filter-marginal P(V_t=1|Y)
    X_mean: list[np.ndarray]     # per-market (T_k,) mixed E[X_t | Y_{0:t}]
    elbo_trace: np.ndarray       # (n_iter_run,) log-marginal per EM iteration (proxy for ELBO)
    n_iter_run: int              # actual EM iterations completed
    m_S: float                   # pooled mean of log_size_ratio (standardization)
    s_S: float                   # pooled std of log_size_ratio; ~0 if degenerate
                                  # (constant size) -- see S_STD_FLOOR
    m_Z: float                   # running pooled mean of E[Z_prev] (centering)
    theta_w_logit_mean: np.ndarray  # (n_wallets,) logit-normal posterior mean = Newton mode
    theta_w_logit_var: np.ndarray   # (n_wallets,) logit-normal posterior var = 1/curvature
    beta_S_orig: float           # beta_S back-transformed to original log_size_ratio units
    beta_Z_orig: float           # == beta_Z (centering-only transform has unit slope)
    beta_fisher_info: np.ndarray  # (2, 2) final IRLS Fisher info (internal scale, incl.
                                  # prior curvature); consumed by the Laplace layer
                                  # (plan 2). (3, 3) over (alpha, beta_S, beta_Z) in
                                  # anonymous mode -- the Laplace layer is wallet-only.
    alpha_orig: float = 0.0      # anonymous-mode intercept undone back to raw covariate
                                 # units; 0.0 and meaningless in wallet mode, which has
                                 # no intercept (see `_alpha_orig`)


def _alpha_orig(
    params: ModelParams, beta_S_orig: float, m_S: float, m_Z: float
) -> float:
    """Undo the covariate centering absorbed into the anonymous-mode intercept.

    The predictor is fit on centered covariates,

        logit(pi) = alpha + beta_S x_S~ + beta_Z x_Z~
                  = alpha + beta_S_orig (x_S - m_S) + beta_Z (x_Z - m_Z),

    so the intercept a caller can compare against a data-generating value on the
    *raw* covariates (what `src.data.synthetic` plants) is
    ``alpha - beta_S_orig * m_S - beta_Z * m_Z``. Wallet mode has no intercept
    and returns 0.0.

    Args:
        params: Fitted parameters; ``beta_Z`` is centering-only so it needs no
            rescaling.
        beta_S_orig: ``beta_S`` already back-transformed to raw log_size_ratio
            units.
        m_S: Pooled mean of log_size_ratio (standardization).
        m_Z: Pooled mean of E[Z_prev] (centering).

    Returns:
        The intercept on the raw covariate scale, or 0.0 in wallet mode.
    """
    if not params.anonymous:
        return 0.0
    return float(params.alpha - beta_S_orig * m_S - params.beta_Z * m_Z)


def _vem_e_step(
    Y: np.ndarray,
    delta: np.ndarray,
    log_size_ratio: np.ndarray,
    wallet_ids: np.ndarray,
    theta_w: np.ndarray,
    params: ModelParams,
    m_S: float,
    s_S: float,
    m_Z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Single-mode ADF forward pass for one market.

    A thin batch driver over `ADFFilter`: it owns the output arrays and the
    log-marginal accumulator, while every per-trade recursion (priors, the
    4-combo Kalman update, the assumed-density collapse) lives in
    `ADFFilter.step`, so the batch and live paths can never diverge.

    Args:
        Y: (T,) logit-price observations.
        delta: (T,) inter-trade times; delta[0] = 0.
        log_size_ratio: (T,) log(S/S_bar) features.
        wallet_ids: (T,) integer wallet index per trade.
        theta_w: (n_wallets,) current per-wallet propensity estimates.
        params: Current model parameters.
        m_S: Pooled mean of log_size_ratio, for standardizing the size covariate.
        s_S: Pooled std of log_size_ratio; at or below `S_STD_FLOOR` (degenerate
            constant-size market/dataset), the 0.5/s_S scale factor is
            skipped and only centering is applied, avoiding an unstable or
            divide-by-zero scale factor.
        m_Z: Pooled mean of E[Z_prev], for centering the persistence covariate.

    Returns:
        q_vz: (T, 4) soft (V_t, Z_t) assignments — q_vz[t, k] = q(V_t=v, Z_t=z)
              where k = 2*v + z.
        mu_filt: (T,) mixed E[X_t | Y_{0:t}].
        sigma2_filt: (T,) mixed Var[X_t | Y_{0:t}].
        log_marginal: scalar approximate log p(Y | params, theta_w).
    """
    T = len(Y)
    adf = ADFFilter(params, theta_w, m_S, s_S, m_Z)

    q_vz = np.empty((T, 4))
    mu_filt = np.empty(T)
    sigma2_filt = np.empty(T)
    log_marginal = 0.0

    for t in range(T):
        out = adf.step(Y[t], delta[t], log_size_ratio[t], wallet_ids[t])
        q_vz[t] = out.q_vz
        mu_filt[t] = out.X_mean
        sigma2_filt[t] = out.X_var
        log_marginal += out.log_evidence

    return q_vz, mu_filt, sigma2_filt, log_marginal


def _pooled_zj_covariates(
    markets: list[MarketData],
    q_vz_list: list[np.ndarray],
    m_S: float,
    s_S: float,
    m_Z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool the standardized (x_S~, x_Z~) design and q(Z) targets, trades j>=1.

    Shared by `_update_theta_w` and `update_beta_irls`: both fit the insider
    predictor on trades j >= 1 across all markets (trade 0 is excluded by the
    Z_0 := 0 convention) using the same standardized/centered covariates the
    E-step uses.

    Args:
        markets: Input market data.
        q_vz_list: Per-market soft (V,Z) assignments, each (T_k, 4).
        m_S: Pooled mean of log_size_ratio, for standardizing the size covariate.
        s_S: Pooled std of log_size_ratio; see `S_STD_FLOOR` for the
            degenerate-scale fallback.
        m_Z: Pooled mean of E[Z_prev], for centering the persistence covariate.

    Returns:
        `(wallet_idx, x_S, x_Z, y)`, each a flat array over all pooled trades:
        integer wallet index, standardized size covariate, centered lagged
        insider-probability covariate, and fractional Bernoulli target q(Z_j).
    """
    wallet_parts = []
    x_S_parts = []
    x_Z_parts = []
    y_parts = []
    for md, q_vz in zip(markets, q_vz_list):
        if len(md.Y) < 2:
            continue
        x_S_centered = md.log_size_ratio[1:].astype(float) - m_S
        x_S = x_S_centered * 0.5 / s_S if s_S > S_STD_FLOOR else x_S_centered
        E_Z = q_vz[:, 1] + q_vz[:, 3]
        wallet_parts.append(md.wallet_ids[1:])
        x_S_parts.append(x_S)
        x_Z_parts.append(E_Z[:-1] - m_Z)
        y_parts.append(E_Z[1:])

    if not wallet_parts:
        return (
            np.empty(0, dtype=int),
            np.empty(0),
            np.empty(0),
            np.empty(0),
        )
    return (
        np.concatenate(wallet_parts),
        np.concatenate(x_S_parts),
        np.concatenate(x_Z_parts),
        np.concatenate(y_parts),
    )


def _update_theta_w(
    markets: list[MarketData],
    q_vz_list: list[np.ndarray],
    n_wallets: int,
    beta_S: float,
    beta_Z: float,
    m_S: float,
    s_S: float,
    m_Z: float,
    a: float,
    b: float,
    phi_init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offset-adjusted per-wallet penalized Newton update for theta_w (R8).

    Maximizes, independently for each wallet w, the 1-D penalized objective

        J(phi_w) = sum_t [y_t (offset_t + phi_w) - log1pexp(offset_t + phi_w)]
                   + a * phi_w - (a + b) * log1pexp(phi_w)

    where `phi_w = logit(theta_w[w])`, the sum runs over wallet w's trades
    (j >= 1), `offset_t = beta_S * x_S~_t + beta_Z * x_Z~_t` is fixed (not
    optimized — the previous iteration's betas, per the ECM block order), and
    `y_t = q(Z_t = 1)`. The Beta(a, b) log-density in the last two terms is
    the exact change-of-variables of a Beta(a, b) prior on theta_w[w] onto
    logit(theta_w[w]) (the Jacobian theta(1-theta) adds exactly 1 to each
    shape parameter), which is algebraically identical to a prior of "a
    successes, b failures" pseudo-trades in logit space. This is why, with
    `offset_t` uniformly 0 (beta_S = beta_Z = 0), J reduces to
    `(a + S_w) phi_w - (a + b + S_w + F_w) log1pexp(phi_w)` and its mode maps
    via sigmoid to exactly `(a + S_w) / (a + b + S_w + F_w)` — the original
    Beta-count posterior mean.

    J is strictly concave in phi_w (curvature >= (a+b) * sigmoid(phi_w) *
    (1 - sigmoid(phi_w)) > 0 from the prior term alone), so it has a unique
    finite mode. Newton steps are backtracked per wallet (halved until the
    objective does not decrease): undamped Newton on a logistic still
    overshoots from a cold start and would otherwise diverge to a NaN on a
    high-count wallet, even though the mode itself is finite.

    Args:
        markets: Input market data.
        q_vz_list: Per-market soft (V,Z) assignments, each (T_k, 4).
        n_wallets: Total wallet count.
        beta_S: Previous iteration's internal-scale size coefficient (offset).
        beta_Z: Previous iteration's internal-scale persistence coefficient
            (offset).
        m_S: Pooled mean of log_size_ratio (standardization).
        s_S: Pooled std of log_size_ratio (standardization); see `S_STD_FLOOR`.
        m_Z: Pooled mean of E[Z_prev] (centering).
        a: Beta prior shape (successes pseudo-count).
        b: Beta prior shape (failures pseudo-count).
        phi_init: (n_wallets,) initial logit(theta_w), typically the previous
            iteration's estimate.

    Returns:
        `(theta_w, logit_mean, logit_var)`: posterior-mean theta_w on the
        probability scale (`sigmoid(logit_mean)`), the Newton-mode logit-scale
        mean, and the curvature-based logit-scale variance `1 / Hessian`.
    """
    wallet_idx, x_S, x_Z, y = _pooled_zj_covariates(markets, q_vz_list, m_S, s_S, m_Z)
    offset = beta_S * x_S + beta_Z * x_Z

    def _obj(phi_vec: np.ndarray) -> np.ndarray:
        """Per-wallet penalized objective J(phi_w), shape (n_wallets,)."""
        eta = offset + phi_vec[wallet_idx]
        data = np.zeros(n_wallets)
        np.add.at(data, wallet_idx, y * eta - log1pexp(eta))
        return data + a * phi_vec - (a + b) * log1pexp(phi_vec)

    def _grad_hess(phi_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-wallet gradient and (positive) Hessian of J at phi_vec."""
        pi_t = sigmoid(offset + phi_vec[wallet_idx])
        grad_data = np.zeros(n_wallets)
        hess_data = np.zeros(n_wallets)
        np.add.at(grad_data, wallet_idx, y - pi_t)
        np.add.at(hess_data, wallet_idx, pi_t * (1.0 - pi_t))
        sig_phi = sigmoid(phi_vec)
        grad = grad_data + a - (a + b) * sig_phi
        hess = hess_data + (a + b) * sig_phi * (1.0 - sig_phi)
        return grad, hess

    phi = np.asarray(phi_init, dtype=float).copy()
    obj = _obj(phi)
    _, hess = _grad_hess(phi)  # seeds logit_var for the n_wallets==0 / no-trade edge
    for _ in range(_THETA_W_MAX_ITER):
        grad, hess = _grad_hess(phi)
        step = grad / hess

        # Per-wallet backtracking line search. J is strictly concave (the
        # Beta prior's (a+b)*sigmoid(phi)(1-sigmoid(phi)) curvature never
        # vanishes), but an *undamped* Newton step from a cold phi_init still
        # overshoots on a high-count wallet — full Newton on a logistic can
        # diverge from a far start — driving sigmoid(phi) to a float 0/1,
        # zeroing the curvature and producing NaN. Halving each wallet's step
        # until its objective does not decrease guarantees monotone ascent to
        # the (finite) mode. Wallets converge independently, so the search is
        # per-wallet and vectorized.
        accepted = np.zeros(n_wallets, dtype=bool)
        scale = np.ones(n_wallets)
        phi_new = phi.copy()
        obj_new = obj.copy()
        for _ in range(_THETA_W_MAX_HALVINGS):
            active = ~accepted
            if not active.any():
                break
            cand = phi + scale * step
            cand_obj = _obj(cand)
            now = active & (cand_obj >= obj - 1e-12)
            phi_new[now] = cand[now]
            obj_new[now] = cand_obj[now]
            accepted |= now
            scale[active & ~now] *= 0.5

        step_taken = phi_new - phi
        phi = phi_new
        obj = obj_new
        phi_norm = max(1.0, float(np.max(np.abs(phi)))) if phi.size else 1.0
        max_step = float(np.max(np.abs(step_taken))) if step_taken.size else 0.0
        if max_step / phi_norm < _THETA_W_REL_TOL:
            break

    # Curvature at the converged mode -> logit-scale posterior variance.
    _, hess = _grad_hess(phi)
    logit_mean = phi
    logit_var = 1.0 / hess
    theta_w_new = sigmoid(logit_mean)
    return theta_w_new, logit_mean, logit_var


class IRLSFit(NamedTuple):
    """Result of `update_beta_irls`.

    Attributes:
        beta_S: Fitted internal-scale size coefficient.
        beta_Z: Fitted internal-scale persistence coefficient.
        fisher_info: Final augmented Fisher information at the converged
            coefficients (data + prior curvature). ``(2, 2)`` over
            ``(beta_S, beta_Z)`` in wallet mode; ``(3, 3)`` over
            ``(alpha, beta_S, beta_Z)`` in anonymous mode, where the intercept
            is a fitted coefficient rather than an offset.
        alpha: Fitted per-market intercept in anonymous mode; echoed back
            unchanged from ``alpha_init`` in wallet mode, where the level comes
            from ``theta_w`` instead.
    """

    beta_S: float
    beta_Z: float
    fisher_info: np.ndarray
    alpha: float


def _beta_grad_fisher(
    X: np.ndarray,
    offset: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    cauchy_scale: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Penalized logistic score and Fisher information at `beta`.

    The Gelman et al. (2008, §3) approximate-EM form: the Cauchy(0, scale)
    prior contributes ``-c * beta`` to the score and ``diag(c)`` to the
    information, with ``c = 2 / (scale**2 + beta**2)`` evaluated at the current
    coefficient. Shared by the IRLS loop and the post-loop information report so
    the two can never drift apart.

    Args:
        X: (n, p) standardized design — ``[x_S~, x_Z~]`` in wallet mode,
            ``[1, x_S~, x_Z~]`` in anonymous mode.
        offset: (n,) fixed per-trade offset ``logit(theta_w[wallet(t)])``;
            identically zero in anonymous mode, whose level is a fitted column.
        y: (n,) fractional Bernoulli targets ``q(Z_t = 1)``.
        beta: (p,) current internal-scale coefficients.
        cauchy_scale: Scale of the Cauchy prior on each coefficient — a scalar
            when every coefficient shares one, or a (p,) vector when the
            intercept carries a wider scale than the slopes.

    Returns:
        `(grad, fisher)`: the (p,) penalized score and the (p, p) augmented
        Fisher information (data curvature + prior curvature).
    """
    mu = sigmoid(offset + X @ beta)
    w = mu * (1.0 - mu)
    c = 2.0 / (cauchy_scale**2 + beta**2)
    grad = X.T @ (y - mu) - c * beta
    fisher = (X * w[:, None]).T @ X + np.diag(c)
    return grad, fisher


def update_beta_irls(
    markets: list[MarketData],
    q_vz_list: list[np.ndarray],
    theta_w: np.ndarray,
    m_S: float,
    s_S: float,
    m_Z: float,
    beta_S_init: float,
    beta_Z_init: float,
    cauchy_scale: float | None = None,
    *,
    anonymous: bool = False,
    alpha_init: float = 0.0,
    alpha_cauchy_scale: float | None = None,
) -> IRLSFit:
    """Pooled IRLS update for the insider predictor with Cauchy priors (KTD1/2/6).

    Fits a logistic regression of the fractional target `y_t = q(Z_t = 1)` on
    the standardized covariates, pooled over all markets, trades j >= 1 only
    (Z_0 := 0 excludes trade 0). The design depends on the model mode, and only
    on the mode — the numerics below are shared:

      * **wallet**: no intercept, `[x_S~_t, x_Z~_t]` offset by
        `logit(theta_w[wallet(t)])` at the (freshly updated) per-wallet
        posterior means. theta_w's Beta hierarchy carries the level, so
        beta_S/beta_Z are pure slopes.
      * **anonymous**: no wallets exist, so no offset does either; a leading
        intercept column fits `alpha` as the level. The intercept is identified
        here *because* theta_w is gone — nothing else competes for it. It is
        also far more weakly shrunk than theta_w ever was under Beta(1, 19),
        which is an incidental-parameters-style bias risk on low-trade-count
        markets; `PhiPrior.alpha_cauchy_scale` documents the measured bias and
        the prior-scale choice (plan 2026-07-23-005 KTD2).

    Each coefficient carries an independent Cauchy(0, scale) prior — 2.5 on the
    standardized slopes, `PhiPrior.alpha_cauchy_scale` on the intercept —
    handled by the Gelman et al. (2008, §3) approximate-EM IRLS modification:
    at the current beta, the prior's contribution to the score is
    `-2 beta_j / (scale^2 + beta_j^2)` and to the curvature is
    `2 / (scale^2 + beta_j^2)` — these match the exact first and (a stabilizing
    local) second derivative of the Cauchy log-density, so the augmented
    Fisher information stays positive definite (hence finite estimates) even
    when q(Z) is perfectly separated by a covariate, where the data alone
    would drive the curvature to 0.

    Public because `online_scorer.OnlineScorer` calls it directly for its
    periodic beta refresh: this signature is a cross-module contract, not an
    M-step implementation detail, and changing it breaks the streaming path.
    The anonymous-mode arguments are keyword-only with wallet-mode defaults for
    that reason; the return became the named `IRLSFit` (whose first three fields
    keep the historical `(beta_S, beta_Z, fisher_info)` order) so the intercept
    has somewhere to go.

    Args:
        markets: Input market data.
        q_vz_list: Per-market soft (V,Z) assignments, each (T_k, 4).
        theta_w: (n_wallets,) current per-wallet propensity estimates (used as
            the fixed offset, per the ECM block order — these are the
            block-(a) values updated earlier in this same M-step). Ignored, and
            legitimately empty, in anonymous mode.
        m_S: Pooled mean of log_size_ratio (standardization).
        s_S: Pooled std of log_size_ratio (standardization); see `S_STD_FLOOR`.
        m_Z: Pooled mean of E[Z_prev] (centering).
        beta_S_init: Warm-start internal-scale beta_S (previous iteration).
        beta_Z_init: Warm-start internal-scale beta_Z (previous iteration).
        cauchy_scale: Scale of the Cauchy(0, scale) prior on each standardized
            *slope*. ``None`` (the default, used by direct callers/tests)
            resolves to `PhiPrior().beta_cauchy_scale` so the single prior spec
            remains authoritative.
        anonymous: Fit the anonymous-mode design (intercept column, no offset)
            instead of the wallet-mode one. Keyword-only.
        alpha_init: Warm-start intercept; also the value echoed back in wallet
            mode, where the intercept is not a parameter. Keyword-only.
        alpha_cauchy_scale: Scale of the Cauchy(0, scale) prior on the
            intercept; ``None`` resolves to `PhiPrior().alpha_cauchy_scale`.
            Unused in wallet mode. Keyword-only.

    Returns:
        The `IRLSFit` for the fitted mode.
    """
    if cauchy_scale is None:
        cauchy_scale = PhiPrior().beta_cauchy_scale
    if alpha_cauchy_scale is None:
        alpha_cauchy_scale = PhiPrior().alpha_cauchy_scale
    wallet_idx, x_S, x_Z, y = _pooled_zj_covariates(markets, q_vz_list, m_S, s_S, m_Z)
    n_coef = 3 if anonymous else 2

    if wallet_idx.size == 0:
        # No trades with j >= 1 anywhere (every market has <= 1 trade): hold
        # the previous coefficients and report zero information.
        return IRLSFit(beta_S_init, beta_Z_init, np.zeros((n_coef, n_coef)), alpha_init)

    if anonymous:
        # Intercept first so the slope block keeps its wallet-mode ordering,
        # and the returned Fisher information's trailing 2x2 stays comparable.
        X = np.column_stack([np.ones_like(x_S), x_S, x_Z])
        offset = np.zeros_like(x_S)
        beta = np.array([alpha_init, beta_S_init, beta_Z_init], dtype=float)
        prior_scale: float | np.ndarray = np.array(
            [alpha_cauchy_scale, cauchy_scale, cauchy_scale]
        )
    else:
        X = np.column_stack([x_S, x_Z])
        offset = logit(theta_w)[wallet_idx]
        beta = np.array([beta_S_init, beta_Z_init], dtype=float)
        # Left as the incoming scalar, not broadcast to a length-2 array: the
        # arithmetic is elementwise-identical either way, and keeping the scalar
        # keeps the wallet-mode path bit-for-bit what it was pre-anonymous-mode.
        prior_scale = cauchy_scale

    def _penalized_obj(b: np.ndarray) -> float:
        """Expected complete-data log-lik + Cauchy log-prior at `b`."""
        eta = offset + X @ b
        log_lik = float(np.sum(y * eta - log1pexp(eta)))
        log_prior = float(-np.sum(np.log1p((b / prior_scale) ** 2)))
        return log_lik + log_prior

    obj = _penalized_obj(beta)
    converged = False
    for _ in range(_IRLS_MAX_ITER):
        grad, fisher = _beta_grad_fisher(X, offset, y, beta, prior_scale)

        try:
            step = np.linalg.solve(fisher, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(fisher, grad, rcond=None)[0]

        # Step-halving: only accept a step that does not decrease the
        # penalized objective (KTD6) — protects against IRLS overshoot near
        # separation, where the data curvature can be poorly scaled.
        step_scale = 1.0
        beta_new = beta
        obj_new = obj
        for _ in range(20):
            candidate = beta + step_scale * step
            candidate_obj = _penalized_obj(candidate)
            if candidate_obj >= obj - 1e-12:
                beta_new, obj_new = candidate, candidate_obj
                break
            step_scale *= 0.5

        rel_change = float(np.max(np.abs(beta_new - beta))) / max(
            1.0, float(np.max(np.abs(beta)))
        )
        beta, obj = beta_new, obj_new
        if rel_change < _IRLS_REL_TOL:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"update_beta_irls hit the {_IRLS_MAX_ITER}-iteration cap without "
            "converging to the relative-change tolerance; returning the "
            "current estimate.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Fisher info at the *final* beta (the loop's last evaluation predates the
    # accepted step), data + prior curvature, for the caller's Laplace block.
    _, fisher = _beta_grad_fisher(X, offset, y, beta, prior_scale)

    if anonymous:
        return IRLSFit(float(beta[1]), float(beta[2]), fisher, float(beta[0]))
    return IRLSFit(float(beta[0]), float(beta[1]), fisher, alpha_init)


def _mstep_sufficient_stats(
    markets: list[MarketData],
    q_vz_list: list[np.ndarray],
    mu_filt_list: list[np.ndarray],
    sigma2_filt_list: list[np.ndarray],
    gamma: float,
) -> dict[str, float | list[float]]:
    """Pool the E-step sufficient statistics the variance/transition blocks need.

    One sweep over the markets accumulating the three statistic families the
    M-step's closed-form blocks consume:

      * ``SS_v``/``N_v`` — q(V)-weighted squared process increments per unit
        time (trades j >= 1 with a positive gap), for the sigma2_v update;
      * ``n_00``/``n_01``/``n_10``/``n_11`` — expected V-transition counts from
        the product of consecutive V-marginals, for the q_01/q_10 update;
      * ``SS_z``/``N_z`` — q(Z)-weighted size-scaled squared observation
        residuals over all trades, for the tau2_z update.

    Shared by ``_vem_m_step`` and ``src.inference.laplace``, whose curvature
    blocks must be evaluated against the very statistics the M-step optimized —
    a second copy of this algebra could silently drift from it.

    Note: the increments use the *filtered* moments the ADF pass returns, and
    the process statistic drops the lag-one cross-covariance a smoothed E-step
    would supply. That is the estimator this M-step has always used; changing it
    is an open scope decision, not something this helper alters.

    Args:
        markets: Input market data.
        q_vz_list: Per-market soft (V, Z) assignments, each (T_k, 4).
        mu_filt_list: Per-market mixed Kalman means, each (T_k,).
        sigma2_filt_list: Per-market mixed Kalman variances, each (T_k,).
        gamma: Size-informativeness scaling from the current parameters.

    Returns:
        Dict with ``SS_v``/``N_v`` and ``SS_z``/``N_z`` (length-2 lists indexed
        by regime) and the four scalar transition counts ``n_ij``.
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
                SS_v[v] += float(
                    (
                        q_V_v[valid] * (resid2[valid] + extra_var[valid]) / dt[valid]
                    ).sum()
                )
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


def _vem_m_step(
    markets: list[MarketData],
    q_vz_list: list[np.ndarray],
    mu_filt_list: list[np.ndarray],
    sigma2_filt_list: list[np.ndarray],
    params: ModelParams,
    theta_w: np.ndarray,
    n_wallets: int,
    m_S: float,
    s_S: float,
    m_Z: float,
    *,
    prior: PhiPrior | None = None,
    estimate_betas: bool = False,
) -> tuple[ModelParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ECM M-step: theta_w Newton, then moment-matched updates, then beta IRLS.

    Block order is fixed (KTD7): (a) theta_w using the previous iteration's
    betas as a fixed offset; (b) q-transitions/variances (unchanged from the
    beta-free update); (c) beta_S/beta_Z IRLS using the freshly updated
    theta_w (from block (a)) as the offset. See the module docstring for why
    this ordering keeps the EM objective monotone blockwise.

    In anonymous mode block (a) is skipped (there are no wallets) and block (c)
    additionally fits the per-market intercept `alpha`, which is then the
    predictor's whole level.

    Args:
        markets: Input market data.
        q_vz_list: Per-market soft (V,Z) assignments, each (T_k, 4).
        mu_filt_list: Per-market mixed Kalman means, each (T_k,).
        sigma2_filt_list: Per-market mixed Kalman variances, each (T_k,).
        params: Current model parameters (beta_S/beta_Z are internal-scale).
        theta_w: Current per-wallet propensity estimates.
        n_wallets: Total wallet count.
        m_S: Pooled mean of log_size_ratio (standardization).
        s_S: Pooled std of log_size_ratio (standardization); see `S_STD_FLOOR`.
        m_Z: Pooled mean of E[Z_prev] (centering).
        prior: The single M-step prior spec (`PhiPrior`); `None` resolves to
            `PhiPrior()` (the behaviour-preserving defaults). Supplies the
            sigma2/tau2 Inverse-Gamma, q Beta, and beta Cauchy hyperparameters
            — no prior constant is hardcoded in this body (plan R8/KTD3).
        estimate_betas: If False (the default, matching `variational_em`'s
            public default — see there for why beta estimation is opt-in), skip
            block (c) and hold beta_S/beta_Z (and, in anonymous mode, `alpha`)
            fixed at their `params` value for every iteration. With the offset
            then uniformly 0 at the default beta_S = beta_Z = 0.0, block (a)
            reduces exactly to the original Beta-count theta_w update.

    Returns:
        `(params, theta_w, theta_w_logit_mean, theta_w_logit_var, beta_fisher_info)`.
    """
    if prior is None:
        prior = PhiPrior()

    # ---- (a) theta_w: offset-adjusted per-wallet penalized Newton ----
    # Anonymous mode has no wallets to fit: the predictor's level is the
    # intercept `alpha`, which block (c) estimates jointly with the slopes.
    if params.anonymous:
        theta_w_new = theta_w
        theta_w_logit_mean = logit(theta_w)
        theta_w_logit_var = np.zeros_like(theta_w_logit_mean)
    else:
        theta_w_new, theta_w_logit_mean, theta_w_logit_var = _update_theta_w(
            markets,
            q_vz_list,
            n_wallets,
            params.beta_S,
            params.beta_Z,
            m_S,
            s_S,
            m_Z,
            params.a,
            params.b,
            logit(theta_w),
        )

    stats = _mstep_sufficient_stats(
        markets, q_vz_list, mu_filt_list, sigma2_filt_list, params.gamma
    )
    SS_v, N_v = stats["SS_v"], stats["N_v"]
    SS_z, N_z = stats["SS_z"], stats["N_z"]

    # ---- q_01, q_10: product-of-marginals Beta update (prior.q_beta_a/b) ----
    q_01_new = prior.q_map(stats["n_01"], stats["n_00"])
    q_10_new = prior.q_map(stats["n_10"], stats["n_11"])
    # Clamp to (0,1) open interval to avoid degenerate regimes
    q_01_new = float(np.clip(q_01_new, 1e-6, 1.0 - 1e-6))
    q_10_new = float(np.clip(q_10_new, 1e-6, 1.0 - 1e-6))

    # ---- sigma2_0, sigma2_1: IG MAP update (mode = beta/(alpha+1)) ----
    sigma2_0_new = max(prior.sigma2_map(SS_v[0], N_v[0]), 1e-6)
    sigma2_1_new = max(prior.sigma2_map(SS_v[1], N_v[1]), 1e-6)
    sigma2_1_new = max(sigma2_1_new, sigma2_0_new)

    # ---- tau2_0, tau2_1: weak-IG pseudo-count MAP (prior.tau2_map) ----
    # Reduces to the pre-refactor moment-match SS/N as the tau2 prior -> 0; the
    # weak default prior gives the Laplace tau2 block a defined curvature at the
    # cost of a numerically negligible estimate shift (plan R8).
    tau2_0_new = max(prior.tau2_map(SS_z[0], N_z[0]), 1e-6)
    tau2_1_new = max(prior.tau2_map(SS_z[1], N_z[1]), 1e-6)
    # Insiders have tighter obs variance (more price-informative trades)
    tau2_1_new = min(tau2_1_new, tau2_0_new)

    # ---- (c) alpha, beta_S, beta_Z: pooled IRLS, offset by the fresh theta_w ----
    # `alpha` rides on this block, so `estimate_betas=False` freezes the
    # anonymous intercept at its incoming value exactly as it freezes the slopes.
    if estimate_betas:
        beta_S_new, beta_Z_new, beta_fisher_info, alpha_new = update_beta_irls(
            markets,
            q_vz_list,
            theta_w_new,
            m_S,
            s_S,
            m_Z,
            params.beta_S,
            params.beta_Z,
            cauchy_scale=prior.beta_cauchy_scale,
            anonymous=params.anonymous,
            alpha_init=params.alpha,
            alpha_cauchy_scale=prior.alpha_cauchy_scale,
        )
    else:
        beta_S_new, beta_Z_new = params.beta_S, params.beta_Z
        alpha_new = params.alpha
        n_coef = 3 if params.anonymous else 2
        beta_fisher_info = np.zeros((n_coef, n_coef))

    new_params = replace(
        params,
        q_01=q_01_new,
        q_10=q_10_new,
        sigma2_0=sigma2_0_new,
        sigma2_1=sigma2_1_new,
        tau2_0=tau2_0_new,
        tau2_1=tau2_1_new,
        beta_S=beta_S_new,
        beta_Z=beta_Z_new,
        alpha=alpha_new,
    )
    return (
        new_params,
        theta_w_new,
        theta_w_logit_mean,
        theta_w_logit_var,
        beta_fisher_info,
    )


def variational_em(
    markets: list[MarketData],
    config: InferenceConfig,
    *,
    n_wallets: int | None = None,
    params_init: ModelParams | None = None,
    theta_w_init: np.ndarray | None = None,
    n_iter: int = 50,
    tol: float = 1e-3,
    n_jobs: int = 1,
    prior: PhiPrior | None = None,
    estimate_betas: bool = False,
) -> VEMOutput:
    """Fit the switching SSM by variational EM (ADF E-step + ECM M-step).

    Substantially faster than Particle Gibbs: no sampling, no MCMC chains.
    Suitable for the "fast tier" wallet ranking. Approximate posteriors are
    good enough for AUC/ranking; credible intervals are not reliable.

    Args:
        markets: List of K markets; K = 1 is valid.
        config: InferenceConfig; warm-start params come from `ModelParams.warm_start`
            if `params_init` is None.
        n_wallets: Total wallet count; inferred from market data if None, and 0
            in anonymous mode, which has no wallets.
        params_init: Optional model parameter initialization. Also selects the
            model mode: anonymous mode is entered by passing params with
            ``anonymous=True`` (e.g. `ModelParams.warm_start(Y, anonymous=True)`),
            since the default warm start is wallet mode.
        theta_w_init: Optional initial per-wallet propensities. Defaults to
            `Beta(a, b)` mean = a/(a+b) for all wallets.
        n_iter: Maximum EM iterations.
        tol: Convergence tolerance on the relative change in log-marginal.
        n_jobs: Reserved for future joblib parallelism over markets; currently
            always sequential.
        prior: The single M-step prior spec (`PhiPrior`); `None` uses the
            behaviour-preserving defaults. Threaded unchanged into every M-step.
        estimate_betas: If False (default), hold beta_S/beta_Z fixed at
            `params_init`'s value across all iterations instead of fitting them
            via IRLS each M-step. With the default beta_S = beta_Z = 0.0 this
            recovers the pre-regression theta_w-only M-step exactly. Default is
            False because the ADF E-step cannot identify Z on the current
            synthetic generator (q(Z) is near-flat — Z modulates only the
            observation variance tau2_Z), so enabling beta estimation fits a
            spurious size-correlated beta_S (~-0.40) whose tilt drops the gate
            AUC from ~0.89 to ~0.68. Beta estimation is therefore opt-in
            (Laplace layer, real-data runs, Z-identifiability investigation)
            until the E-step identifies Z; pass True to enable it explicitly.

    Returns:
        VEMOutput with fitted params, posterior marginals, convergence trace,
        and the standardization/centering constants (m_S, s_S, m_Z).
    """
    if prior is None:
        prior = PhiPrior()

    if params_init is None:
        Y_concat = np.concatenate([m.Y for m in markets])
        params = ModelParams.warm_start(Y_concat)
    else:
        params = params_init

    if n_wallets is None:
        # Anonymous mode has no wallet layer at all; the synthetic generator
        # stamps a placeholder id 0 on every trade, which must not be mistaken
        # for one real wallet whose theta_w is worth fitting.
        n_wallets = (
            0
            if params.anonymous
            else int(max(int(m.wallet_ids.max()) for m in markets)) + 1
        )

    if theta_w_init is None:
        theta_w = np.full(n_wallets, params.a / (params.a + params.b))
    else:
        theta_w = np.array(theta_w_init, copy=True)

    # Standardization constants (Gelman et al. 2008), fit once per call from
    # the pooled dataset over all markets. s_S is ~0 for a degenerate
    # constant-size dataset; _vem_e_step guards that case (centering only).
    log_size_ratio_all = np.concatenate([m.log_size_ratio for m in markets])
    m_S = float(np.mean(log_size_ratio_all))
    s_S = float(np.std(log_size_ratio_all))
    # m_Z centers the persistence covariate E[Z_prev]; unlike (m_S, s_S) it
    # cannot be known before any E-step has run, so it starts at 0.0 (matching
    # the Z_0 := 0 convention) and is refreshed as a running pooled mean of
    # E[Z_prev] after each E-step, for use in the next iteration's predictor.
    m_Z = 0.0

    elbo_trace: list[float] = []
    prev_lm = float("-inf")

    q_vz_list: list[np.ndarray] = []
    mu_filt_list: list[np.ndarray] = []
    sigma2_filt_list: list[np.ndarray] = []
    # Persist the last completed M-step's per-wallet Laplace summary and beta
    # Fisher info across iterations: the EM loop can break out (convergence)
    # right after an E-step, before that iteration's M-step runs, so the
    # final VEMOutput must report the *last computed* values, not recompute.
    theta_w_logit_mean = logit(theta_w)
    theta_w_logit_var = np.zeros(n_wallets)
    beta_fisher_info = np.zeros((3, 3) if params.anonymous else (2, 2))

    for em_it in range(n_iter):
        # ---- E-step ----
        q_vz_list = []
        mu_filt_list = []
        sigma2_filt_list = []
        total_lm = 0.0
        for md in markets:
            q_vz, mu_f, sigma2_f, lm = _vem_e_step(
                md.Y,
                md.delta,
                md.log_size_ratio,
                md.wallet_ids,
                theta_w,
                params,
                m_S,
                s_S,
                m_Z,
            )
            q_vz_list.append(q_vz)
            mu_filt_list.append(mu_f)
            sigma2_filt_list.append(sigma2_f)
            total_lm += lm

        elbo_trace.append(total_lm)

        # Refresh m_Z from this iteration's E[Z_prev] values (indices 0..T-2
        # of each market, i.e. exactly the values the predictor consumed as
        # "prev_E_Z"), pooled across markets, for the next E-step.
        prev_E_Z_all = np.concatenate(
            [q_vz[:-1, 1] + q_vz[:-1, 3] for q_vz in q_vz_list if len(q_vz) > 1]
        )
        if prev_E_Z_all.size > 0:
            m_Z = float(np.mean(prev_E_Z_all))

        # ---- Convergence check ----
        if em_it > 0:
            denom = max(abs(prev_lm), 1.0)
            if abs(total_lm - prev_lm) / denom < tol:
                break
        prev_lm = total_lm

        # ---- M-step ----
        params, theta_w, theta_w_logit_mean, theta_w_logit_var, beta_fisher_info = (
            _vem_m_step(
                markets,
                q_vz_list,
                mu_filt_list,
                sigma2_filt_list,
                params,
                theta_w,
                n_wallets,
                m_S,
                s_S,
                m_Z,
                prior=prior,
                estimate_betas=estimate_betas,
            )
        )

    n_iter_run = len(elbo_trace)

    # Final posterior summaries
    Z_prob_list = [q_vz[:, 1] + q_vz[:, 3] for q_vz in q_vz_list]
    V_prob_list = [q_vz[:, 2] + q_vz[:, 3] for q_vz in q_vz_list]

    # Back-transform beta_S to original log_size_ratio units (§ module
    # docstring); beta_Z is unaffected since x_Z~ is centering-only.
    if s_S > S_STD_FLOOR:
        beta_S_orig = params.beta_S * 0.5 / s_S
    else:
        beta_S_orig = params.beta_S
    beta_Z_orig = params.beta_Z

    return VEMOutput(
        params=params,
        theta_w=theta_w,
        Z_prob=Z_prob_list,
        V_prob=V_prob_list,
        X_mean=mu_filt_list,
        elbo_trace=np.asarray(elbo_trace),
        n_iter_run=n_iter_run,
        m_S=m_S,
        s_S=s_S,
        m_Z=m_Z,
        theta_w_logit_mean=theta_w_logit_mean,
        theta_w_logit_var=theta_w_logit_var,
        beta_S_orig=beta_S_orig,
        beta_Z_orig=beta_Z_orig,
        beta_fisher_info=beta_fisher_info,
        alpha_orig=_alpha_orig(params, beta_S_orig, m_S, m_Z),
    )
