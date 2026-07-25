---
title: An EM fixed point is not a posterior mode, so ECM curvature is not a Laplace approximation
date: 2026-07-25
category: best-practices
module: inference validation / VEM uncertainty layer
problem_type: best_practice
component: testing_framework
severity: high
applies_when:
  - "Building a Gaussian uncertainty layer over the point estimate of an EM/ECM/VEM optimizer"
  - "Reading a PSIS khat, ESS, or any importance-sampling diagnostic against a hand-built proposal"
  - "A plan encodes a stop condition of the form 'if diagnostic D exceeds threshold T, do remedy R'"
  - "Curvature is evaluated at an optimum where an order constraint or box bound may be active"
tags: [laplace-approximation, psis-khat, expected-vs-observed-information, louis-identity, variational-em, importance-sampling, diagnostics, identifiability]
---

# An EM fixed point is not a posterior mode, so ECM curvature is not a Laplace approximation

## Context

Plan `docs/plans/2026-07-23-002-feat-vem-validation-suite-plan.md` specified a
"Laplace uncertainty layer" over the eight-dimensional unconstrained parameter
vector `phi`, built (per its R2) from the M-step's *own expected complete-data*
curvature at the VEM optimum, and then used that Gaussian as the proposal for a
PSIS-khat diagnostic (R5). khat came back **5.82 at dev scale and 24.0 at gate
scale** against the standard 0.7 threshold.

The plan's stop condition read: khat > 0.7 is "the report's trigger to enrich
the variational family before scaling". That prescription was wrong. A
methods-critic pass established numerically that khat was diagnosing the
*proposal*, not the variational family, and acting on the stop condition would
have spent a whole scope cycle enriching a family that was never implicated.

## Guidance

**Before reading any importance-sampling diagnostic, check that the proposal is
centred at a mode of the very target the diagnostic weights against.** The cheap
check: the gradient of the log target at the proposal centre, expressed in
proposal-sd units. A mode gives ~0 in every dimension. Anything large means khat
is measuring displacement, not family adequacy — a *richer* family centred at
the same non-mode with the same curvature scores exactly the same khat. This
cycle added that check as `phi_centring_gradient` in `src/analysis/validation.py`
(2·8 ADF passes, under 2% of a 1000-draw `psis_khat` call at gate scale — there
is no cost argument for skipping it).

Two independent defects made the ECM-curvature Gaussian unusable as a Laplace
approximation, both verified numerically on the standard dev fixture:

1. **Wrong information matrix.** Expected complete-data information is not
   observed information. By the Louis (1982) missing-information identity,
   `observed = complete − missing`, so complete-data curvature systematically
   *over*-states precision. Measured at `log sigma2_0`: observed information
   **2.24** versus the layer's **252.3** — 113x over-precise. Any Gaussian built
   this way is far too tight, and importance weights against a diffuser target
   blow up.

2. **Wrong centre.** An EM/ECM fixed point is stationary for the
   variational/complete-data objective, not for the marginal target
   `log p(Y|phi) + log p(phi)` that PSIS (and SBC) weight against. Run all the
   way to convergence (1500 iterations, relative change 1.3e-8), the target's
   gradient at the centre was still ~**10 Laplace sd** along `log sigma2_0`, and
   the observed information at `tau2_1` was **negative (−3.96)** — the centre is
   a local *minimum* along that axis, which no Laplace approximation can be.

A third, compounding finding: **curvature was being evaluated on an active
constraint boundary.** The M-step's order clamp `sigma2_1 = max(sigma2_1,
sigma2_0)` in `src/inference/variational_em.py` binds *exactly* at every fitted
point — `sigma2_0 == sigma2_1` to the last bit on 5/5 dev restarts, and still at
convergence. An unconstrained quadratic is simply the wrong local model there,
and **~75% of draws from the fitted Gaussian violate the estimator's own
constraints.** The `V` regime is additionally non-identified at that point: the
ADF log-marginal moves **less than 5e-13 over ±4 sd** in both `logit q_01` and
`logit q_10`, while the layer assigns those blocks precisions of ~106 and ~19 —
confident curvature over a flat direction.

Finally, a structural lesson about plans themselves: **a stop condition of the
form "if diagnostic D exceeds threshold T, do remedy R" must also state D's
preconditions**, or the threshold breach routes the next cycle into the wrong
work. Plan 002's stop condition named the remedy but not the precondition, so
the breach pointed at the variational family instead of at the proposal.

## Why This Matters

A PSIS/importance-sampling diagnostic is a statement about the *proposal*, not
about the model. A bad khat is uninterpretable as evidence about the target
until the centring precondition has been checked. Without that check the failure
mode is expensive and silent in the wrong direction: the number looks like
model-level evidence, the plan's stop condition converts it into scope, and a
correct model gets "fixed".

The over-precision half matters independently for anything downstream that
*consumes* the Gaussian rather than diagnosing it. SBC coverage or credible
intervals built on ECM curvature inherit both the 113x over-precision and the
centre offset, and would report spuriously tight, spuriously mis-calibrated
uncertainty for reasons that have nothing to do with the inference being tested.

## When to Apply

- Any time an EM/ECM/VEM point estimate is dressed up as a posterior via
  curvature at the optimum. Use observed information via the Louis identity, or
  optimize the marginal target directly — expected complete-data curvature is
  neither.
- Before reporting khat, ESS, or any importance-weight statistic against a
  hand-constructed proposal: report the centring gradient beside it, always.
- When an optimizer applies order clamps, box bounds, or projections: check
  whether the constraint is active at the optimum before treating local
  curvature as a Gaussian, and check what fraction of draws fall outside the
  feasible set.
- When writing a plan's stop conditions: for every "if D > T then R", also write
  down what must be true for D to be readable at all.

## Examples

The precondition check, as wired in `scripts/validate_vem.py` immediately after
the khat call:

```python
phi_posterior = laplace_from_vem(best_vem, inputs.markets, prior)
psis_result = psis_khat(best_vem, phi_posterior, inputs.markets, ...)
# The centring precondition for reading khat at all: 2*8 ADF passes,
# negligible beside the psis_draws passes just spent.
centring = phi_centring_gradient(
    best_vem, phi_posterior, inputs.markets, prior=prior, n_jobs=args.n_jobs
)
```

`phi_centring_gradient` central-differences the log PSIS target
`log p(Y | phi, theta_w_hat) + log p(phi)` along each *unconstrained* coordinate
(the coordinates the Gaussian actually lives on, where it is stationary at its
centre by construction), carrying the change-of-variables term, with a step of
1e-2 sd. It reports `centring_grad_sd_units` per dimension plus
`centring_grad_max_abs_dim` / `centring_grad_max_abs_sd`. A mode gives ~0
everywhere; an entry of −13 says the target rises 13 posterior-sds' worth away
from where the proposal sits.

The wrong reading and the right reading of the same khat:

```
# WRONG (plan 002's stop condition):
#   khat = 24.0 > 0.7  ->  enrich the variational family.
# RIGHT:
#   khat = 24.0 with centring_grad_max_abs_sd ~= 10  ->  the proposal is not
#   centred at a mode of its own target; khat says nothing about the family.
#   Fix the proposal (Louis observed information, or optimize the marginal),
#   then re-read khat.
```

The `khat_interpretation` text in `src/analysis/validation.py` was rewritten to
carry this, so the number cannot be read in isolation from the artifact:

> khat is only a family-adequacy measure once the proposal is centred at a mode
> of its own target — a richer family centred at the same non-mode with the same
> curvature scores the same khat. Read `phi_centring_gradient` before concluding
> anything about the family.

## Related

- `src/inference/laplace.py` — module docstring documents all three defects
  (wrong information matrix, wrong centre, active order constraint) with the
  measured numbers; read it before building anything on `PhiPosterior`.
- `src/analysis/validation.py` — `phi_centring_gradient`, `CentringDiagnostic`,
  `CENTRING_NOTE`, `PSIS_SCOPE_NOTE`, `khat_interpretation`.
- `src/inference/variational_em.py` — the `sigma2_1 = max(sigma2_1, sigma2_0)`
  order clamp that is active at every fitted point.
- `scripts/validate_vem.py` — the validation CLI that reports khat and the
  centring gradient side by side.
- `agent_reference/ARCHITECTURE.md` §6.2 and §12.1; `agent_reference/STATUS.md`
  roadmap P8 — current status of the uncertainty layer and its fallback ladder.
- `docs/plans/2026-07-23-002-feat-vem-validation-suite-plan.md` — R2/R5 and the
  stop condition that encoded the wrong remedy.
- `docs/solutions/best-practices/default-off-flag-makes-gate-evidence-vacuous.md`
  — the sibling learning from this cycle on acceptance evidence.
- Louis, T. A. (1982), "Finding the observed information matrix when using the
  EM algorithm", JRSS-B — the missing-information identity.
- Yao, Vehtari, Simpson & Gelman (2018), "Yes, but did it work?: Evaluating
  variational inference" — PSIS khat and its bands.
- Talts, Betancourt, Simpson, Vehtari & Gelman (2018), "Validating Bayesian
  inference algorithms with simulation-based calibration" — SBC as the actual
  faithfulness test that khat is only a necessary-not-sufficient precursor to.
