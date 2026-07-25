---
title: VEM Validation Suite and Laplace Layer - Plan
type: feat
date: 2026-07-23
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# VEM Validation Suite and Laplace Layer - Plan

**Sequence:** Plan 2 of 5. Depends on plan 1 (`2026-07-23-001`, IRLS Fisher info + centered covariates). Plan 3 (SBC) consumes this plan's Laplace layer.

---

## Goal Capsule

Stand up the referee-facing validation suite for the VEM fast path — ELBO traces, multi-seed stability, held-out one-step predictive log-likelihood, and the PSIS-k̂ diagnostic (Yao et al. 2018) — plus the Laplace uncertainty layer over `phi` that makes parameter-level PSIS/SBC/coverage possible at all (VEM's M-step is a point estimator).

**Authority hierarchy:** CLAUDE.md hard rules > agent_reference docs > this plan. PG freeze is hard: no diagnostic may be framed as, or computed by, comparison against PG. Historical PG numbers (AUC 0.962, 3117.5 s) are cited context only.

**Stop conditions:** surface (do not silently proceed) if (a) the Laplace Hessian is not positive-definite at the VEM optimum on the standard synthetic fixture after the documented fallbacks, or (b) PSIS-k̂ > 0.7 on synthetic data at gate scale — that is the report's trigger to enrich the variational family before scaling, a scope change the user owns.

---

## Product Contract

### Summary

Because the exact sampler is retired, the approximation must be defended with the accepted samplerless toolbox. This plan delivers: (1) `PhiPosterior` — a Laplace/curvature Gaussian over unconstrained `phi` built from the IRLS Fisher information (betas) and analytic conditional curvature (variances, transition probabilities); (2) held-out one-step predictive log-likelihood via the ADF forward pass; (3) PSIS-k̂ computed from importance ratios between the (ADF-approximated) parameter marginal and the Laplace posterior — a Laplace-shape adequacy check, with SBC (plan 3) carrying the faithfulness claim; (4) a `scripts/validate_vem.py` CLI bundling ELBO traces, R-restart stability (AUC spread, top-K Jaccard, beta spread), held-out LL, and k̂ into one JSON + figures artifact.

### Problem Frame

The single most likely referee rejection is "you replaced an exact sampler with an approximation and never showed it is faithful." The report's answer is SBC + PSIS + coverage (plans 2–3). All of these need a posterior distribution over parameters; VEM currently produces points. The cheap, citable bridge is Laplace at the mode.

### Requirements

Uncertainty layer:
- R1. `PhiPosterior` dataclass: unconstrained-scale mean and covariance for `phi = (sigma2_0, sigma2_1, q_01, q_10, beta_S, beta_Z, tau2_0, tau2_1)` with named dimensions, `sample(rng, n)`, `logpdf(x)`, and transforms to/from the constrained scale (log for variances, logit for q's, identity for betas on the internal standardized scale).
- R2. Beta block uses the IRLS Fisher information from plan 1 (includes Cauchy-prior curvature); variance and transition blocks use analytic curvature of their expected complete-data conditionals at the optimum (Inverse-Gamma / Beta forms on transformed scales). Cross-block covariance may be zero (block-diagonal) — documented as part of the approximation.
- R3. Non-PD or degenerate curvature triggers a documented fallback (jitter, then per-dimension curvature), never a crash.

Validation metrics:
- R4. Held-out one-step predictive log-likelihood: per market, fit on the first (1 − h) fraction of trades and score the tail via ADF one-step predictive densities of `Y`; h configurable (default 0.2); pooled and per-market values reported.
- R5. PSIS-k̂: S draws (default 1000) from `PhiPosterior`; log ratio = ADF log-marginal `log p(Y | phi_s)` + `log p(phi_s)` − Laplace `log q(phi_s)`; smoothed via PSIS (arviz `psislw`); k̂ reported with the standard < 0.5 / 0.5–0.7 / > 0.7 interpretation. **Scope of the claim (methods-critic MAJOR):** because both q and the target are built on the ADF surface, this k̂ measures *Laplace-shape adequacy for the ADF-implied parameter marginal* — it cannot detect ADF's own bias relative to the true posterior. Name it accordingly (JSON key like `psis_khat_laplace_vs_adf`), state in the docstring that a good k̂ is expected and is a necessary-not-sufficient check, and cross-reference SBC (plan 3) as the actual faithfulness test.
- R8. VEM's hardcoded M-step prior constants are refactored to import from the single prior spec: the q-transition `Beta(1,1)` (`variational_em.py:188`), the sigma² `IG(2,1)` MAP terms (`variational_em.py:204-205`), and the currently prior-free tau² moment-match (`variational_em.py:245`), which gains a weakly-informative InvGamma prior so its Laplace block has defined curvature. `PhiPrior` defaults are set equal to the current effective values (Beta(1,1); IG(2,1); tau² prior weak enough to be numerically negligible at gate scale) so gate behavior is preserved — verified by the gate re-run. Without this refactor, SBC/PSIS would compare against densities the estimator never used (methods-critic MAJOR; SBC validity depends on it).
- R6. Multi-seed stability: R restarts (default 5) → per-restart terminal ELBO, pooled AUC, back-transformed betas, top-K wallet sets; report ELBO/AUC/beta spreads and mean pairwise top-K Jaccard.
- R7. One CLI (`scripts/validate_vem.py`) produces a single JSON artifact plus figures (ELBO trace overlay, k̂ diagnostic, held-out LL bar per market) under `results/validation/`; runs at gate scale in well under an hour on synthetic defaults.

### Scope Boundaries

- **Out of scope:** SBC and coverage (plan 3); enriching the variational family if k̂ > 0.7 (explicitly a new decision for the user — see stop conditions); posterior over `theta_w` beyond the existing per-wallet Beta (already distributional); real-data runs (the CLI must accept them, but the acceptance evidence here is synthetic).
- **Deferred to follow-up work:** PG-VB (Polson-Scott-Windle variational block) as the robustness appendix if a referee demands a fully variational `beta` posterior.

---

## Planning Contract

### Key Technical Decisions

- KTD1 — **Laplace over `phi`, block-diagonal, on unconstrained scales.** Cheapest defensible object; matches Gelman-et-al IRLS curvature for betas. Block-diagonality is honest (stated) and testable via SBC in plan 3 — if coverage fails, that finding is itself reportable.
- KTD2 — **PSIS targets the parameter marginal, not the full latent joint.** Ratios use the ADF log-marginal (already computed for `elbo_trace` at `variational_em.py:336`) as `log p(Y | phi)`. This avoids sampling the high-dimensional discrete latent space and matches what `PhiPosterior` approximates. Per R5, the diagnostic is framed as Laplace-shape adequacy for the ADF surrogate — never as evidence that ADF matches the true posterior; SBC (plan 3) owns the faithfulness claim.
- KTD3 — **Priors used in `log p(phi)` are the model's own, enforced by refactor, not convention (R8):** Cauchy(0, 2.5) on standardized betas (plan 1) plus the conjugate-form priors that `_vem_m_step` actually optimizes against, lifted out of hardcoded constants into a single `PhiPrior` spec that inference, Laplace, PSIS, and plan 3's SBC all import. Defaults equal current effective values; changing any hyperparameter later is a one-line, everywhere-consistent edit.
- KTD4 — **New module `src/inference/laplace.py`**, not more weight in `variational_em.py`; analysis-side metrics live in `src/analysis/validation.py`; `scripts/validate_vem.py` stays a thin CLI (repo rule: logic in `src/`).
- KTD5 — **Held-out split is per-market tail**, preserving temporal order (no shuffling — leakage discipline the signal-eval plan will inherit). `delta` for the first held-out trade is measured from the last training trade.

---

## Implementation Units

### U1. Prior spec, M-step prior refactor, and `PhiPosterior` Laplace layer

**Goal:** Curvature-based Gaussian over unconstrained `phi`; one authoritative prior spec that the M-step itself uses.
**Requirements:** R1, R2, R3, R8, KTD3.
**Dependencies:** plan 1 U3 (Fisher info exposed on `VEMOutput`).
**Files:** `src/inference/laplace.py` (new), `src/inference/variational_em.py` (R8 refactor), `config/default_params.py` (`PhiPrior`), `tests/test_laplace.py` (new), `tests/test_variational_em.py` (refactor regression).
**Approach:** `PhiPrior` in `default_params.py` with defaults equal to the current effective M-step values (q's Beta(1,1); sigma² IG(2,1); new weak InvGamma on tau²; Cauchy(0, 2.5) on standardized betas). Refactor `_vem_m_step` per R8 to consume it. `PhiPosterior` per R1; builder `laplace_from_vem(vem_output, markets)` assembling blocks per KTD1/R2.
**Test scenarios:**
- R8 regression: with `PhiPrior` defaults, VEM outputs on the standard fixture are unchanged (tight tolerance vs pre-refactor pinned values); tau²'s new weak prior shifts estimates by a documented negligible amount at gate scale.
- Prior-consistency audit (importable as a test): `PhiPrior` is the only definition — grep-style assertion that `variational_em.py` contains no residual hardcoded prior constants.
- Shapes/round-trip: constrained↔unconstrained transforms invert to machine precision for random valid `phi`.
- Sampling sanity: on the standard synthetic fixture, `sample(rng, 2000)` back-transformed means within ~2 posterior sds of the VEM point estimates.
- Non-PD fallback: hand a singular Hessian block → jitter/diagonal fallback path returns a valid distribution and flags it.
- Beta block: variance decreases as synthetic T grows (500 vs 2000) — curvature scales with data.
**Verification:** new tests green; gate re-run reproduces plan 1 numbers (the R8 refactor is behavior-preserving by construction).

### U2. Held-out one-step predictive log-likelihood

**Goal:** Temporal train/tail split and ADF predictive scoring.
**Requirements:** R4, KTD5.
**Dependencies:** none within this plan (uses plan 1's E-step; file-disjoint from U1 — parallelizable).
**Files:** `src/analysis/validation.py` (new), `tests/test_validation.py` (new).
**Approach:** `holdout_split(markets, h)` → head/tail `MarketData` pairs preserving order and recomputing the boundary `delta`; `heldout_predictive_ll(vem_output, head, tail)` runs the ADF forward pass over head then accumulates one-step log predictive densities over tail (predictive of `Y_t` given filtered state, mixture over V/Z branches — reuse the E-step's per-step predictive quantities rather than reimplementing).
**Test scenarios:**
- Split integrity: head+tail partition the market; timestamps ordered; boundary `delta` equals the true inter-trade gap; h=0 returns empty tail handled gracefully.
- Better-model-wins: on synthetic data, the generating `phi` scores higher held-out LL than a corrupted `phi` (e.g. tau2 doubled) — directional sanity.
- Determinism: same seed/inputs → identical LL.
**Verification:** tests green; function importable by U4 CLI.

### U3. PSIS-k̂ diagnostic

**Goal:** Referee-recognized "did the approximation work" scalar.
**Requirements:** R5.
**Dependencies:** U1.
**Files:** `src/analysis/validation.py`, `tests/test_validation.py`.
**Approach:** `psis_khat(vem_output, phi_posterior, markets, n_draws)`: for each draw, run the ADF forward pass to get `log p(Y | phi_s)` (batch E-step, no EM), add `log p(phi_s)` from the prior spec, subtract `q.logpdf`; feed log-weights to `arviz.psislw`. Vectorize the per-draw loop with joblib over draws (embarrassingly parallel; `n_jobs` arg, default 1). Runtime note: one E-step pass per draw — at dev-scale synthetic this is ~1 s/draw; the CLI default uses a reduced synthetic size, and 1000 draws with `n_jobs 8` stays in the minutes range at that scale.
**Test scenarios:**
- Self-consistency: when the "posterior" IS the sampling distribution (score a Gaussian toy where p ≡ q), k̂ is small (< 0.5).
- Mismatch detection: inflate the Laplace covariance ×100 or shift its mean → k̂ degrades toward/past 0.7 (monotone-direction assertion, not exact value).
- Log-weight finiteness: no NaN/inf for all draws on the standard fixture (Kalman log-lik floor −500 already guards the extreme).
**Verification:** tests green; k̂ recorded under the `psis_khat_laplace_vs_adf` key with the R5 scope-of-claim string (no hard threshold asserted in tests — the 0.7 rule is a reporting/stop-condition matter, not CI).

### U4. `scripts/validate_vem.py` CLI and figures

**Goal:** One command → one JSON + figures bundle: ELBO traces, stability, held-out LL, k̂.
**Requirements:** R6, R7.
**Dependencies:** U1, U2, U3.
**Files:** `scripts/validate_vem.py` (new), `tests/test_scripts.py` (smoke test), `src/analysis/plots.py` (figure helpers if needed).
**Approach:** Follow `benchmark.py` conventions: `add_common_args`/`build_config`/`load_inputs` from `scripts/_runner.py`, `--json-out`, `--seeds`, `--n-restarts`, `--holdout-frac`, `--psis-draws`, `--n-jobs`; synthetic by default, real chains accepted via the standard input flags. JSON records config, prior spec, per-restart metrics, spreads, k̂ (with caveat string), held-out LL. Figures to `results/validation/`.
**Test scenarios:**
- CLI smoke (fast): tiny synthetic (K=2, T=100, 2 restarts, 50 PSIS draws) → exit 0, JSON exists with all top-level keys, figures written to tmp dir.
- Restart independence: restarts differ by seed → per-restart ELBOs not all identical.
**Verification:** smoke test in fast suite; full run at gate scale executed once and JSON committed under `results/validation/`.

---

## Verification Contract

- `python -m pytest -q -m "not slow"` green; new slow tests (gate-scale stability) pass in `python -m pytest -q`.
- `python -m scripts.validate_vem --synthetic --config dev --json-out results/validation/dev.json` exits 0 and writes a complete artifact.
- No inference-path regression: plan 1's gate re-run numbers unchanged (this plan adds post-hoc layers only).
- PG freeze audit: `git diff` touches no PG/iPMCMC/CSMC/Kalman module.

## Definition of Done

- U1–U4 landed; JSON + figures for the standard synthetic config committed.
- k̂ value known and recorded; if > 0.7, the run is still "done" but the stop-condition escalation to the user has been raised (variational-family enrichment is a new decision, not silent scope).
- Prior spec exists in exactly one place and is imported by both this plan's code and (later) plan 3.
- No abandoned experimental code.

---

## Documentation Notes (for /finish docs-updater)

- ARCHITECTURE.md §8 module map: add `laplace.py`, `analysis/validation.py`, `scripts/validate_vem.py`.
- ARCHITECTURE.md §12 Validation: add the samplerless validation ladder (ELBO/stability/held-out LL/PSIS-k̂ here; SBC/coverage in plan 3) and the k̂ > 0.7 escalation rule.
- STATUS.md: changelog row with first k̂/held-out-LL numbers.
- Paper (final /paper cycle): validation section leads with SBC + PSIS framing; cite Talts et al. 2018, Yao et al. 2018, Vehtari et al. 2024.
