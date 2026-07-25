---
title: VEM Weighted-Logistic M-Step with Cauchy Prior - Plan
type: feat
date: 2026-07-23
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# VEM Weighted-Logistic M-Step with Cauchy Prior - Plan

**Sequence:** Plan 1 of 5 (2026-07-23 series). No upstream plan dependencies. Plans 2–5 depend on this one.

---

## Goal Capsule

Implement the weighted-logistic IRLS M-step for `beta_S`/`beta_Z` in the VEM fast path — they are currently **held fixed at 0.0** — with the Gelman et al. (2008) Cauchy(0, 2.5) prior from day one, centered covariates, and no free intercept (the `theta_w` Beta hierarchy carries the level). Also formally retire PG/iPMCMC from all new computation in STATUS.md.

**Authority hierarchy:** CLAUDE.md hard rules > agent_reference/ARCHITECTURE.md + STATUS.md > this plan. Math-symbol names (`beta_S`, `theta_w`, …) are intentional. The sequential PG path stays bit-exact — this plan must not touch `particle_gibbs.py`, `csmc.py`, `kalman.py`, `parameter_updates.py`, or `ipmcmc.py`.

**Stop conditions:** stop and surface if (a) any change would alter PG-path outputs, (b) the synthetic gate (pooled AUC ≥ 0.85, 100% insider recall@K) regresses and no parameter/initialization fix recovers it, or (c) IRLS divergence cannot be tamed by the Cauchy prior + step-halving.

---

## Product Contract

### Summary

The VEM engine (`src/inference/variational_em.py`) never learns the size and persistence coefficients of the insider propensity model — `logit(pi_Z) = logit(theta_w) + beta_S·log(S/S̄) + beta_Z·1{Z_prev}` runs with `beta_S = beta_Z = 0.0` defaults from `ModelParams`. This plan adds the standard M-step: IRLS on fractional Bernoulli targets `q(Z_j)`, immunized against separation by the Cauchy(0, 2.5) prior implemented as Gelman's approximate-EM modification of IRLS, with identifiability protected by centering and by refusing a free intercept.

### Problem Frame

Without a `beta` M-step the fast path cannot address the open P3 question (negative `beta_S` seen under PG), cannot claim the "weighted-logistic M-step" the paper's method section needs, and ignores trade size/persistence signal at inference time. Naive IRLS would diverge under separation (plausible: planted insiders trade large), and a free intercept would compete with the `theta_w` random-effect mean for the same variance.

### Requirements

- R1. `beta_S` and `beta_Z` are estimated each M-step by IRLS on weights/targets from `q(Z_j)`, with `logit(theta_w[wallet])` entering as a fixed offset (not an estimated coefficient).
- R2. Cauchy(0, 2.5) priors on standardized coefficients via the Gelman et al. (2008) approximate-EM IRLS modification; estimation never diverges, including under complete separation.
- R3. No free intercept in the logistic predictor; covariates are centered so the `theta_w` hierarchy carries the level.
- R4. Covariate standardization follows Gelman's convention — continuous `log_size_ratio` scaled to mean 0, sd 0.5; the `E[Z_prev]` term centered only — applied inside VEM, with constants stored on the output and coefficients back-transformed to the original scale for reporting.
- R5. The E-step uses the same centered covariates and the current `beta` estimates, so E- and M-steps see one consistent model.
- R6. The synthetic gate still passes (pooled AUC ≥ 0.85, 100% insider recall@K), and `beta` recovery on synthetic data with planted nonzero `beta_S` has the correct sign across ≥ 3 seeds.
- R8. The `theta_w` update is made coherent with the logistic model: once betas are nonzero, the current Beta-count update (`variational_em.py:176-185`) implicitly assumes `logit(pi_Z) = logit(theta_w)` and absorbs the size effect into `theta_w`, biasing `beta_S` toward 0/negative (the likely mechanism behind the open P3 artifact). Replace it with a per-wallet offset-adjusted update: 1-D penalized Newton on `logit(theta_w)` under the logistic likelihood with the Beta(a, b) prior and offset `beta_S·x_S + beta_Z·x_Z`, yielding a logit-normal (mean, var) per-wallet posterior from the 1-D curvature. Must reduce to the existing count update when `beta_S = beta_Z = 0` (regression anchor). Plan 3's `theta_w` coverage consumes the logit-normal intervals.
- R7. STATUS.md records full PG retirement: deferred PG bench stages cancelled, P2 re-anchored on VEM real-data runs, P3 `beta_S` investigation re-anchored on VEM estimates, decision row added. PG code and tests remain untouched as the cited historical baseline.

### Scope Boundaries

- **Out of scope:** parameter uncertainty for `beta` (plan 2's Laplace layer reuses the IRLS Fisher information built here); any PG/iPMCMC code change; any anonymous-mode intercept (plan 5 adds `alpha` for the Kalshi variant, where the no-intercept rule does not apply because there is no `theta_w`).
- **Deferred to follow-up work:** updating the Beta(a, b) hyperparameters themselves (currently fixed at 1.0/19.0 — keep fixed; revisit only if SBC in plan 3 shows miscalibrated `theta_w` coverage).

---

## Planning Contract

### Key Technical Decisions

- KTD1 — **`logit(theta_w)` is an offset, not a coefficient.** The IRLS design matrix has exactly two columns (centered size, centered `E[Z_prev]`). This is the concrete mechanism for "let `theta_w` carry the level" and removes the random-effect/fixed-effect identifiability competition.
- KTD2 — **Fractional targets, not sampled labels.** Each trade contributes a Bernoulli observation with target `q(Z_j)` and weight 1 (equivalently expected complete-data log-likelihood). Trade 0 is excluded (`Z_0 := 0` by construction).
- KTD3 — **Cauchy prior via Gelman's approximate-EM IRLS**, not a generic optimizer: at each IRLS step the prior contributes a data-augmentation/curvature term computed from the current coefficient value (per Gelman et al. 2008 §3). Standardize internally (sd 0.5 for the continuous covariate), place Cauchy(0, 2.5) on standardized coefficients, back-transform for reporting and for the E-step predictor.
- KTD4 — **Centering happens inside `variational_em.py` only.** `MarketData.log_size_ratio` and preprocessing are untouched, so the frozen PG path and all existing data artifacts are bit-identical. Centering constants are computed once per fit from the pooled dataset and stored on `VEMOutput` (plan 4's online scorer needs them at scoring time).
- KTD5 — **Slopes are approximately centering-invariant, not exactly** (methods-critic finding). Exact invariance would require every wallet's `theta_w` to absorb the same additive logit shift, but the Beta-prior shrinkage is wallet-specific (heavy below ~20 trades, ARCHITECTURE §9.5), so global centering perturbs fitted slopes in the sparse-wallet regime. State this in the docstring; test it directly (two different centering constants → betas equal within a loose tolerance, with the sparse-wallet caveat documented) rather than asserting equality. Recovery tests still compare back-transformed values.
- KTD6 — **IRLS safeguards:** cap iterations (~25), converge on relative coefficient change, step-halve when the penalized objective decreases. With the Cauchy prior this always terminates finitely (Gelman et al.: "always gives answers, even with complete separation").
- KTD7 — **ECM block order is fixed and stated:** per M-step, `theta_w` block first (offset-adjusted per R8, using the previous betas as offsets), then variances/transitions, then IRLS for betas using the freshly updated `theta_w` as offset. One consistent block-coordinate scheme; document that monotonicity holds blockwise under this order.
- KTD8 — **`E[Z_prev]` as a design covariate is a plug-in mean-field move with known attenuation risk on `beta_Z`** (regression-dilution: the binary variance of `Z_prev` is discarded). Accepted for this plan; the expected bias direction (|beta_Z| underestimated) is stated in the module docstring and probed by a dedicated test, so the paper does not overclaim.

### High-Level Technical Design

```
_vem_e_step (per market, sequential)          _vem_m_step (pooled)
  logit_pi = logit(theta_w[w])                  1. theta_w Beta update   (existing)
           + beta_S_int * x_S~                  2. variances/transitions (existing)
           + beta_Z_int * x_Z~     ──q(Z)──▶    3. NEW _update_beta_irls:
  where x_S~ = (x_S - m_S) * 0.5/s_S               design [x_S~, x_Z~], offset logit(theta_w)
        x_Z~ = E[Z_prev] - m_Z                     targets q(Z_j), Cauchy(0,2.5) prior
  (internal-scale betas)                           → beta_S_int, beta_Z_int → back-transform
```

Directional guidance, not implementation specification.

---

## Implementation Units

### U1. STATUS.md PG-retirement and roadmap rewrite

**Goal:** Record the resolved decision: PG/iPMCMC retired from all new computation; roadmap reflects the five-plan 2026-07-23 series.
**Requirements:** R7.
**Dependencies:** none (file-disjoint from U2–U4; parallelizable).
**Files:** `agent_reference/STATUS.md`.
**Approach:** Update Current focus (drop "Deferred bench stages… real-data half-prod runs" as PG work; new focus = this plan series). Roadmap: mark deferred PG bench stages / C4-on-PG CANCELLED; P2 becomes "VEM real-data runs"; P3 note "via VEM estimates once beta M-step lands". Add decisions-table row: "PG/iPMCMC retirement — CLOSED (2026-07-23): frozen as cited historical baseline (AUC 0.962, 3117.5 s); all new computation on the VEM fast path; validation via SBC/PSIS (Talts et al. 2018; Yao et al. 2018), never PG comparison." Add changelog row. Do not rewrite history rows.
**Test scenarios:** Test expectation: none — docs-only unit; verification is review against the resolved decisions listed here.
**Verification:** STATUS.md contains no remaining planned PG/iPMCMC computation; changelog and decisions rows added.

### U2. Covariate centering and scaling inside VEM

**Goal:** Centered/standardized covariates threaded through E- and M-steps; constants stored on `VEMOutput`.
**Requirements:** R3, R4, R5.
**Dependencies:** none.
**Files:** `src/inference/variational_em.py`, `tests/test_variational_em.py`.
**Approach:** Compute pooled `m_S`, `s_S` (over all markets' `log_size_ratio`) and running-mean center for the `E[Z_prev]` term once per fit; store `(m_S, s_S, m_Z)` on `VEMOutput` (new fields). E-step predictor (currently `variational_em.py:107-111`) uses internal-scale covariates and internal-scale betas. Guard `s_S > 0` (degenerate constant-size markets fall back to centering only).
**Patterns to follow:** existing `VEMOutput` dataclass (`variational_em.py:35-45`); Google docstrings per CODE_QUALITY.md.
**Test scenarios:**
- With `beta_S = beta_Z = 0`, outputs (`Z_prob`, `V_prob`, `X_mean`, `theta_w`, `elbo_trace`) are bit-identical to pre-change VEM on the same seed — centering is inert when betas are zero (regression anchor).
- Constant-size market (all `log_size_ratio` equal): no NaN/inf; fallback path exercised.
- Stored constants round-trip: `m_S` equals the pooled mean of inputs to machine precision.
**Verification:** fast suite green; the beta=0 bit-identity test passes against a pinned pre-change fixture (generate fixture before refactor).

### U3. IRLS M-step with Cauchy(0, 2.5) prior + coherent `theta_w` update

**Goal:** `beta_S`, `beta_Z` estimated every M-step; separation-proof; `theta_w` update offset-adjusted so the two blocks don't fight over the same variance.
**Requirements:** R1, R2, R3, R6 (estimation half), R8.
**Dependencies:** U2.
**Files:** `src/inference/variational_em.py`, `tests/test_variational_em.py`.
**Approach:** Two coupled changes, one unit (they are incoherent apart — architecture-critic MAJOR):
(1) New private helper (e.g. `_update_beta_irls`) called from `_vem_m_step` (`variational_em.py:151-259`): pooled design over all markets/trades j ≥ 1, offset `logit(theta_w[wallet])` at current per-wallet posterior means, weights/targets from `q(Z_j)`; 2-parameter Newton/IRLS with the Gelman prior term; safeguards per KTD6; back-transform per U2; expose final IRLS Fisher information (internal scale) on `VEMOutput` for plan 2. Warn (don't raise) at the iteration cap.
(2) Replace the Beta-count `theta_w` update (`variational_em.py:176-185`) with the R8 offset-adjusted 1-D penalized Newton per wallet, storing per-wallet logit-normal (mean, var); `VEMOutput.theta_w` stays the posterior-mean vector (back-transformed) so downstream ranking code is unchanged, with the (mean, var) pairs added as a new field. Block order per KTD7.
**Test scenarios:**
- Beta-zero reduction: with `beta_S = beta_Z = 0`, the new `theta_w` update matches the old Beta-count update's posterior means (tight tolerance) — the R8 regression anchor.
- Recovery: synthetic data with `beta_S = 1.0`, `beta_Z = 1.5` (generator `synthetic.py:135-139`) → back-transformed estimates correct sign and within ±35% relative across 3 seeds (tightened from ±50%: the offset-adjusted `theta_w` update exists precisely to remove the absorption bias, so demand more).
- Absorption-bias probe: same synthetic data fit with the OLD count-based `theta_w` update forced (test-only flag or monkeypatch) → `beta_S` estimate is materially smaller than under the new update, demonstrating the R8 mechanism (and documenting the P3 connection).
- Attenuation signature (KTD8): sweep T ∈ {300, 1000, 3000} at fixed density → recovered `beta_Z` under-estimates the planted value with a stable-or-shrinking gap as T grows; assert direction, record magnitude.
- Two-centerings test (KTD5): fit with pooled-mean centering vs a shifted constant → back-transformed slopes equal within ~10% on a dense-wallet fixture; documented-looser on a sparse-wallet fixture.
- Null case: data with `beta_S = beta_Z = 0` → estimates shrink near 0 (|beta| < 0.5 internal scale), no divergence.
- Separation stress: `q(Z)` perfectly separated on size → finite estimates, IRLS terminates under cap.
- Monotone objective: penalized expected log-likelihood non-decreasing across IRLS iterations (step-halving guarantee).
- ELBO trace finite; terminal ELBO ≥ pre-change terminal ELBO on the standard fixture.
**Verification:** all new tests pass; `test_vem_elbo_non_decreasing`, `test_vem_z_prob_discriminates_insiders` still pass.

### U4. Gate re-run, seed stability, and runtime check

**Goal:** Prove the gate still passes and the fast path stays fast; record results.
**Requirements:** R6.
**Dependencies:** U2, U3.
**Files:** `tests/test_variational_em.py` (stability test), `results/` (bench JSON), `agent_reference/STATUS.md` (one changelog line with new gate numbers).
**Approach:** Run `scripts/benchmark.py --method vem --gate` on the standard synthetic config across ≥ 3 seeds. Acceptance: pooled AUC ≥ 0.85, 100% insider recall@K (the post-tau criteria — do not use Kendall tau; see `docs/solutions/best-practices/kendall-tau-acceptance-criterion-sparse-rankings.md`). Runtime: VEM mean wall-clock within ~15% of the 68.8 s baseline at gate scale (IRLS on 2 parameters is cheap; a large regression means a bug).
**Test scenarios:**
- Multi-seed stability (marked slow): 3 seeds → sign of each beta consistent; AUC spread < 0.05.
**Verification:** bench JSON committed under `results/`; gate PASS recorded in STATUS changelog.

---

## Verification Contract

- Fast suite: `python -m pytest -q -m "not slow"` green (dispatch `test-triager` in /finish).
- Full suite including slow VEM tests: `python -m pytest -q` green.
- Gate: `python -m scripts.benchmark --method vem --gate` → pooled AUC ≥ 0.85, 100% recall@K, ≥ 3 seeds.
- Bit-exactness sentinel: existing PG tests (`test_particle_gibbs.py`, `test_csmc.py`, `test_kalman.py`, `test_parameter_updates.py`) pass unmodified — this plan must not touch their modules.
- No-speed-regression: VEM gate-scale mean within ~15% of 68.8 s.

## Definition of Done

- All four units land (U1 may land first and independently).
- R1–R7 verified per unit; gate numbers recorded in STATUS.md changelog.
- No changes outside `variational_em.py`, its tests, STATUS.md, and results artifacts.
- No leftover experimental code paths (e.g., a dead un-centered branch).

---

## Documentation Notes (for /finish docs-updater)

- ARCHITECTURE.md §6 table: VEM row — "moment-matched M-step" → "moment-matched + IRLS-Cauchy logistic M-step (Gelman et al. 2008); beta_S/beta_Z estimated; theta_w offset-adjusted (logit-normal)".
- ARCHITECTURE.md §5.3: note that VEM centers covariates internally (model spec unchanged).
- **PG-canonical reconciliation (architecture-critic DOC GAP):** ARCHITECTURE §2 roadmap, §3 decision #3 ("half-prod default" as a PG/iPMCMC run), §6 tables, and §10 default-run block still present PG/iPMCMC as the canonical engine. After U1 lands, docs-updater must reconcile ARCHITECTURE with the retirement decision (VEM canonical; PG rows marked historical/frozen) — STATUS.md alone is not enough.
- Centering-invariance caveat (approximate, sparse-wallet regime) belongs in both the module docstring and the paper's methods section.
- STATUS.md: handled by U1/U4.
- Paper (later /paper cycle): M-step subsection + Gelman et al. (2008) citation; expected attenuation direction for `beta_Z` (KTD8) stated honestly.
