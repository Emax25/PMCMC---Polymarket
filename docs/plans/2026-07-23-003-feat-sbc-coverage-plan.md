---
title: Simulation-Based Calibration and Coverage - Plan
type: feat
date: 2026-07-23
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Simulation-Based Calibration and Coverage - Plan

**Sequence:** Plan 3 of 5. Depends on plan 1 (beta M-step) and plan 2 (`PhiPosterior` Laplace layer, prior spec). Front-runs the paper's headline validation claim. **File-contention note:** plan 5 also edits `src/data/synthetic.py` (anonymous mode) — land this plan's generator changes before plan 5 starts.

---

## Goal Capsule

Implement simulation-based calibration (Talts et al. 2018) and credible-interval coverage for the VEM+Laplace pipeline: draw `phi` from the model priors, simulate datasets, fit, and check that rank statistics are uniform and that nominal 90% intervals cover in [0.85, 0.95] — for `phi` (via the Laplace layer) and for latent quantities (`theta_w` Beta posteriors, `Z` discrimination).

**Authority hierarchy:** CLAUDE.md hard rules > agent_reference docs > this plan. PG freeze is hard. SBC validity rule: simulation priors and inference priors must be the same object in code (plan 2's single prior spec).

**Stop conditions:** surface if (a) rank histograms show the ∪-shape (overconfidence) or coverage < 0.85 for a headline parameter after the documented remedies (more sims, wider synthetic size) — that finding changes the paper's claims and may trigger variational-family enrichment, a user decision; (b) total runtime at the chosen replicate count exceeds ~4 h on this machine — reduce per-sim size, don't silently reduce replicates below 100.

---

## Product Contract

### Summary

A harness (`scripts/sbc.py`) runs N_sims ≥ 200 replicates of prior-draw → simulate → VEM fit → Laplace posterior → rank + coverage bookkeeping, parallelized with joblib and resumable; an analysis step turns the accumulated ranks into rank-ECDF plots with confidence bands, a coverage table, and a JSON summary consumed by the paper. This is the substitute for "compare to the exact posterior" and the validation section's lead evidence.

### Problem Frame

SBC and coverage-under-simulation are the two checks a computational-statistics referee most directly accepts in place of an MCMC gold standard. They are cheap only if engineered deliberately: hundreds of fits, so per-fit size and parallelism are first-class design parameters, and interruption must not lose completed replicates.

### Requirements

- R1. Prior draws come from plan 2's single prior spec (`PhiPrior`) — the same densities the refactored M-step optimizes against (plan 2 R8) — with **no truncation anywhere** (methods-critic BLOCKER: truncating Cauchy draws at |beta| ≤ 5 excludes ≈30% of prior mass and breaks rank uniformity even for a perfect posterior). Extreme-beta replicates are handled by plan 1's separation-proof IRLS plus the R5 degenerate-replicate flagging, never by changing the simulation prior.
- R1b. The generator gains a **prior-predictive mode** (architecture-critic MAJOR): `generate_market` currently plants insiders by forcing wallets 0..n_insider into Beta(9,1) propensities with upweighted trade frequency (`src/data/synthetic.py:89-110`) — that is a test-bench distribution, not the model's prior-predictive. SBC uses a mode where ALL `theta_w ~ Beta(a, b)`, no forced insiders, no frequency upweighting; the existing planted-insider mode is untouched for gate/recovery tests.
- R2. Per replicate: draw `phi`, generate a prior-predictive dataset (`src/data/synthetic.py:52` entry points) at a configurable reduced size (default ~K=3, T=400, ~30 wallets), fit VEM (+ Laplace), and record: rank of each true `phi` component among L i.i.d. posterior draws (L = 999 drawn directly — no thinning; Talts-style thinning exists to decorrelate MCMC draws and would only coarsen i.i.d. Gaussian draws, methods-critic), 90%-interval hit/miss per component, per-wallet `theta_w` interval hits for a fixed subsample of wallets (logit-normal intervals from plan 1 R8), and pooled Z AUC.
- R3. Replicates run under joblib (`--n-jobs`), each with an independent seeded `default_rng` stream; results append to an on-disk store (one row per replicate) so a killed run resumes with `--resume`.
- R4. Analysis produces: rank-ECDF (or histogram) plots with ~95% uniformity bands per `phi` component, a coverage table (target within [0.85, 0.95] for nominal 90%), `theta_w` aggregate coverage, and a JSON summary; figures under `results/figures/sbc/`.
- R5. Degenerate replicates (VEM non-convergence, Laplace fallback triggered) are recorded with a flag, not dropped silently; the summary reports their rate.
- R6. Guardrail: with N_sims = 200 at default size, wall-clock on this machine stays under ~4 h using `--n-jobs 8` (per-fit target: seconds at reduced size).

### Scope Boundaries

- **Out of scope:** VSBC variants beyond rank-SBC (cite, don't build); SBC on real data (meaningless — SBC is prior-predictive); fixing any miscalibration found (that is a finding to escalate, not silent rework); PG anything.
- **Deferred to follow-up work:** ModrÃ¡k-style test-quantity extensions (joint/likelihood-weighted quantities) if referees ask.

---

## Planning Contract

### Key Technical Decisions

- KTD1 — **Reduced per-sim size is a feature, not a compromise.** SBC checks calibration of the procedure, not headline AUC; K=3/T=400 keeps 200 fits tractable (target < 4 h; see benchmark-ops lesson: launch long runs detached via `setsid nohup`, tail a log). Size is a CLI knob so a larger confirmation run is possible later.
- KTD2 — **Ranks from i.i.d. draws, no thinning:** L = 999 draws directly from the (i.i.d.-sampling) Laplace/logit-normal posteriors, rank in {0..L}; uniformity assessed with ECDF confidence bands (arviz has plotting support; fall back to a chi-square bin test if needed).
- KTD3 — **Latent-side SBC is targeted, not exhaustive:** `theta_w` interval coverage — under plan 1's offset-adjusted update the per-wallet posterior is an approximate logit-normal, so this coverage tests that approximation chain, not an exact conjugate identity (state as such) — plus pooled Z AUC as a health metric. Full per-trade latent SBC (X paths) is out of scope — dimension too high to be informative at this budget.
- KTD4 — **Storage: append-only JSONL** (one line per replicate: seed, phi_true, ranks, hits, flags, timings) under `results/sbc/`; analysis is a separate pass reading the JSONL — resumability and re-analysis for free.
- KTD5 — **Prior-spec consistency is enforced by import**, not duplication: `scripts/sbc.py` imports `PhiPrior`; the generator gains an optional `params_from_prior(prior, rng)` helper rather than hand-rolled draws in the script (repo rule: logic in `src/`).

---

## Implementation Units

### U1. Prior-draw helper and prior-predictive generator mode

**Goal:** Draw a valid `ModelParams` from `PhiPrior` (untruncated); generator gains the prior-predictive mode.
**Requirements:** R1, R1b.
**Dependencies:** plan 2 U1 (`PhiPrior`).
**Files:** `src/data/synthetic.py` (helper + mode), `tests/test_synthetic.py`.
**Approach:** `params_from_prior(prior, rng) -> ModelParams`: draws variances (InvGamma), q's (Beta), betas (Cauchy, untruncated per R1), fixed gamma/s0_2/a/b as constants; seed order documented (generator convention `synthetic.py:67-68`). Prior-predictive mode per R1b: a flag/entry point that draws all `theta_w ~ Beta(a, b)` and uses uniform wallet trade assignment (no insider forcing at `synthetic.py:89-110`); planted-insider mode untouched.
**Test scenarios:**
- Validity: 500 draws → all params in domain (variances > 0, q in (0,1)); no NaN; occasional large |beta| present (untruncated — e.g. max over 500 draws exceeds 5).
- Determinism: same rng seed → identical draws.
- Prior-predictive property: with fixed phi, empirical mean of drawn `theta_w` over many wallets ≈ a/(a+b); no wallet cluster with forced high propensity; wallet trade-count distribution shows no insider upweighting.
- Planted-insider mode regression: existing generator tests pass unchanged.
**Verification:** tests green; helper and mode used by U2 only through import.

### U2. SBC harness `scripts/sbc.py`

**Goal:** Resumable, parallel replicate loop writing JSONL.
**Requirements:** R2, R3, R5, R6.
**Dependencies:** U1; plan 2 U1 (Laplace).
**Files:** `scripts/sbc.py` (new), `src/analysis/sbc.py` (new — replicate logic), `tests/test_scripts.py` (smoke), `tests/test_sbc.py` (new).
**Approach:** `run_replicate(sim_seed, size_cfg, prior) -> dict` in `src/analysis/sbc.py` (draw, simulate, fit, rank per KTD2, hits, flags, elapsed). CLI: `--n-sims`, `--n-jobs`, `--sim-K/T/wallets`, `--posterior-draws` (L, default 999, i.i.d., no thinning), `--out` (JSONL), `--resume` (skips completed seeds found in the file), `--seed-base`. joblib over replicates; each replicate is single-threaded internally (avoid nested parallelism).
**Test scenarios:**
- Replicate unit: one replicate at tiny size (K=2, T=100) returns all keys, ranks in {0..L}, runs in seconds.
- Resume: run 4 sims, kill file after 2 (simulate by truncating), `--resume` completes exactly the missing 2 (seed set equality).
- Parallel equivalence: `--n-jobs 1` vs `--n-jobs 2` produce the same set of rows (order-insensitive) for fixed seeds.
- Failure capture: monkeypatch VEM to raise on one seed → row written with failure flag, run continues.
**Verification:** fast smoke in CI; a real N=200 run launched detached (`setsid nohup`, per benchmark-ops lesson) with `results/sbc/*.jsonl` committed.

### U3. Rank-uniformity and coverage analysis + figures

**Goal:** Turn JSONL into the paper's calibration evidence.
**Requirements:** R4, R5.
**Dependencies:** U2 (format), file-disjoint from U1 — U3 can be built against a fixture JSONL in parallel with U2.
**Files:** `src/analysis/sbc.py` (analysis functions), `scripts/sbc.py` (`--analyze` mode or separate `--report`), `src/analysis/plots.py` (figure helpers), `tests/test_sbc.py`.
**Approach:** Per-component rank-ECDF with simultaneous confidence bands (arviz if available at pinned version; else binomial bin bands), coverage table with binomial CIs, failure-rate line, JSON summary (`results/sbc/summary.json`). Interpretation legend in the figure caption data (∪-shape = overconfident, ∩ = underconfident, slope = bias) so the paper figure is self-explanatory.
**Test scenarios:**
- Calibrated fixture: synthetic JSONL with ranks drawn uniform → uniformity not rejected; coverage ≈ 0.9 within tolerance.
- Miscalibrated fixture: ranks concentrated at extremes → uniformity rejected; coverage flagged out of [0.85, 0.95].
- Failure accounting: fixture with flagged rows → summary reports rate; flagged rows excluded from ranks but counted.
**Verification:** figures render at fixture scale in tests (tmp dir); real-run figures land in `results/figures/sbc/`.

---

## Verification Contract

- `python -m pytest -q -m "not slow"` green (harness smoke at tiny size included).
- Real evidence run: `setsid nohup python -m scripts.sbc --n-sims 200 --n-jobs 8 ... &` completes; summary JSON shows per-component coverage and uniformity results; failure rate < 5%.
- Prior-consistency audit: grep confirms `PhiPrior` is imported (not re-declared) by `src/analysis/sbc.py` and `src/data/synthetic.py` helper.
- PG freeze audit: no PG/iPMCMC imports anywhere in new code.

## Definition of Done

- U1–U3 landed; 200-replicate run completed and artifacts committed (`results/sbc/` JSONL + summary, `results/figures/sbc/`).
- **Production-representative confirmation run (methods-critic, required not optional):** a smaller-replicate run (e.g. 50 sims) at production-representative per-sim size (T ≈ 2000) completed and reported alongside the reduced-size run, so the paper's calibration claim isn't confined to the tiny-data regime where `theta_w` is prior-dominated.
- Coverage/uniformity outcomes recorded in STATUS.md changelog — pass or fail; a fail is a reported finding plus escalation, not a blocker to "done".
- No abandoned experimental code.

---

## Documentation Notes (for /finish docs-updater)

- ARCHITECTURE.md §8: add `src/analysis/sbc.py`, `scripts/sbc.py`; §12: SBC/coverage joins the validation ladder with the [0.85, 0.95] rule.
- STATUS.md: changelog + (if miscalibration) a decisions-table row for the escalation.
- Paper (final /paper cycle): SBC figure + coverage table lead the validation section; cite Talts et al. 2018 and Modrák et al. 2023.
