# Project Status

> **Quick-update file.** Edit this when priorities, work items, or decisions change.
> Stable architecture detail lives in [ARCHITECTURE.md](ARCHITECTURE.md).

**Last updated:** 2026-07-04

> **Session handoff:** See [HANDOFF_FAST_INSIDER_DETECTION.md](HANDOFF_FAST_INSIDER_DETECTION.md) for full gate results, uncommitted changes, and next steps (2026-07-03 orchestration session).

---

## Current focus

**P0 — Stage 2 gate CLOSED (2026-07-04):** C1 VEM **GATE PASS** (AUC 0.885, 68.8 s mean, 100% recall@K across seeds). C0 PG half-prod **GATE PASS** (N=250/1500-iter: 3117.5 s, AUC 0.962). Kendall τ criterion invalidated by PG-vs-PG control (τ=0.787). **New acceptance criteria:** 100% insider recall@K + ≥0.85 pooled AUC (per-market floors + descriptive top-K overlap). Filter-only ablation **GATE FAIL** (AUC 0.524). **Next:** Deferred bench stages (N=100 PG control, C4 full-scale eval, gated iPMCMC ablation); real-data half-prod runs; paper bench table.

---

## Priority roadmap

Status key: `PLANNED` → `WIP` → `DONE`

| P | Work item | Status | Owner / notes |
|---|-----------|--------|---------------|
| P0 | Stage 2 gate: VEM vs PG benchmark | DONE | C1 VEM (AUC 0.885, 68.8s), C0 PG (AUC 0.962, 3117.5s); new criteria adopted |
| P1 | Pre-resolution filter (`--pre-resolution-days`) | DONE | Default 7 days before close; wired through `pull_data.py` |
| P2 | Half-prod inference runs for paper | PLANNED | `--n-iter 1500 --n-burnin 300 --n-particles 250` on real data |
| P3 | Fix `theta_w` update; investigate negative `β_S` | WIP | `theta_w` RWMH fix DONE; `β_S` still open |
| P4 | Refreshed paper figures + Pareto curve | DONE | Pareto (AUC-vs-wall-clock) committed; bench table pending |
| P5 | γ / s₀² sensitivity script | PLANNED | Synthetic grid only |
| P6 | Paper refs + narrative update | DONE | +11 BibTeX entries; narrative shifted to C1 core, iPMCMC ablation |

---

## Active work tracker

| Item | Status | File(s) |
|------|--------|---------|
| numba `_kalman_step_all_combos` | DONE | `src/inference/kalman.py` |
| joblib parallel K markets | DONE | `src/inference/particle_gibbs.py` (`n_jobs` field in `InferenceConfig`) |
| filter-only screening mode | DONE (ablation FAIL) | `src/analysis/prefilter.py` — AUC 0.524 at K=10/T=2000 |
| Variational EM (C1) | DONE (gate PASS) | `src/inference/variational_em.py` — ADF E-step + moment-matched M-step |
| Stage 2 gate | DONE | C1 VEM AUC 0.885, C0 PG AUC 0.962; C3/C2 cancelled |
| Pre-resolution subsetting | DONE | `src/data/preprocess.py`, `scripts/pull_data.py` |
| Approximate `theta_w` Gibbs | DONE | `src/inference/parameter_updates.py` |
| Benchmark script (--method support) | DONE | `scripts/benchmark.py` — supports `{pg,vem,filter,ipmcmc}` |
| --n-jobs market parallelism | DONE | `scripts/run_pg.py` — `dataclasses.replace` on preset config |
| Pareto figure + bench tooling | DONE | `scripts/pareto.py` → `results/figures/pareto.png` + CSV |

---

## Changelog

Newest first. One line per meaningful change.

| Date | Change |
|------|--------|
| 2026-07-04 | Stage 2 gate CLOSED: C1 VEM gate PASS (AUC 0.885, 68.8s, 100% recall@K); C0 PG gate PASS (AUC 0.962, 3117.5s). Kendall τ criterion invalidated (PG-vs-PG ctrl τ=0.787). New criteria: 100% recall + ≥0.85 AUC. Filter-only ablation FAIL (AUC 0.524). Deferred: C4 full-scale, gated iPMCMC. Paper: +11 refs, narrative to C1 core. Scripts: benchmark --method {pg,vem,filter,ipmcmc}, run_pg --n-jobs, pareto.py. |
| 2026-06-26 | Stage 2 C1: `src/inference/variational_em.py` — single-mode ADF E-step + moment-matched M-step; `VEMOutput` dataclass; 6 non-slow tests pass. |
| 2026-06-26 | Stage 1 C0: `filter_screen` + `_filter_screen_worker` in `particle_gibbs.py` — fast per-wallet Z_prob shortlist tier; 4 new tests. |
| 2026-06-26 | Stage 1 C0: `joblib.Parallel` over K markets in `particle_gibbs.py`; `n_jobs: int = 1` added to `InferenceConfig`; sequential path bit-exact unchanged; 2 new tests. |
| 2026-06-15 | Stage 1a: `numba.njit` `_kalman_step_all_combos` in `kalman.py`; AUC unchanged (0.9550). |
| 2026-06-15 | Stage 0 done: P1 pre-resolution filter; `theta_w` RWMH fix; `scripts/benchmark.py` + `spearman_theta_w`. Synthetic gate PASS (pooled AUC 0.955, insiders top-3); full suite 206 passed. |
| 2026-06-14 | `agent_reference/` trimmed to ARCHITECTURE.md + STATUS.md only. |
| 2026-06-14 | Post-submission pivot: speed P0, trading path, half-prod canonical. |
| 2026-06-14 | Created `ARCHITECTURE.md` as agent-canonical doc. |

---

## Resolved decisions (quick reference)

| Topic | Decision |
|-------|----------|
| Stage 2 gate — C1 VEM promotion | **CLOSED (2026-07-04):** C1 VEM gate PASS; C3 twisted-CSMC NOT implemented; C2 rSLDS moot. |
| Kendall τ acceptance criterion | **INVALIDATED:** PG-vs-PG control (N=250, 1500-iter, different seed) → τ=0.787 (below 0.85 threshold). Replacement: 100% insider recall@K (top_cutoff rule) across ≥3 synthetic seeds + pooled AUC ≥0.85 (per-market floors); weighted τ / top-K overlap reported descriptively. |
| Filter-only screening ablation | **GATE FAIL:** Pooled AUC 0.524 at K=10/T=2000. Kept as negative-result ablation row. |
| Bounded/absorbing price model | **SKIPPED PERMANENTLY:** P1 filter sufficient per Stage 0. |
| Model | Baseline spec; refinements OK if synthetic tests pass |
| Inference default | Half-prod, not full prod |
| Data source | Polymarket Data API only (no Goldsky) |
| CSMC reference index | 0 (code authoritative) |
| Doc hierarchy | `ARCHITECTURE.md` + this file for agents |
| Entrypoints | `scripts/` CLIs only |
