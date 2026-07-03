# Project Status

> **Quick-update file.** Edit this when priorities, work items, or decisions change.
> Stable architecture detail lives in [ARCHITECTURE.md](ARCHITECTURE.md).

**Last updated:** 2026-07-03

> **Session handoff:** See [HANDOFF_FAST_INSIDER_DETECTION.md](HANDOFF_FAST_INSIDER_DETECTION.md) for full gate results, uncommitted changes, and next steps (2026-07-03 orchestration session).

---

## Current focus

**P0 — Close Stage 2 gate + evaluation:** C1 VEM passes AUC (0.885) and <5 min on synthetic batch proxy; global Kendall τ vs PG ≈ 0 (likely wrong metric — see handoff §5). C0 PG half-prod takes **64 min** (Stage 1 wall-clock gate FAIL). **Next:** PG-vs-PG τ control, Pareto figure from bench JSONs, paper refs; implement C3 only if C1 tuning fails.

---

## Priority roadmap

Status key: `PLANNED` → `WIP` → `DONE`

| P | Work item | Status | Owner / notes |
|---|-----------|--------|---------------|
| P0 | Stage 2 gate: VEM vs PG benchmark | PLANNED | Run `scripts/benchmark.py --gate`; target AUC>=0.85, batch<5min |
| P1 | Pre-resolution filter (`--pre-resolution-days`) | DONE | Default 7 days before close; wired through `pull_data.py` |
| P2 | Half-prod inference runs for paper | PLANNED | `--n-iter 1500 --n-burnin 300 --n-particles 250` |
| P3 | Fix `theta_w` update; investigate negative `β_S` | WIP | `theta_w` RWMH fix DONE; `β_S` still open |
| P4 | Refreshed paper figures + Pareto curve | PLANNED | AUC-vs-wall-clock across PG, C0, C1 candidates |
| P5 | γ / s₀² sensitivity script | PLANNED | Synthetic grid only |
| P6 | Paper refs + narrative update | PLANNED | Add C1/C3 BibTeX; shift iPMCMC to ablation |

---

## Active work tracker

| Item | Status | File(s) |
|------|--------|---------|
| numba `_kalman_step_all_combos` | DONE | `src/inference/kalman.py` |
| joblib parallel K markets | DONE | `src/inference/particle_gibbs.py` (`n_jobs` field in `InferenceConfig`) |
| filter-only screening mode | DONE | `src/inference/particle_gibbs.py` (`filter_screen`, `_filter_screen_worker`) |
| Variational EM (C1) | DONE | `src/inference/variational_em.py` — ADF E-step + moment-matched M-step |
| Stage 2 gate | PLANNED | Run synthetic AUC gate; pick C1 or implement C3 |
| Pre-resolution subsetting | DONE | `src/data/preprocess.py`, `scripts/pull_data.py` |
| Approximate `theta_w` Gibbs | DONE | `src/inference/parameter_updates.py` |
| Benchmark script | DONE | `scripts/benchmark.py` |

---

## Changelog

Newest first. One line per meaningful change.

| Date | Change |
|------|--------|
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
| Model | Baseline spec; refinements OK if synthetic tests pass |
| Inference default | Half-prod, not full prod |
| Data source | Polymarket Data API only (no Goldsky) |
| CSMC reference index | 0 (code authoritative) |
| Doc hierarchy | `ARCHITECTURE.md` + this file for agents |
| Entrypoints | `scripts/` CLIs only |
