# Project Status

> **Quick-update file.** Edit this when priorities, work items, or decisions change.
> Stable architecture detail lives in [ARCHITECTURE.md](ARCHITECTURE.md).

**Last updated:** 2026-06-15

---

## Current focus

**P0 — Speed:** `numba` on `kalman_step`, `joblib` for iPMCMC chains + K markets.
Stage 0 correctness prerequisites (P1 filter, `theta_w` fix, benchmark) are **DONE**;
speed work (Stage 1 / "C0") is next.

---

## Priority roadmap

Status key: `PLANNED` → `WIP` → `DONE`

| P | Work item | Status | Owner / notes |
|---|-----------|--------|---------------|
| P0 | Speed: numba + joblib | PLANNED | Hot path: `kalman.py`, `ipmcmc.py` |
| P1 | Pre-resolution filter (`--pre-resolution-days`) | DONE | Default 7 days before close; wired through `pull_data.py` |
| P2 | Half-prod inference runs for paper | PLANNED | `--n-iter 1500 --n-burnin 300 --n-particles 250` |
| P3 | Fix `theta_w` update; investigate negative `β_S` | WIP | `theta_w` RWMH fix DONE; `β_S` still open |
| P4 | Refreshed paper figures | PLANNED | Depends on P1–P2 |
| P5 | γ / s₀² sensitivity script | PLANNED | Synthetic grid only |
| P6 | Trading infrastructure | PLANNED | Filter-only CSMC, live ingest |

---

## Active work tracker

| Item | Status | File(s) |
|------|--------|---------|
| numba on `kalman_step` | PLANNED | `src/inference/kalman.py` |
| joblib parallel iPMCMC | PLANNED | `src/inference/ipmcmc.py` |
| joblib parallel K markets | PLANNED | `src/inference/particle_gibbs.py`, `ipmcmc.py` |
| Pre-resolution subsetting | DONE | `src/data/preprocess.py` (`filter_pre_resolution`), `scripts/pull_data.py` (`--pre-resolution-days`) |
| Approximate `theta_w` Gibbs | DONE | Per-wallet RWMH on logit scale under full logistic Z model; `parameter_updates.py` |
| Benchmark script | DONE | `scripts/benchmark.py` — wall-clock, cProfile cost breakdown, `--gate` synthetic eval |

---

## Changelog

Newest first. One line per meaningful change.

| Date | Change |
|------|--------|
| 2026-06-15 | Stage 1a: `numba.njit` `_kalman_step_all_combos` in `kalman.py`; CSMC inner loop uses one jitted call/step. Numerically identical (atol 1e-10 equivalence test); AUC unchanged (0.9550). Finding: at N=100 the Kalman arithmetic is only ~19% of PG wall-clock (not ~95%) — the bigger speed wins are jitting the *whole* CSMC per-step math and `joblib` over K markets. |
| 2026-06-15 | Stage 0 done: P1 pre-resolution filter; `theta_w` RWMH fix (full logistic, β≠0 correct); `scripts/benchmark.py` + `spearman_theta_w`. Synthetic gate PASS (pooled AUC 0.955, insiders top-3); full suite 206 passed. |
| 2026-06-14 | `agent_reference/` trimmed to ARCHITECTURE.md + STATUS.md only. |
| 2026-06-14 | Post-submission pivot: speed P0, trading path, half-prod canonical. Split living status into this file. |
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
