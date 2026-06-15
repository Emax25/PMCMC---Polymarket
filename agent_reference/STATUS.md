# Project Status

> **Quick-update file.** Edit this when priorities, work items, or decisions change.
> Stable architecture detail lives in [ARCHITECTURE.md](ARCHITECTURE.md).

**Last updated:** 2026-06-14

---

## Current focus

**P0 — Speed:** `numba` on `kalman_step`, `joblib` for iPMCMC chains + K markets.

---

## Priority roadmap

Status key: `PLANNED` → `WIP` → `DONE`

| P | Work item | Status | Owner / notes |
|---|-----------|--------|---------------|
| P0 | Speed: numba + joblib | PLANNED | Hot path: `kalman.py`, `ipmcmc.py` |
| P1 | Pre-resolution filter (`--pre-resolution-days`) | PLANNED | Default target: 7 days before close |
| P2 | Half-prod inference runs for paper | PLANNED | `--n-iter 1500 --n-burnin 300 --n-particles 250` |
| P3 | Fix `theta_w` update; investigate negative `β_S` | PLANNED | After P2 chains exist |
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
| Pre-resolution subsetting | PLANNED | `src/data/preprocess.py`, `scripts/pull_data.py` |
| Approximate `theta_w` Gibbs | PLANNED | `src/inference/parameter_updates.py` |
| Benchmark script | PLANNED | `scripts/benchmark.py` (not yet created) |

---

## Changelog

Newest first. One line per meaningful change.

| Date | Change |
|------|--------|
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
