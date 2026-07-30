# Project Status

> **Quick-update file.** Edit this when priorities, work items, or decisions change.
> Stable architecture detail lives in [ARCHITECTURE.md](ARCHITECTURE.md).

**Last updated:** 2026-07-29

> **Session handoff:** See [HANDOFF_FAST_INSIDER_DETECTION.md](HANDOFF_FAST_INSIDER_DETECTION.md) for full gate results, uncommitted changes, and next steps (2026-07-03 orchestration session).

---

## Current focus

**P0 — Stage 2 gate CLOSED (2026-07-04):** C1 VEM **GATE PASS** (AUC 0.885, 68.8 s mean, 100% recall@K across seeds). C0 PG half-prod **GATE PASS** (N=250/1500-iter: 3117.5 s, AUC 0.962). Kendall τ criterion invalidated by PG-vs-PG control (τ=0.787). **New acceptance criteria:** 100% insider recall@K + ≥0.85 pooled AUC (per-market floors + descriptive top-K overlap). Filter-only ablation **GATE FAIL** (AUC 0.524).

**Current focus (2026-07-29):** PG/iPMCMC retired from all new computation — frozen as cited historical baseline (see Resolved decisions); VEM is the canonical engine. Plan 1 (logistic M-step) and plan 2 (validation suite) both **DONE**, with caveats that gate the rest of the series: beta estimation is opt-in and FAILS the gate when on; the committed validation artifacts are pre-convergence; PSIS k̂ = 5.82 (dev) / 24.0 (gate) has fired the k̂ > 0.7 stop condition, and the cause is a mis-centred/mis-scaled proposal, not a too-simple variational family. **Plan 3 (SBC/coverage): CODE LANDED, EVIDENCE BLOCKED** — the prior-predictive generator, resumable harness and rank/coverage analysis all ship (`src/analysis/sbc.py`, `scripts/sbc.py`), but `default_sbc_prior()` fails fast on the improper `tau2` IG(1e-9, 1e-9), so the 200-replicate run and the production-size confirmation run stay open until P8/P11 are resolved (user-owned). **Plan 4 (online scorer + ingestion v2): DONE** (U1–U6) — per-trade P(Z|D≤i) via `OnlineScorer` over `ADFFilter`, RTDS live capture, replay/live streaming scorer. Plan 5 (Kalshi variant) unchanged; replay mode is its no-lookahead evaluation substrate.

---

## Priority roadmap

Status key: `PLANNED` → `WIP` → `DONE`

| P | Work item | Status | Owner / notes |
|---|-----------|--------|---------------|
| P0 | Stage 2 gate: VEM vs PG benchmark | DONE | C1 VEM (AUC 0.885, 68.8s), C0 PG (AUC 0.962, 3117.5s); new criteria adopted |
| P1 | Pre-resolution filter (`--pre-resolution-days`) | DONE | Default 7 days before close; wired through `pull_data.py` |
| P2 | VEM real-data runs | PLANNED | Half-prod real-data inference via VEM fast path (was PG spec; PG retired 2026-07-23) |
| P3 | Fix `theta_w` update; investigate negative `β_S` | WIP | `theta_w` RWMH fix DONE; beta M-step landed 2026-07-24. Key finding: spurious `β_S` ≈ −0.40 arises on beta=0 data whenever q(Z) is uninformative (ADF E-step cannot identify Z on the synthetic generator) — likely the same mechanism as the PG-era negative-`β_S` artifact. Beta estimation therefore ships opt-in (`estimate_betas=False`); with `--estimate-betas` ON the synthetic gate FAILS (pooled AUC 0.547 vs 0.9435 off), so plan 1's gate PASS validates the beta-fixed path only. Real-data verdict still open |
| P4 | Refreshed paper figures + Pareto curve | DONE | Pareto (AUC-vs-wall-clock) committed; bench table filled (355fa0a) |
| P5 | γ / s₀² sensitivity script | PLANNED | Synthetic grid only |
| P6 | Paper refs + narrative update | DONE | +11 BibTeX entries; narrative shifted to C1 core, iPMCMC ablation |
| P7 | Deferred PG bench stages (N=100 PG control, C4 full-scale eval, gated iPMCMC ablation) | CANCELLED | PG/iPMCMC retired 2026-07-23; frozen as historical baseline (AUC 0.962, 3117.5 s) |
| P8 | Laplace/PSIS foundation: right information matrix + a real mode | PLANNED | User-owned scope decision. `PhiPosterior` uses expected-complete-data (ECM) curvature, not observed information (Louis 1982), and the VEM fixed point is not stationary for the PSIS target. Options: Louis-identity observed information; optimize the ADF marginal directly + numerical Hessian; or move SBC to the ranking output. **Blocks plan 3's EVIDENCE RUNS** — the SBC harness itself is landed (P15) and refuses to run: `default_sbc_prior()` fails fast rather than draw from the improper `tau2` prior |
| P9 | `SS_v` smoothed moments + lag-one cross-covariance | PLANNED | `src/inference/variational_em.py` uses filtered rather than smoothed moments and omits `-2Cov(X_t, X_{t-1})`; most likely cause of the `sigma2` mis-centring. Inference-path change — would move gate numbers |
| P10 | `sigma2_1 = max(sigma2_1, sigma2_0)` order clamp | PLANNED | Binds exactly at every fitted point; V regime non-identified there. Identifiability/model decision, not a bug fix |
| P11 | Weakly-informative `tau2` prior | PLANNED | `PhiPrior` IG(1e-9, 1e-9) is numerically improper Jeffreys 1/x; fine for point estimate and k̂, but plan 3's SBC cannot draw from it — `default_sbc_prior()` raises rather than sample |
| P12 | Re-run dev + gate validation artifacts to convergence | PLANNED | Dev needs ~1500 iterations, gate ~10×; deliberately not spent this cycle |
| P13 | Register the `slow` pytest marker | DONE | `pytest.ini` added 2026-07-28; `@pytest.mark.slow` no longer emits `PytestUnknownMarkWarning` |
| P14 | Online scorer + ingestion v2 (trading infrastructure — ARCHITECTURE §2 P6) | DONE | Plan 2026-07-23-004 U1–U6: `src/inference/{adf_filter,online_scorer,stream_scoring}.py`, `src/data/{rtds,trade_stream}.py`, `fetch_trades_windowed`, `scripts/{stream_trades,score_stream}.py`, `pull_data --full-history`. ARCHITECTURE §8/§9/§9.4/§10/§17. CAVEAT: an unvisited `V` regime's `sigma2` stays prior/seed-dominated online, not a data-driven fit |
| P15 | SBC / coverage harness | DONE | Plan 2026-07-23-003 U1–U3: prior-predictive mode in `src/data/synthetic.py`, resumable JSONL harness `src/analysis/sbc.py` + `scripts/sbc.py`, rank-uniformity + coverage analysis and figures. **Evidence runs (200 replicates + production-size confirmation) still open — blocked on P8/P11** (user-owned); acceptance semantics in ARCHITECTURE §12.1 |

---

## Active work tracker

| Item | Status | File(s) |
|------|--------|---------|
| numba `_kalman_step_all_combos` | DONE | `src/inference/kalman.py` |
| joblib parallel K markets | DONE | `src/inference/particle_gibbs.py` (`n_jobs` field in `InferenceConfig`) |
| filter-only screening mode | DONE (ablation FAIL) | `src/analysis/prefilter.py` — AUC 0.524 at K=10/T=2000 |
| Variational EM (C1) | DONE (gate PASS) | `src/inference/variational_em.py` — ADF E-step + moment-matched M-step |
| VEM logistic M-step (`beta_S`/`beta_Z`) | DONE (opt-in; gate FAILS when on) | `src/inference/variational_em.py` — IRLS-Cauchy(0, 2.5); `estimate_betas=False` default |
| `PhiPrior` spec | DONE | `config/default_params.py` — single authoritative prior consumed by M-step, Laplace layer, PSIS |
| Laplace curvature layer | DONE (foundation unsound — P8) | `src/inference/laplace.py` — `PhiPosterior`, `laplace_from_vem` |
| Validation metric layer | DONE | `src/analysis/validation.py` — held-out predictive LL, PSIS-k̂, restart stability, ELBO convergence, `phi_centring_gradient` |
| `validate_vem` CLI + artifacts | DONE (pre-convergence — P12) | `scripts/validate_vem.py` → `results/validation/dev.json`, `results/validation/gate/gate.json` |
| Stage 2 gate | DONE | C1 VEM AUC 0.885, C0 PG AUC 0.962; C3/C2 cancelled |
| Pre-resolution subsetting | DONE | `src/data/preprocess.py`, `scripts/pull_data.py` |
| Approximate `theta_w` Gibbs | DONE | `src/inference/parameter_updates.py` |
| Benchmark script (--method support) | DONE | `scripts/benchmark.py` — supports `{pg,vem,filter,ipmcmc}` |
| --n-jobs market parallelism | DONE | `scripts/run_pg.py` — `dataclasses.replace` on preset config |
| Pareto figure + bench tooling | DONE | `scripts/pareto.py` → `results/figures/pareto.png` + CSV |
| Stepwise ADF filter (extracted E-step) | DONE | `src/inference/adf_filter.py` — output-identical to the VEM E-step (exact-equality fixture) |
| Online EM scorer (Cappé–Moulines) | DONE | `src/inference/online_scorer.py` — decayed sums, `rho` schedules `fixed`/`robbins_monro` |
| Streaming scorer library + CLI | DONE | `src/inference/stream_scoring.py` (`StreamScorer`, warm-start artifact), `scripts/score_stream.py` |
| RTDS live adapter + trade stream | DONE | `src/data/rtds.py` (wss live), `src/data/trade_stream.py` (ordering/corruption policy), `scripts/stream_trades.py` |
| Full-history windowed backfill | DONE | `fetch_trades_windowed` in `src/data/polymarket_api.py`; `pull_data --full-history` |
| SBC harness + rank/coverage analysis | DONE (evidence runs blocked — P8/P11) | `src/analysis/sbc.py`, `scripts/sbc.py`, prior-predictive mode in `src/data/synthetic.py` |

---

## Changelog

Newest first. One line per meaningful change.

| Date | Change |
|------|--------|
| 2026-07-29 | /finish CLOSED for plans 2026-07-23-003 + -004 (commits 057a73c..03cc697). Multi-agent review cycle applied 20+ validated fixes (9707d0e, b763e55, 03cc697), highlights: hand-rolled `adf_filter._logsumexp4` pinned bit-exact to scipy's algorithm (570k-vector fuzz, 0 mismatches; identity fixture enforces it) and scipy removed from the per-trade path → **~5.6× on `ADFFilter.step`, ~4.8× on the batch E-step** (closes the 2026-07-24 standing optimization target); `variational_em.update_beta_irls` promoted to PUBLIC as a cross-module contract with `online_scorer`; `OnlineScorer` keeps decayed **sums** (not averages) because the batch MAP maps consume sum-scale statistics, and `rho_t0` (default 50, ≥ 2) exists because `rho(0)=1` annihilated the seed; `delta=0` variance exclusion promoted to a cross-cutting invariant honored by `ADFFilter`/`OnlineScorer` (§6.1); `score_stream` writes a deterministic `<output>.meta.json` sidecar and pre-seeds its dedupe set so a live restart skips already-scored trades; `scripts/sbc.py` refuses cross-regime stores and prints Wilson-CI coverage verdicts; `pull_data --full-history` isolates per-market failures with an INCOMPLETE summary. `pytest.ini` now registers the `slow` marker (P13 DONE). |
| 2026-07-28 | Plans 2026-07-23-003 (SBC/coverage) and 2026-07-23-004 (ingestion v2 + streaming scorer) implemented. Plan 3 (**code DONE, evidence BLOCKED**): `params_from_prior` + prior-predictive generator mode in `src/data/synthetic.py`, resumable joblib harness `src/analysis/sbc.py` + `scripts/sbc.py` (JSONL store, schema v2 with an `(L, size, prior)` regime guard), rank-uniformity + coverage analysis and figures; the 200-replicate and production-size runs are NOT executable because `default_sbc_prior()` fails fast on the improper `tau2` IG(1e-9, 1e-9) — see P8/P11. New acceptance semantics recorded in ARCHITECTURE §12.1 (Bonferroni-corrected Wilson CI overlapping nominal 0.90, [0.85, 0.95] conclusiveness band, ~400 replicates for a conclusive `phi` row, `ks_floor = 1/(2(L+1))`). Plan 4 (**DONE**, U1–U6): `src/inference/adf_filter.py` (stepwise ADF filter extracted from the VEM E-step, output-identical), `src/inference/online_scorer.py` (Cappé–Moulines online EM), `src/inference/stream_scoring.py` (`StreamScorer` + warm-start artifact), `src/data/rtds.py` (RTDS websocket, schema VERIFIED live 2026-07-25), `src/data/trade_stream.py` (ordering/corruption policy), `fetch_trades_windowed` + `pull_data --full-history`, `scripts/stream_trades.py`, `scripts/score_stream.py`. Data API probe findings (2026-07-25) recorded in ARCHITECTURE §9.4: server offset ceiling 10000, `start`/`end` in inclusive UNIX **seconds**, 26 other candidate param names silently ignored, `transactionHash` unique across taker-side rows. |
| 2026-07-25 | /finish CLOSED for plans 2026-07-23-001 + -002; full suite 325 passed, 1 xfailed (intentional multi-seed-stability pin on the beta-estimation path). Honest status of plan 1's R6: beta estimation is opt-in (`estimate_betas=False`) because the ADF E-step does not identify Z — with `--estimate-betas` ON the synthetic gate **FAILS** (pooled AUC 0.547 vs 0.9435 off), so the recorded gate PASS validates the beta-fixed path only. New finding: **restart instability** — pooled AUC across 10 jittered restarts (init jitter log-sd 0.1, fixed data) spans 0.376–0.915 (dev) and 0.388–0.877 (gate), gate top-K Jaccard 0.171; this is INITIALIZATION sensitivity, not data-seed sensitivity (the deterministic unjittered warm start used by `scripts/benchmark.py` gives 0.885/0.899/0.893/0.915 across data seeds). Headline gate AUC 0.885 must be reported as single-initialization, deterministic-warm-start, at-the-iteration-cap — stable across data seeds, not across initializations. Also: `sigma2_1 = max(sigma2_1, sigma2_0)` binds exactly at every fitted point (P10); `m_Z` refresh ordering makes blockwise monotonicity only conditional (docstring corrected, reorder deferred); `slow` pytest marker unregistered (P13). |
| 2026-07-25 | Plan 2026-07-23-002 (VEM validation suite) implemented: `PhiPrior` in `config/default_params.py` as the single prior spec (M-step + Laplace + PSIS), `src/inference/laplace.py` (`PhiPosterior`, `laplace_from_vem`), `src/analysis/validation.py` (held-out one-step predictive LL, PSIS-k̂, restart-stability + ELBO-convergence blocks, `phi_centring_gradient`), `src/analysis/plots.py` additions, `scripts/validate_vem.py`, artifacts under `results/validation/`. **k̂ > 0.7 stop condition FIRED**: k̂ = 5.82 (dev), 24.0 (gate). The plan's prescribed remedy (enrich the variational family) is REFUTED — the proposal is mis-centred and mis-scaled, so a richer family at the same centre gives the same k̂: `PhiPosterior` uses expected-complete-data (ECM) curvature rather than observed information (Louis 1982), and the VEM fixed point is not stationary for the PSIS target (at dev scale run to convergence: target gradient ≈ 10 Laplace-sd on `log sigma2_0`; observed information 2.24 vs Laplace 252.3, 113× over-precise; observed information on `tau2_1` is NEGATIVE, −3.96, i.e. a local minimum along that axis). ~75% of Laplace draws violate the estimator's own order constraints. Committed artifacts are PRE-CONVERGENCE (all 10 restarts hit the 50-iteration cap; final relative ELBO change 5.35e-4 dev / 1.31e-3 gate vs tol 1e-4, still climbing) and now carry a `convergence_status` block; best-restart selection is not meaningful there (terminal-ELBO spread 1.302 dev / 64.6 gate < one iteration's gain 1.431 / 79.16). See P8–P12. |
| 2026-07-24 | Runtime "regression" resolved — no code fault. Interleaved same-machine A/B (baseline 8c445f9 vs HEAD, THREADS=1, identical inputs): base 95.6 s vs head 98.7 s mean (+3.3%, within run noise); the historical 68.8 s figure is environment drift (same commit times at 95.6 s today) — re-baseline before any cross-date VEM timing comparison. `theta_w` Newton is 0.14 s/fit (warm-started, ~2–3 damped steps; entire M-step 0.1% of runtime). Standing optimization target found (pre-existing, not from this series): per-trade `scipy.special.logsumexp` over length-4 arrays, 1M calls ≈ 80% of profiled self-time via scipy's array-api dispatch — candidate for a hand-rolled 4-element LSE or folding into the numba kalman kernel; scope as its own unit. |
| 2026-07-24 | Plan 2026-07-23-001 (VEM logistic M-step) implemented: covariate centering per Gelman 2008 (2de937c), IRLS-Cauchy(0, 2.5) M-step for `beta_S`/`beta_Z` + offset-adjusted per-wallet logit-normal `theta_w` Newton update (4aaadc5). M-step verified in isolation with oracle q(Z): planted `beta_S`=1.0 → 0.98–1.11, `beta_Z`=1.5 → 1.44–1.66 (3 seeds); separation-proof; absorption-bias mechanism (R8) confirmed. **Gate PASS** under `estimate_betas=False` default (pooled AUC 0.885, 100% recall@4, seeds 42/43/44). Beta estimation is opt-in (f4bcdc4): ADF E-step cannot identify Z on the synthetic generator (q(Z) near-flat), so default-on estimation fit a spurious `beta_S` ≈ −0.40 (AUC 0.68) — pinned by an xfail multi-seed test; P3-relevant. Open: E-step Z-identifiability (candidate next plan item); VEM runtime ~98 s vs 68.8 s baseline (+42%, both modes — under profiling). |
| 2026-07-23 | PG/iPMCMC retired from all new computation: frozen as cited historical baseline (C0 PG AUC 0.962, 3117.5 s). Opened 2026-07-23 five-plan VEM logistic M-step series: plan 1 (WIP) weighted-logistic IRLS M-step with Cauchy(0, 2.5) prior for `beta_S`/`beta_Z`; plan 2 Laplace uncertainty; plan 3 SBC/PSIS validation + `theta_w` coverage; plan 4 online scorer; plan 5 Kalshi anonymous-mode variant. Validation path is SBC/PSIS (Talts et al. 2018; Yao et al. 2018), never PG comparison. Deferred PG bench stages / C4-on-PG marked CANCELLED. PG code and tests untouched as historical baseline. |
| 2026-07-05 | /finish cycle CLOSED: full fast suite green (240 passed pre-fix, 22/22 script tests post-fix). Code-review fixes (commit 4ff274f): scripts/benchmark.py --method ipmcmc resets cfg.n_jobs=1 when warning flag is inert; JSON config block records effective value; warns when --M/--P passed to non-ipmcmc methods; prints M/P in ipmcmc report header; collapses duplicate pg/ipmcmc timing branches into _MCMC_RUNNERS dispatch map. tests/test_scripts.py adds test_artifacts_from_mcmc_chain_flattens_ipmcmc_theta pinning iPMCMC (n_iter, P, n_wallets) -> (n_iter*P, n_wallets) post-burn-in theta_w pooling. |
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
| Canonical inference run | **CLOSED (2026-07-25):** VEM (`scripts/benchmark.py --method vem`, `scripts/validate_vem.py`) is the canonical engine; the half-prod PG/iPMCMC "default refinement run" is historical/frozen. ARCHITECTURE §2/§3/§6/§10 reconciled. |
| `beta_S`/`beta_Z` estimation | **CLOSED (2026-07-25):** opt-in only (`estimate_betas=False` default, `--estimate-betas` to enable). The ADF E-step does not identify Z; gate FAILS with it on (AUC 0.547 vs 0.9435). Plan 1's gate PASS covers the beta-fixed path only. |
| Prior specification | **CLOSED (2026-07-25):** `PhiPrior` in `config/default_params.py` is the single authoritative spec, consumed by the VEM M-step, the Laplace layer and PSIS. Its `tau2` IG(1e-9, 1e-9) is numerically improper Jeffreys 1/x — usable for point estimates and k̂, not for SBC prior draws (P11). |
| PSIS target semantics | **CLARIFIED (2026-07-25):** the PSIS target is CONDITIONAL on `theta_w_hat` (held fixed across draws), not a parameter marginal. Earlier docs mis-stated this. |
| k̂ > 0.7 escalation remedy | **CLOSED (2026-07-25):** enriching the variational family is REJECTED as the remedy — the proposal is mis-centred/mis-scaled (ECM curvature ≠ observed information; VEM fixed point not stationary for the target), so a richer family at the same centre gives the same k̂. Rebuild the foundation instead → P8; plan 3's SBC **evidence runs** are blocked until then (the harness itself landed 2026-07-28, P15). |
| PG/iPMCMC retirement | **CLOSED (2026-07-23):** frozen as cited historical baseline (AUC 0.962, 3117.5 s); all new computation on the VEM fast path; validation via SBC/PSIS (Talts et al. 2018; Yao et al. 2018), never PG comparison. |
| Stage 2 gate — C1 VEM promotion | **CLOSED (2026-07-04):** C1 VEM gate PASS; C3 twisted-CSMC NOT implemented; C2 rSLDS moot. |
| Kendall τ acceptance criterion | **INVALIDATED:** PG-vs-PG control (N=250, 1500-iter, different seed) → τ=0.787 (below 0.85 threshold). Replacement: 100% insider recall@K (top_cutoff rule) across ≥3 synthetic seeds + pooled AUC ≥0.85 (per-market floors); weighted τ / top-K overlap reported descriptively. |
| Filter-only screening ablation | **GATE FAIL:** Pooled AUC 0.524 at K=10/T=2000. Kept as negative-result ablation row. |
| Bounded/absorbing price model | **SKIPPED PERMANENTLY:** P1 filter sufficient per Stage 0. |
| Model | Baseline spec; refinements OK if synthetic tests pass |
| Inference default | Half-prod, not full prod |
| Data source | Polymarket Data API is the sole historical/backfill source (no Goldsky/CLOB); RTDS is its live counterpart (decision #9, spirit unchanged — see ARCHITECTURE §9) |
| CSMC reference index | 0 (code authoritative) |
| Doc hierarchy | `ARCHITECTURE.md` + this file for agents |
| Entrypoints | `scripts/` CLIs only |
