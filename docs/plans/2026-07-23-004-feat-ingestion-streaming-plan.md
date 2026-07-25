---
title: Ingestion v2 and Streaming Scorer - Plan
type: feat
date: 2026-07-23
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Ingestion v2 and Streaming Scorer - Plan

**Sequence:** Plan 4 of 5. Depends on plan 1 (centered covariates + beta M-step; centering constants on `VEMOutput`). Independent of plans 2–3. Plan 5 depends on this plan's replay mode and adapters.

---

## Goal Capsule

Build the two-adapter ingestion layer (timestamp-paginated Data-API backfill; RTDS `activity/trades` live WebSocket) emitting one normalized trade schema, and the streaming scorer: a stepwise ADF filter object with Cappé–Moulines online sufficient-statistic updates, an explicit forgetting factor, and Beta-prior shrinkage as the cold-start policy for unseen wallets.

**Authority hierarchy:** CLAUDE.md hard rules > agent_reference docs > this plan. Data API remains the sole trade source for backfill (resolved decision #9 — RTDS is the live counterpart, not CLOB/Goldsky). `scripts/` stays the only entrypoint. Hot-path rule: the batch VEM path must not slow down (68.8 s gate-scale baseline) and the ADF refactor must be output-identical.

**Stop conditions:** surface if (a) empirical probing shows the Data API cannot deliver history beyond the current ~3000-trade tail by any documented parameter (then full-history backfill is deferred, the flag ships disabled, and the rest proceeds), (b) RTDS message schema lacks `proxyWallet` in practice, or (c) the ADF-refactor regression test cannot achieve output-identity.

---

## Product Contract

### Summary

One normalized schema `{ts, market_id, outcome_idx, p, S, wallet, side, tx_hash}` with two producers: (a) `fetch_trades` upgraded to timestamp-windowed pagination with the per-market trade cap kept as the default and full history behind `--full-history`; (b) a new RTDS WebSocket client streaming live trades to an append-only sink. One consumer beyond batch: `OnlineScorer` — the ADF forward step extracted into a filter object, wrapped with online parameter updates (forgetting factor) and per-wallet Beta count updates — driven by `scripts/score_stream.py` in live and replay modes. Replay mode is the no-lookahead evaluation surface plan 5 builds on.

### Problem Frame

The trading path (ARCHITECTURE §17) needs per-trade `P(Z=1 | D≤i)` in O(1) per trade; batch VEM re-fits are neither online nor cheap. The current backfill is truncated at ~3000 tail trades per market (`DATA_API_MAX_OFFSET`, `polymarket_api.py:353`), which blocks full-history case studies (plan 5's Van Dyke markets). The report documents official offset limits raised to 10,000 and a start-timestamp parameter — both to be verified empirically, not hard-coded.

### Requirements

Backfill:
- R1. `fetch_trades` supports timestamp-windowed pagination (walk backward/forward by trade timestamp with `offset` reset per window), deduplicating on `transaction_hash` across windows; behavior with default arguments is unchanged (existing callers and artifacts unaffected).
- R2. An empirical probe (small script or documented manual run) verifies the current offset cap and the timestamp parameter's actual name/semantics before the pagination logic is finalized; findings recorded in ARCHITECTURE §9.4.
- R3. `pull_data.py` gains `--full-history` (default off). Default runs keep today's tail-capped behavior byte-compatible; full-history runs still respect `--tail-trades` if explicitly passed (cap applied after retrieval).
- R4. Rate-limit discipline: `/trades` calls stay under 200/10 s (worst case), reusing the existing 429/5xx exponential backoff (`polymarket_api.py:161-194`).

Live adapter:
- R5. `src/data/rtds.py`: WebSocket client for the RTDS `activity`/`trades` topic (public, no auth), normalizing messages to the same `RawTrade` shape (wallet from `proxyWallet`), with auto-reconnect + exponential backoff and a heartbeat/staleness log line.
- R6. `scripts/stream_trades.py`: CLI running the client and appending normalized trades to an on-disk sink (JSONL default; `--parquet-every N` optional compaction), with `--markets` filtering by condition id and clean SIGINT shutdown (no truncated records).
- R7. New dependency (`websocket-client` or `websockets`) added to requirements with a version pin; import guarded so the rest of `src/data` works without it installed.

Streaming scorer:
- R8. `ADFFilter`: stepwise filter object exposing init-from-(`params`, `theta_w`, centering constants) and `step(trade) -> (P(Z), P(V), X_mean)` carrying exactly the per-trade state the batch E-step carries (`prev_q_V`, `prev_E_Z`, Kalman moments); the batch E-step is re-expressed on top of it and stays output-identical and within noise of current runtime.
- R9. `OnlineScorer`: wraps `ADFFilter` with Cappé–Moulines stochastic sufficient-statistic updates for `(sigma2_v, tau2_z, q_01, q_10)` under a forgetting factor `lambda` (explicit hyperparameter; `lambda = 1` reproduces frozen-parameter filtering), per-wallet Beta count updates for `theta_w` (cold start = Beta(a, b) prior mean — the shrinkage policy), and a periodic decayed-IRLS refresh of betas every `n_refresh` trades reusing plan 1's IRLS on decayed statistics.
- R10. `scripts/score_stream.py`: live mode (sink or socket → scores JSONL) and replay mode (historical Parquet/JSONL → scores JSONL), one code path for scoring; replay guarantees no lookahead (trades strictly in timestamp order, state never sees the future).

### Scope Boundaries

- **Out of scope:** trading/signal logic on top of scores (plan 5); Kalshi (plan 5); CLOB order-book channels; on-chain mint/burn double-counting correction (recorded as a follow-up; tail-capped politics-market trades are the current paper surface); any daemon/service management (systemd etc. — user runs CLIs).
- **Deferred to follow-up work:** share mint/burn volume double-counting normalization (arXiv:2603.03136) if full-history volume analysis becomes a paper claim; Parquet-native sink as default.

---

## Planning Contract

### Key Technical Decisions

- KTD1 — **Probe before pagination logic hardens (R2).** The gap between documented (10,000 offset, start-timestamp param) and previously observed (400 above 3000) behavior is a known API-drift risk; the pagination unit starts with the probe and encodes what is actually true, with the offset cap kept as a fallback constant.
- KTD2 — **Windowed pagination keys on timestamps, dedupes on `transaction_hash`.** Same-second bursts make timestamp cursors non-unique; overlap windows by one second and dedupe rather than trusting cursor uniqueness.
- KTD3 — **`ADFFilter` extraction is a pure refactor with an output-identity gate AND a measured runtime gate.** Fixture: pinned `Z_prob/V_prob/X_mean` from the current `_vem_e_step` on the standard synthetic config; the refactored batch E-step must match to tight tolerance (target: exact equality of floating-point operations order where feasible; else atol ≤ 1e-12). Runtime (architecture-critic MAJOR — the loop at `variational_em.py:92-147` is the hot path, T×50 EM iterations, and per-trade method dispatch + state packing adds Python overhead): measure the batch E-step wall-clock BEFORE refactoring on the gate-scale config, gate the refactor at ≤5% measured regression, and if exceeded fall back to a shared free-function step (module-level function over plain arrays/scalars that both the batch loop and a thin `ADFFilter` wrapper call) rather than an object-method-per-trade design in the batch path.
- KTD4 — **Online updates follow Cappé–Moulines (2009) form:** running sufficient statistics `s ← (1−ρ_t)s + ρ_t s(trade)` with `ρ_t` either Robbins–Monro (`t^-α`) or fixed (forgetting factor `1−λ`); M-step maps unchanged. Betas refresh by periodic IRLS on decayed statistics rather than per-trade SGD — reuses plan 1 code and keeps the separation-proof prior.
- KTD5 — **Cold start = prior shrinkage, no special case:** an unseen wallet's Beta counts are simply (a, b); the scorer needs no wallet registry beyond a growable dict alongside `WalletIndex`.
- KTD6 — **Sink is append-only JSONL first.** Crash-safe, greppable, streamable; Parquet compaction is an optional batch step, not the write path.

### High-Level Technical Design

```
backfill:  Data API /trades ──(timestamp windows, dedupe)──▶ RawTrade ─▶ preprocess ─▶ ProcessedMarket
live:      RTDS ws activity/trades ──(normalize)──▶ RawTrade ─▶ JSONL sink ─┐
                                                                            ├─▶ score_stream (live)
replay:    historical JSONL/Parquet ────────────────────────────────────────┘        │
                                                                                     ▼
           OnlineScorer:  ADFFilter.step ─▶ P(Z),P(V),X ─▶ scores JSONL
                          └ suff-stats (forgetting λ) ─ periodic M-step + IRLS refresh
                          └ theta_w Beta counts (cold start = prior)
```

Directional guidance, not implementation specification.

---

## Implementation Units

### U1. Offset/timestamp probe + windowed backfill in `polymarket_api.py`

**Goal:** Full-history capability with verified API behavior; defaults byte-compatible.
**Requirements:** R1, R2, R4.
**Dependencies:** none.
**Files:** `src/data/polymarket_api.py`, `tests/test_polymarket_api.py`.
**Approach:** Probe first (KTD1) against a known high-volume resolved market; record findings. Then `fetch_trades_windowed(condition_id, *, start_ts, end_ts, ...)` built on the existing `_get_json` backoff; `fetch_trades` keeps its exact current signature/behavior; windowing per KTD2. Respect 200/10 s with the existing `sleep_between`.
**Test scenarios (mocked HTTP, following existing `test_polymarket_api.py` patterns):**
- Window walk: mock server with 7000 trades and a 3000-offset cap → windowed fetch returns all 7000 exactly once (dedupe on tx hash).
- Same-second burst at a window boundary → no loss, no duplicates.
- Default path: `fetch_trades` calls with default args produce identical requests/results to a pinned pre-change recording.
- 429 mid-window → backoff and resume, no duplicate rows.
**Verification:** tests green; probe findings written into ARCHITECTURE §9.4 (via /finish docs-updater note) and echoed in the unit's commit message.

### U2. `--full-history` in `pull_data.py`

**Goal:** CLI surface for deep backfill without changing defaults.
**Requirements:** R3.
**Dependencies:** U1.
**Files:** `scripts/pull_data.py`, `tests/test_scripts.py`.
**Approach:** Flag routes to `fetch_trades_windowed` with `start_ts` epoch 1; post-retrieval `--tail-trades` still honored; log the retrieved-vs-kept counts.
**Test scenarios:** flag off → identical behavior (pinned call pattern); flag on (mocked) → windowed function invoked, tail cap applied after.
**Verification:** smoke test green.

### U3. RTDS live adapter + `stream_trades.py`

**Goal:** Live normalized trade feed to disk.
**Requirements:** R5, R6, R7.
**Dependencies:** none (file-disjoint from U1/U2 — parallelizable).
**Files:** `src/data/rtds.py` (new), `scripts/stream_trades.py` (new), `requirements.txt`, `tests/test_rtds.py` (new), `tests/test_scripts.py`.
**Approach:** Empirical schema check first (architecture-critic DOC GAP): `RawTrade.from_dict` (`polymarket_api.py:123-136`) parses REST field names — the RTDS message schema is assumed to mirror it (`proxyWallet` etc.) but is unverified; capture and check a handful of real RTDS messages at the start of this unit, pin one as the normalization fixture, and only then finalize the mapping. Thin client class with injected socket factory (tests inject a fake); normalize via a shared message→RawTrade function reusing (not duplicating) the existing fallback logic; reconnect with capped exponential backoff + jitter; staleness warning when no message for N s. CLI writes JSONL lines atomically (write+flush per record), SIGINT closes cleanly.
**Test scenarios:**
- Normalization: sample RTDS trade payload (fixture from docs) → RawTrade with wallet=proxyWallet, ts seconds, price/size floats.
- Malformed message → logged and skipped, stream continues.
- Reconnect: fake socket raises after 3 messages → client reconnects, no message loss for delivered ones, backoff sequence correct.
- Sink integrity: SIGINT mid-stream → last line is valid JSON (no partial record).
- Market filter: `--markets` keeps only matching condition ids.
**Verification:** unit tests green without network; a manual smoke against the real endpoint documented in the commit message (message received + normalized), not in CI.

### U4. `ADFFilter` extraction (pure refactor)

**Goal:** Stepwise O(1) filter; batch E-step re-expressed on top, output-identical.
**Requirements:** R8.
**Dependencies:** plan 1 (centered covariates live in the E-step).
**Files:** `src/inference/variational_em.py` (or new `src/inference/adf_filter.py` with VEM importing it — implementer's call, favor the new module per module-size), `tests/test_variational_em.py`, `tests/test_adf_filter.py` (new).
**Approach:** Move the loop body of `_vem_e_step` (`variational_em.py:92-147`) into `ADFFilter.step`; state = exactly the carried variables (`prev_q_V`, `prev_E_Z`, Kalman `mu`, `sigma2`); `_vem_e_step` becomes init + loop over `step` + the same accumulators. Fixture-gated per KTD3.
**Execution note:** BEFORE refactoring, generate the output-identity fixture AND record the measured batch E-step baseline wall-clock at gate scale; the ≤5% runtime gate compares against that measurement, not the historical 68.8 s. Fall back to the KTD3 free-function design if the object-per-trade shape misses the gate.
**Test scenarios:**
- Output identity: pinned fixture (standard synthetic config, fixed seed) vs refactored batch E-step — `Z_prob/V_prob/X_mean/log-marginal` equal within atol 1e-12 (or exactly).
- Stepwise equivalence: feeding trades one-by-one through `ADFFilter` equals the batch arrays.
- Runtime: batch E-step wall-clock within ~5% on the gate-scale synthetic config (assert generously in a slow test; benchmark evidence in the commit).
- State isolation: two interleaved filter instances don't share state.
**Verification:** fast + slow tests green; VEM gate numbers from plan 1 reproduce exactly.

### U5. `OnlineScorer` with forgetting factor

**Goal:** Online parameter + `theta_w` adaptation around the filter.
**Requirements:** R9.
**Dependencies:** U4; plan 1 U3 (IRLS reuse).
**Files:** `src/inference/online_scorer.py` (new), `tests/test_online_scorer.py` (new), `config/default_params.py` (scorer config dataclass: `forgetting`, `n_refresh`, `rho_schedule`).
**Approach:** Per KTD4/KTD5. Sufficient statistics mirror the batch M-step's (reuse its accumulator forms); `lambda = 1, n_refresh = ∞` degenerates to frozen-parameter filtering (key regression anchor). Delta=0 steps excluded from sigma² stats (mirror the real-data fix, ARCHITECTURE §6.1).
**Test scenarios:**
- Frozen-limit regression: `lambda=1`, no refresh → per-trade scores identical to bare `ADFFilter`.
- Adaptation: synthetic stream whose true `sigma2_1` doubles mid-stream → online `sigma2_1` estimate moves toward the new value with `lambda<1`, stays near the old with `lambda=1`.
- Cold start: unseen wallet scores with prior-mean `theta_w`; after that wallet's high-`q(Z)` trades, its `theta_w` rises.
- Beta refresh: decayed-IRLS refresh changes betas smoothly (no jump discontinuity > sanity bound) and never diverges under the separation stress stream.
- delta=0 trades don't corrupt variance stats (NaN guard).
**Verification:** tests green; no change to batch VEM behavior.

### U6. `scripts/score_stream.py` live + replay

**Goal:** The end-to-end CLI; replay is plan 5's evaluation substrate.
**Requirements:** R10.
**Dependencies:** U3, U5.
**Files:** `scripts/score_stream.py` (new), `tests/test_scripts.py`.
**Approach:** One scoring loop consuming an iterator of normalized trades; `--replay <path>` sorts strictly by (timestamp, tx_hash) and iterates; `--live` tails the RTDS sink (or embeds the client); `--warm-start <chain.pkl or vem.json>` initializes params/theta_w/centering from a fitted artifact (via `scripts/_runner.py` loaders); output JSONL: `{ts, tx_hash, market, wallet, p_z, p_v, x_mean}`.
**Test scenarios:**
- Replay determinism: same input file + warm start → byte-identical output.
- No-lookahead: score of trade i unchanged when trades > i are deleted from the file (prefix-invariance test on a small fixture).
- Warm-start loading from a VEM artifact restores centering constants (scores match an in-process run).
- Out-of-order input → either sorted (replay) or rejected with clear error (live), per mode.
**Verification:** smoke tests green; a documented replay run over one pulled market committed under `results/streaming/`.

---

## Verification Contract

- `python -m pytest -q -m "not slow"` green; slow runtime tests in full suite.
- Output-identity fixture test (U4) is the hard gate for the refactor; VEM gate re-run (`benchmark.py --method vem --gate`) reproduces plan 1 numbers.
- No-speed-regression: batch VEM within ~5% of the post-plan-1 baseline; document numbers in commit.
- Live smoke (manual, documented): `stream_trades.py` captures ≥ 1 real RTDS trade with a wallet.
- PG freeze audit: no PG/iPMCMC computation added.

## Definition of Done

- U1–U6 landed; replay pipeline runs end-to-end on one historical market.
- API probe findings recorded (ARCHITECTURE §9.4 note for docs-updater).
- requirements.txt pin added; imports guarded.
- Dead-end code removed (esp. any pre-refactor E-step duplicate).

---

## Documentation Notes (for /finish docs-updater)

- ARCHITECTURE.md: §8 module map (+`rtds.py`, `adf_filter.py`/filter location, `online_scorer.py`, 3 new scripts); §9 data pipeline (windowed pagination, probe findings, RTDS adapter + verified message schema, sink format); §9.4 offset-cap row updated with probe results; §10 CLI (new flags/commands); §17 trading section updated from "future" to the shipped scorer shape.
- ARCHITECTURE.md §6.1: promote the delta=0 variance fix to a documented cross-cutting invariant that `ADFFilter`/`OnlineScorer` must also honor (architecture-critic DOC GAP).
- STATUS.md: roadmap P6 → WIP/DONE rows; changelog.
- Paper (final /paper cycle): streaming scorer = online recursive EM (Cappé & Moulines 2009) with forgetting factor; cite Campbell et al. 2021 as precedent; cold-start = hierarchical shrinkage.
