---
title: Kalshi Anonymous Variant and Signal Evaluation - Plan
type: feat
date: 2026-07-23
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Kalshi Anonymous Variant and Signal Evaluation - Plan

**Sequence:** Plan 5 of 5. Depends on plan 1 (IRLS M-step — the anonymous intercept rides on it) and plan 4 (normalized schema, full-history backfill, replay scorer). Land after plan 3: both edit `src/data/synthetic.py` (file-contention; plan 3's prior-predictive mode first), and the paper's validation claims should precede its evaluation claims.

---

## Goal Capsule

Ship (1) the anonymous-venue model variant — `theta_w` dropped, market-level intercept `alpha`, native taker-side sign, VPIN prefilter with volume controls — implemented in this repo with a clean importable scorer interface for the separate Kalshi trading system, with a Kalshi `GetTrades` adapter; and (2) the detection-focused signal evaluation: a no-lookahead event study (does elevated `P(Z)` precede the terminal move?) and the Van Dyke / January-2026 Maduro-cluster labeled case study. A costed deflated-Sharpe backtest is an explicit stretch unit.

**Authority hierarchy:** CLAUDE.md hard rules > agent_reference docs > this plan. PG freeze is hard. Evaluation discipline: every score at trade j uses only data ≤ j (replay mode guarantees this); no Kendall-tau acceptance criteria; VPIN is a gating signal only (filter-only detection already GATE FAIL, AUC 0.524).

**Stop conditions:** surface if (a) the Kalshi public trade schema turns out to carry any account key (would re-open the theta_w-lite design — user decision), (b) the Maduro-cluster markets cannot be identified/pulled with full history, or (c) the event study shows no signal — that is a reportable negative result, but how to frame it in the paper is the user's call.

---

## Product Contract

### Summary

Kalshi's public trade feed has no per-account identifier, so the wallet-anchored model cannot port as-is. The honest port keeps the trade-level machinery (Y = logit(p), regime V, state X, insider indicator Z) and replaces the wallet term: `logit(pi_Z) = alpha + beta_S·log(S/S̄) + beta_Z·1{Z_prev}`, with `alpha` an estimated market-level intercept — allowed here precisely because there is no `theta_w` to carry the level. The variant lives in this repo (shared model core, paper contribution); the trading system imports the scorer. Evaluation: an ILS-style event study over replayed Polymarket markets and the first externally-labeled ground truth (U.S. v. Van Dyke, ~13 bets, ~$33k, Venezuela/Maduro markets, Dec 27 2025 – Jan 2).

### Problem Frame

The paper's novelty claim ("streaming, per-trade, generative, uncertainty-quantified layer") needs (a) evidence the score is temporally informative, not just discriminative in-sample, and (b) positioning that survives the anonymous-venue objection. The Van Dyke case is the only public labeled insider episode; the anonymous variant is the bridge to the regulated-venue (Kalshi) deployment.

### Requirements

Anonymous variant:
- R1. Model mode switch: anonymous mode replaces the `logit(theta_w[w])` offset with a per-market intercept `alpha` estimated in the IRLS M-step (3-column design: intercept, centered size, centered `E[Z_prev]`; Cauchy(0, 10) on the intercept per Gelman's intercept-scale recommendation). Wallet mode remains the default and is bit-unchanged.
- R2. `src/data/synthetic.py` generates anonymous-mode data (known `alpha`, no wallets) for validation; VEM, `ADFFilter`, and `OnlineScorer` all accept anonymous mode.
- R3. The scorer interface is importable without Polymarket-specific code: init from params + mode, `step(trade)` where `trade` needs only `{ts, p, S, side}` (+ wallet when in wallet mode) — the seam the external Kalshi trading system consumes.

Kalshi adapter:
- R4. `src/data/kalshi_api.py`: public `GetTrades` REST client (no auth) with pagination and the existing backoff pattern, normalizing to the plan-4 schema with `wallet = None`, `side` = native taker side; `scripts/pull_kalshi.py` CLI mirrors `pull_data.py` conventions.
- R5. VPIN prefilter hardening in `src/analysis/prefilter.py`: when native taker side exists, use it for classification and report the existing price-change proxy as a robustness comparison; VPIN usage documented as gating-with-volume-controls (volume included as a covariate/control in any analysis using it).

Signal evaluation:
- R6. Event study (`scripts/event_study.py`): for each replayed market, ONE pre-registered primary statistic — **mean** `P(Z)` elevation over trades in [t_close − W, t_close − w] — tested against a **within-market time-shifted-window permutation null** (the primary scheme: conditions on each market's own `P(Z)` baseline and avoids the length-dependent false-positive inflation a cross-market shuffle with extreme statistics invites — methods-critic MAJOR). Max-elevation and cross-market-shuffle variants are reported only as labeled robustness checks, never as independent confirmation. Strictly replay-mode inputs (no lookahead by construction).
- R7. Van Dyke case study: identify the Venezuela/Maduro Polymarket markets active Dec 2025 – Jan 2026, pull full history (plan 4 `--full-history`), run the wallet-anchored scorer in replay, and report where the cluster's activity ranks (per-trade scores and wallet ranking), alongside honest caveats (single case, post-hoc identification). Artifacts under `results/case_studies/van_dyke/`.
- R8 (stretch). Deflated-Sharpe costed backtest: threshold-follow strategy on `P(Z)` with spread + taker-fee model (Kalshi fee ≈ 0.07·p·(1−p)), purged/embargoed walk-forward splits, deflated Sharpe with disclosed trial count. Explicitly optional; lands only if U1–U5 finish within budget.

### Scope Boundaries

- **Out of scope:** any claim of a validated alpha strategy (detection PoC framing is mandatory); Kalshi private/authenticated channels; PIN/adjusted-PIN estimation (cited alternatives, not built); White/Hansen bootstrap tests (deferred until there is a strategy worth testing); wash-trade network detection (cited as confound, Sirolly et al.).
- **Outside this product's identity:** the trading system's execution/order logic — lives in the separate Kalshi repo, which imports this scorer.
- **Deferred to follow-up work:** theta_w-lite re-enablement if Kalshi ever exposes persistent account tokens; ILS computation proper (Nechepurenko) as a comparison metric.

---

## Planning Contract

### Key Technical Decisions

- KTD1 — **Mode is a model-level flag, not a fork.** One predictor builder used by VEM/ADFFilter/OnlineScorer/synthetic switches on mode; wallet mode's code path and outputs stay bit-identical (fixture-gated). No duplicated inference modules.
- KTD2 — **`alpha` is per-market, estimated, and its prior scale is an open empirical knob, not settled at Cauchy(0, 10).** In anonymous mode the intercept is identified (no random effect competes), but a per-market `alpha` fit from one market's data under a wide prior is far more weakly shrunk than `theta_w` ever was under Beta(1,19) — an incidental-parameters-style bias risk for low-trade-count markets (methods-critic MAJOR). U1 therefore requires an `alpha`-recovery-vs-T test (T ∈ {200, 1000, 3000}); if small-T recovery is poor under Cauchy(0, 10), tighten to Cauchy(0, 2.5) centered at the pooled base-rate logit and record the choice. Hierarchical pooling over markets is deliberately not built (deferred — keep the variant simple and honest).
- KTD3 — **The Kalshi adapter emits the plan-4 schema with `wallet = None`;** downstream code treats wallet-nullability as the mode signal at data-load time, with an explicit override flag on CLIs.
- KTD4 — **Event-study statistic is committed in this plan, calibrated on synthetic, then run once on real data:** primary = mean elevation + within-market time-shift permutation (R6); window W fixed by U4's synthetic calibration before any real-data run; real-data runs report exactly that statistic. Max/cross-market variants and any post-hoc statistic are labeled robustness/exploratory.
- KTD5 — **Van Dyke market identification is manual + documented:** slugs/condition-ids listed in a checked-in manifest (`results/case_studies/van_dyke/markets.json`) with sources (DOJ indictment dates, market questions), so the case study is reproducible from the manifest.

---

## Implementation Units

### U1. Anonymous mode in model core (synthetic + VEM + filter + scorer)

**Goal:** One flag, four consumers; wallet mode bit-unchanged.
**Requirements:** R1, R2, R3.
**Dependencies:** plan 1 U3 (IRLS), plan 4 U4/U5 (filter/scorer).
**Files:** `src/inference/variational_em.py`, `src/inference/online_scorer.py` (+ filter module), `src/data/synthetic.py`, `config/default_params.py` (mode + alpha fields), `tests/test_variational_em.py`, `tests/test_synthetic.py`, `tests/test_online_scorer.py`.
**Approach:** Per KTD1/KTD2: predictor builder with mode enum; IRLS design gains an intercept column only in anonymous mode; synthetic generator draws anonymous data with known alpha.
**Test scenarios:**
- Wallet-mode bit-identity: standard fixture outputs unchanged (hard gate).
- Anonymous recovery: synthetic anonymous data (alpha = logit(0.05), beta_S = 1.0) → recovered alpha/betas correct sign, loose tolerance, 3 seeds.
- Alpha-vs-T shrinkage sweep (KTD2): recovery of alpha at T ∈ {200, 1000, 3000} → bias shrinks with T; small-T bias under the chosen prior recorded, prior tightened if recovery is poor.
- Anonymous AUC: Z discrimination beats size-only ranking baseline on synthetic anonymous data (sanity that regime/state machinery adds value without theta_w).
- Scorer seam: `step` accepts a trade dict without wallet in anonymous mode; raises a clear error if wallet mode gets wallet-less trades.
**Verification:** fast suite green; anonymous synthetic gate documented in test asserts (AUC floor chosen at implementation time from first runs, then pinned).

### U2. Kalshi `GetTrades` adapter + `pull_kalshi.py`

**Goal:** Kalshi public trades → normalized schema.
**Requirements:** R4.
**Dependencies:** plan 4 U1 (schema/windowing conventions).
**Files:** `src/data/kalshi_api.py` (new), `scripts/pull_kalshi.py` (new), `tests/test_kalshi_api.py` (new), `tests/test_scripts.py`.
**Approach:** REST client mirroring `polymarket_api.py` structure (session, backoff, cursor pagination per Kalshi docs); normalize (price in cents → (0,1) probability, ts, size, taker side, `wallet=None`, ticker as market id); CLI mirrors `pull_data.py` (market tickers, output-dir, pre-resolution filter reuse).
**Test scenarios (mocked HTTP):**
- Pagination via cursor until exhausted; dedupe on trade id.
- Normalization: cents → probability correct at boundaries (1¢, 99¢); taker side mapped to sign convention.
- No-identity invariant: normalized output has wallet None for every row (regression against silently inventing identity).
- Backoff on 429.
**Verification:** tests green; one real pull documented (a resolved political market) with row counts in the commit message.

### U3. VPIN prefilter hardening

**Goal:** Referee-proof VPIN usage: native-side classification + volume controls + robustness comparison.
**Requirements:** R5.
**Dependencies:** U2 (side field availability); file-disjoint from U1 — parallelizable.
**Files:** `src/analysis/prefilter.py`, `tests/test_prefilter.py` (extend existing prefilter coverage).
**Approach:** `vpin_scores` (`prefilter.py:103`) gains `side` input path (use provided aggressor sign when present, else existing bulk-volume proxy); new `vpin_robustness(markets)` returning both classifications' scores and their rank correlation; docstrings state the Andersen–Bondarenko caveat and the gating-only role.
**Test scenarios:**
- Side-provided vs proxy paths both run on the same synthetic market; outputs differ when price-change proxy misclassifies (constructed fixture).
- Volume control: VPIN score decorrelates from raw volume after control on a constructed high-volume/no-toxicity fixture.
- Backward compatibility: existing prefilter tests pass unchanged when side is absent.
**Verification:** tests green; no change to `prefilter_wallets` default behavior.

### U4. No-lookahead event study

**Goal:** Temporal informativeness evidence: elevated `P(Z)` precedes terminal moves.
**Requirements:** R6, KTD4.
**Dependencies:** plan 4 U6 (replay outputs).
**Files:** `scripts/event_study.py` (new), `src/analysis/event_study.py` (new — logic), `tests/test_event_study.py` (new), `tests/test_scripts.py`.
**Approach:** Consume replay-score JSONL + market resolution metadata; statistic and permutation scheme per KTD4, locked on synthetic first: markets with planted late-insider activity → statistic separates them from null markets at known rates. CLI: `--scores`, `--window`, `--n-permutations`, `--json-out`; figures (elevation vs terminal move scatter, permutation null histogram) to `results/figures/event_study/`.
**Test scenarios:**
- Power check (synthetic): planted signal → p-value small; no-signal null → p-value uniform-ish across repeated generation (coarse check, ~20 reps).
- Prefix invariance inherited: statistic computed from scores that plan 4's replay guarantees are no-lookahead; test asserts the CLI refuses score files lacking replay provenance flag.
- Resolution-join correctness: market with missing resolution metadata → excluded with warning, counted in JSON.
**Verification:** synthetic-calibrated statistic locked; real-data run on the pulled politics markets committed with JSON + figures.

### U5. Van Dyke / Maduro-cluster case study

**Goal:** The labeled-case sanity check on real data.
**Requirements:** R7, KTD5.
**Dependencies:** plan 4 U1/U2 (full history), U6 (replay); this plan U4 (report format reuse).
**Files:** `results/case_studies/van_dyke/markets.json` (manifest), `scripts/case_study.py` (new, thin: manifest → pull → replay → report), `tests/test_scripts.py` (smoke with synthetic manifest).
**Approach:** Manifest per KTD5; pull with `--full-history --pre-resolution-days 0` (the insider window is near the event, not resolution — document this deviation from the default in the report); wallet-anchored replay scoring; report: top-K wallet table for the window, per-trade score timeline figure, DOJ-timeline overlay (Dec 27 2025 – Jan 2 2026 bets; ~$33,034 total per DOJ — primary-source figures only). **Mandatory data-sufficiency subsection (methods-critic):** report each candidate wallet's total trade count against the documented `theta_w` reliability thresholds (prior-dominated below ~20 trades, meaningful above ~100 — ARCHITECTURE §9.5); with ~13 DOJ-alleged bets, state plainly where the ranking is prior-dominated and lean on per-trade `P(Z)` timing evidence rather than wallet ranking for the headline claim.
**Test scenarios:**
- Smoke: synthetic manifest + synthetic scores → report generated, all sections present.
- Window logic: trades outside the manifest's analysis window excluded from the elevation table but present in the timeline.
**Verification:** committed case-study bundle (manifest, JSON, figures, short md report) under `results/case_studies/van_dyke/`; honest-caveats paragraph included in the report template.

### U6 (STRETCH). Deflated-Sharpe costed backtest

**Goal:** PoC-framed tradeability check with honest multiple-testing accounting.
**Requirements:** R8.
**Dependencies:** U4 (uses the same replay scores).
**Files:** `src/analysis/backtest.py` (new), `scripts/backtest.py` (new), `tests/test_backtest.py` (new).
**Approach:** Single pre-declared strategy family (enter on `P(Z)` threshold, exit at resolution), spread + fee model, purged/embargoed walk-forward, deflated Sharpe (Bailey & López de Prado 2014) computed with the empirical variance across trial Sharpes per the PSR/DSR formula — not a naive raw trial count, which overstates the deflator when adjacent thresholds are correlated (methods-critic) — with the full grid disclosed in the JSON. Framed in all outputs as detection-signal evaluation, not validated alpha.
**Test scenarios:** cost model unit tests (fee at p=0.5 maximal; zero at bounds); purge/embargo split property (no overlapping label windows across folds — constructed fixture); deflated-Sharpe monotone in trial count.
**Verification:** only lands with all tests; skipping this unit entirely is an acceptable outcome recorded in STATUS.

---

## Verification Contract

- `python -m pytest -q -m "not slow"` green; full suite green.
- Wallet-mode bit-identity fixture (U1) is the hard gate — plan 1/4 gate numbers reproduce after this plan.
- Event-study statistic validated on synthetic before any real-data run (KTD4 order enforced in review).
- PG freeze audit; no Kendall-tau criteria anywhere.
- Real-data artifacts committed: Kalshi pull sample, event-study JSON/figures, Van Dyke bundle.

## Definition of Done

- U1–U5 landed (U6 optional, explicitly recorded either way in STATUS.md).
- Anonymous scorer importable with the R3 seam and a minimal usage snippet in its module docstring (the Kalshi system's integration point).
- Case-study bundle reproducible from the manifest.
- Negative results (if any) reported in STATUS with the user's framing decision requested, not buried.
- No abandoned experimental code.

---

## Documentation Notes (for /finish docs-updater)

- ARCHITECTURE.md: §5 model — anonymous-mode predictor variant; §8 module map (+`kalshi_api.py`, `event_study.py`, `backtest.py`, new scripts); §9 — Kalshi adapter (pagination scheme, fee model 0.07·p·(1−p), no-identity property — none currently documented in agent_reference, architecture-critic DOC GAP); §17 — scorer seam consumed by the external trading system.
- STATUS.md: roadmap rows for the variant + evaluation; changelog; decisions row "Kalshi variant: anonymous, alpha-intercept, in-repo (2026-07-23)".
- Paper (final /paper cycle): anonymous-venue section (identity asymmetry as a feature), event study + Van Dyke case, VPIN caveat (Andersen–Bondarenko), positioning vs Mitts-Ofir / Gomez-Cram / ILS / Sirolly; detection-PoC framing throughout.
