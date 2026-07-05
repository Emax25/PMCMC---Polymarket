---
title: Global rank-correlation gates are invalid when most entities carry no signal
date: 2026-07-05
category: best-practices
module: inference evaluation / benchmark gates
problem_type: best_practice
component: testing_framework
severity: high
applies_when:
  - "Gating an approximate method against a gold-standard sampler on rank agreement of per-entity scores"
  - "Most ranked entities are prior-dominated noise with only a few true positives (sparse signal)"
  - "Two stochastic estimators are compared on a global similarity statistic with a fixed threshold"
tags: [kendall-tau, acceptance-criteria, ranking, variational-em, particle-gibbs, benchmark-gates, sparse-signal]
---

# Global rank-correlation gates are invalid when most entities carry no signal

## Context

The Fast Insider Detection Stage 2 gate required global Kendall τ ≥ 0.85 between
C1 VEM's and C0 Particle Gibbs' posterior-mean `theta_w` rankings (40 wallets, 3
planted insiders). VEM scored τ ≈ -0.005 despite pooled AUC 0.885 and ranking all
insiders 1-2-3 — which read as a VEM failure and nearly triggered implementing the
C3 twisted-CSMC fallback.

## Guidance

Before adopting a similarity-based acceptance criterion between two stochastic
estimators, **measure the criterion's self-consistency ceiling**: run the gold
standard against itself under seed replication (same data, different RNG seed)
and check whether it passes its own gate. Here, a matched PG-vs-PG control
(N=250, 1500 iterations, different seed) reached only τ = 0.787 — below the 0.85
threshold. The criterion was unattainable even for the reference method, so the
gate was invalid, not the candidate method.

Diagnose *why* before replacing it. In this case 37/40 wallets were
prior-dominated (PG θ range [0.010, 0.854] vs VEM's prior-compressed
[0.058, 0.096]), so global τ was dominated by meaningless noise ordering among
low-signal wallets while the signal-bearing head agreed well (top-4 overlap 75%,
weighted τ 0.36, τ within PG's top-10 = 0.60).

Replace the global statistic with criteria that live where the signal is:

- **100% insider recall@K** (top-cutoff rule) across ≥3 synthetic seeds
- **Pooled ROC AUC ≥ 0.85** with per-market floors
- Report weighted τ and top-K overlap **descriptively only** — never as gates

## Why This Matters

A statistically broken gate fails good methods and burns implementation budget
on unnecessary fallbacks (C3 twisted CSMC was nearly built because of this).
Global rank correlation weights every pairwise inversion equally, so with sparse
signal it measures agreement on noise. Any fixed threshold above the
method-vs-itself ceiling is unattainable by construction.

## When to Apply

- Designing acceptance gates for approximate inference (VI, ADF, filtering) vs MCMC
- Any evaluation comparing rankings where true positives are a small fraction of entities
- Whenever a candidate method "fails" a similarity gate while passing task-level metrics (AUC, recall) — audit the gate with a self-consistency control before blaming the method

## Examples

The control run that exposed the ceiling (same synthetic data as the baseline, different seed):

```bash
python -m scripts.benchmark --method pg --seeds 99 \
  --n-particles 250 --n-iter 1500 --n-burnin 300 --n-jobs 8 \
  --gate --compare-theta results/theta_c0_baseline.npy \
  --json-out results/bench_pg_control.json
# → Kendall τ = 0.787 vs the τ ≥ 0.85 gate: the gold standard fails its own gate.
```

Replacement gate, as adopted (STATUS.md resolved decisions, 2026-07-04):
100% insider recall@K + pooled AUC ≥ 0.85 with per-market floors; weighted
τ / top-K overlap reported descriptively.

## Related

- `agent_reference/HANDOFF_FAST_INSIDER_DETECTION.md` §4-5 — measured gate results and the τ diagnostic
- `agent_reference/STATUS.md` — resolved decision "Kendall τ acceptance criterion: INVALIDATED"
- `src/analysis/results.py` — `insider_recall_at_k`, `recall_k_cutoff`, `evaluate_synthetic_gate`
