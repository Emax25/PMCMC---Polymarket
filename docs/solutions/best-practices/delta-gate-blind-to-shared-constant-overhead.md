---
title: A delta-only A/B runtime gate cannot see constant overhead shared by both arms
date: 2026-07-29
category: best-practices
module: ADF filter / hot-path runtime gates
problem_type: best_practice
component: development_workflow
related_components: [testing_framework, tooling]
severity: high
applies_when:
  - "A refactor's acceptance gate is a relative A/B comparison (delta <= X% vs a baseline arm)"
  - "Both arms of the comparison share the same underlying library calls"
  - "A per-item cost in a hot loop has never been measured in absolute terms"
tags: [benchmark-gates, hot-path, scipy, dispatch-overhead, absolute-cost, a-b-testing, adf-filter, logsumexp]
---

# A delta-only A/B runtime gate cannot see constant overhead shared by both arms

## Context

The ADFFilter extraction (plan `2026-07-23-004` U4) carried a carefully
engineered runtime gate: interleaved A/B at gate scale, 15 reps x 4 runs,
frozen pre-refactor baseline arm, measured against an identical-code noise
floor. The refactor passed at +0.10%..+1.35% deltas — a genuinely rigorous
relative measurement.

The /finish review's performance pass then measured the same step in
**absolute** terms and found that `scipy.special.logsumexp` on a 4-element
array cost 68-80 us/call — ~40 Python-level calls of array-API dispatch
(`xp_promote`, `is_torch_array`, ...) to reduce four floats — which was **64%
of the entire 108 us trade step**, sitting next to a numba Kalman kernel that
cost 1.55 us. The A/B gate was structurally incapable of seeing this: the
oversized constant was present in both arms, so it cancelled out of every
delta. STATUS.md had even recorded "logsumexp is the real hot path" a cycle
earlier, but no gate connected that observation to a number.

The fix (`src/inference/adf_filter.py:_logsumexp4`) reproduces scipy's exact
algorithm locally — max separated out, residual sum divided by the tie count,
`log1p(s) + log(m) + a_max` — verified bit-identical against scipy on 570,014
fuzz vectors (ties, -inf, NaN, the trade-0 `-500` pattern) so the repo's
bit-exactness hard rule held. Result: ~5.6x on `ADFFilter.step`, ~4.8x on the
batch E-step, multiplying through every SBC replicate and the live scorer.

## Guidance

**A relative gate answers "did the refactor make it worse?"; it cannot answer
"is it fast?". Hot-path acceptance needs one absolute number per item
alongside the delta.** Concretely:

1. **Record absolute per-item cost (us/trade, us/step) in the gate artifact**,
   not only the A-vs-B ratio. A reader of the committed benchmark should be
   able to ask "what does one step cost and where does it go?"
2. **Compare the per-item cost against an in-repo reference kernel.** The tell
   here was not the absolute number itself but the ratio to the neighbor: a
   4-float reduction costing 44x the adjacent numba Kalman update is a smell
   no threshold table is needed for.
3. **Profile once per gate, not once per project.** A single
   `cProfile`/`line_profiler` pass over the gated loop attributes the cost;
   here it put 3.48 s of 4.32 s in scipy's dispatch layer, turning a "passed
   the gate" refactor into a 5x win.
4. **Library calls inside per-item loops deserve suspicion proportional to
   (call rate x dispatch generality).** scipy's array-API layer is priced for
   arrays, not for 4 scalars in a 200k-iteration loop. Replacing it is safe
   when the replacement is pinned bit-exact (fuzz + fixture), which the
   identity-fixture infrastructure already made cheap.

## Why This Matters

The cost of the blind spot compounds: this one call ran once per trade in the
batch E-step (T x 50 EM iterations = 17.9 min of E-step at gate scale, 3.6 min
after), once per trade in the streaming scorer, and inside every planned SBC
replicate — hundreds of fits whose budget ceiling (R6's 4-hour guardrail) was
sized around a step that was 64% dispatch overhead. A delta-only gate certifies
such a step forever: every future refactor will also "pass" against it.

## When to Apply

- Writing any plan KTD that gates a refactor at "<= X% regression" — add
  "record absolute per-item cost + top-1 profile attribution" to the same
  gate.
- Reviewing a passed runtime gate: ask what the baseline arm's absolute
  per-item cost was and whether anyone has ever profiled inside it.
- Seeing a general-purpose library call (scipy/pandas/sklearn) on a tiny,
  fixed-size input inside a per-item loop next to compiled kernels.

## Examples

The gate that passed and the measurement that mattered:

```text
A/B gate (relative):  +0.10%..+1.35% vs +1.60% noise floor  -> PASS
Absolute profile:     logsumexp 68.2 us of 107.7 us/trade (64%) -> 5.6x win
```

The replacement contract (src/inference/adf_filter.py): a local
`_logsumexp4` pinned to scipy's algorithm operation-for-operation, gated by
the exact-equality identity fixture (`tests/fixtures/adf_filter_identity.npz`)
plus a 570k-vector bit-equality fuzz test
(`tests/test_adf_filter.py::test_logsumexp4_is_bit_identical_to_scipy`), so a
future scipy algorithm change cannot silently break output identity.

## Related

- `docs/solutions/best-practices/default-off-flag-makes-gate-evidence-vacuous.md`
  — sibling gate-evidence failure: a gate that never runs the new code.
  Together with this doc: a gate must reach the code, measure something
  attainable, *and* measure the right axis (absolute, not only relative).
- `docs/solutions/best-practices/kendall-tau-acceptance-criterion-sparse-rankings.md`
  — a gate that runs but measures the wrong quantity.
- `src/inference/adf_filter.py` — `_logsumexp4` and its bit-exactness comment.
- `agent_reference/STATUS.md` — the earlier "logsumexp is the real hot path"
  note that lacked a number until this cycle.
