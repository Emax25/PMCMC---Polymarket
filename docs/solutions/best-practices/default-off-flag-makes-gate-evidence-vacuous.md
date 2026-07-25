---
title: Acceptance evidence collected with a default-off flag validates the old code path
date: 2026-07-25
category: best-practices
module: VEM M-step / benchmark gates
problem_type: best_practice
component: development_workflow
related_components: [testing_framework, tooling]
severity: high
applies_when:
  - "A feature ships behind an opt-in flag that defaults to off"
  - "A plan's acceptance gate is a CLI run that has no way to enable the new flag"
  - "A test asserts a bound on a quantity the new feature is supposed to compute"
tags: [acceptance-criteria, feature-flags, benchmark-gates, vacuous-tests, opt-in, variational-em, evidence]
---

# Acceptance evidence collected with a default-off flag validates the old code path

## Context

Plan `docs/plans/2026-07-23-001-feat-vem-logistic-mstep-plan.md` added an
IRLS-Cauchy logistic M-step for `beta_S`/`beta_Z`. Because the ADF E-step does
not identify `Z`, the block had to be made opt-in:
`variational_em(..., estimate_betas=False)` by default.

Three separate pieces of "evidence" then silently validated the **old** code
path:

- The plan's R6 gate re-run (`scripts/benchmark.py --method vem --gate`) had no
  way to enable the flag at all. The recorded gate PASS (pooled AUC 0.885) was
  bit-for-bit the pre-change number.
- The U3 "null case" test called `variational_em(...)` without the flag, so its
  assertion `abs(beta_S) < 0.5` was checking an untouched `params_init`
  constant — a test that could never fail.
- Two other unit tests drove the IRLS helper directly on oracle `q(Z)`, so no
  test exercised the new block through the public entry point.

Net result: a complete, mathematically sound M-step landed with **zero
end-to-end evidence**. When the flag was finally wired into the bench this
cycle, the gate **FAILED with it on: pooled AUC 0.547 versus 0.9435 off**.

## Guidance

**When a feature ships behind a default-off flag, the acceptance evidence must
be produced with the flag ON — or the plan must state explicitly that acceptance
covers the off path only.** "The gate still passes" is not a claim about the new
code unless the gate ran the new code.

Three concrete checks worth adopting:

1. **Every gate/bench CLI must expose the flag and record it in its output
   JSON.** Exposing it makes on-path evidence *possible*; recording it makes the
   artifact self-describing, so a future reader can tell which path a committed
   number came from. This cycle added `--estimate-betas` to
   `scripts/benchmark.py` and an `estimate_betas` field to the VEM bench
   payload.

2. **A test asserting a bound on a value the feature computes must first assert
   the feature actually ran.** Either enable the flag explicitly and say why in
   the docstring, or assert the value moved off its initialization. A bound
   check against an untouched constant is a green test with no information
   content.

3. **At least one test must reach the new block through the public entry
   point.** Unit tests that drive an internal helper with oracle inputs prove
   the helper's math; they prove nothing about whether the helper is wired in,
   called with the right arguments, or gated correctly.

The tests were repaired accordingly — `test_vem_null_betas_stay_near_zero` in
`tests/test_variational_em.py` now passes `estimate_betas=True` and records the
reason in its docstring, so the assertion can fail again.

## Why This Matters

An opt-in flag creates two products from one plan, and acceptance criteria
written before the flag existed silently attach to whichever one the default
selects. The failure is invisible in exactly the way that matters: every test
green, gate PASS recorded in the changelog, plan closed. The wrong number ends
up in the project's status history and gets treated as a measured baseline by
later cycles.

Here the on-path number (AUC 0.547) is barely above chance, versus 0.9435 off.
That gap is real information about a genuine identifiability problem in the ADF
E-step — information the plan's own gate was structurally incapable of
surfacing. The cost of the vacuous evidence was not just a false PASS, it was
delaying the discovery of the underlying issue by a full cycle.

## When to Apply

- Any plan whose implementation notes contain "made opt-in", "default off",
  "behind a flag", or "guarded by a config option" — re-read every acceptance
  criterion in that plan and ask which path it exercises.
- Reviewing a diff that adds a keyword-argument gate to an existing function:
  grep the test suite and the CLIs for call sites and check whether *any* of
  them pass the new argument.
- Reading a committed benchmark artifact: if the payload does not record the
  feature flags in effect, treat its numbers as attributable to the defaults at
  that commit, not to the feature named in the changelog line.

## Examples

The wiring that made on-path evidence possible (`scripts/benchmark.py`):

```bash
# Before this cycle: no way to run the gate on the new code path at all.
python -m scripts.benchmark --method vem --gate

# After: the flag is exposed, threaded through to variational_em, and recorded.
python -m scripts.benchmark --method vem --gate --estimate-betas
```

The payload now carries the path it ran, so no future reader has to infer it:

```python
"estimate_betas": args.estimate_betas,
```

The test defect and its repair:

```python
# BEFORE — could never fail: with the default estimate_betas=False the betas
# are pinned at params_init's 0.0, so this asserts abs(0.0) < 0.5.
out = variational_em([md], cfg, n_wallets=20, params_init=params, ...)
assert abs(out.params.beta_S) < 0.5

# AFTER — the docstring states why the flag is mandatory here, and the
# assertion is now about the IRLS M-step's null behaviour.
out = variational_em([md], cfg, n_wallets=20, ..., estimate_betas=True)
assert abs(out.params.beta_S) < 0.5, f"beta_S (internal) = {out.params.beta_S:.3f}"
```

## Related

- `scripts/benchmark.py` — `--estimate-betas` flag, the
  `vem_estimate_betas` thread-through, and the `estimate_betas` payload field.
- `src/inference/variational_em.py` — the `estimate_betas` parameter and the
  IRLS block it gates.
- `tests/test_variational_em.py` — `test_vem_null_betas_stay_near_zero` and the
  opt-in stability test, both now explicit about which path they exercise.
- `docs/plans/2026-07-23-001-feat-vem-logistic-mstep-plan.md` — R6 and unit U4,
  the acceptance criteria that could not reach the new code.
- `agent_reference/STATUS.md` roadmap P8 — current status of the beta-estimation
  path and the identifiability question the on-path gate exposed.
- `docs/solutions/best-practices/kendall-tau-acceptance-criterion-sparse-rankings.md`
  — the companion failure mode: a gate that *can* run but measures the wrong
  thing. Together: a gate must be able to reach the code, and must measure
  something attainable once it does.
- `docs/solutions/best-practices/em-fixed-point-is-not-a-posterior-mode.md` —
  the sibling learning from this cycle, including the structural point that a
  plan's stop conditions must state their own preconditions.
