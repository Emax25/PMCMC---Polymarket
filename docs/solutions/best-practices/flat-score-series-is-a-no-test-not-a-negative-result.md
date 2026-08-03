---
title: A flat score series is a no-test, not a negative result
date: 2026-08-03
category: best-practices
module: Van Dyke case study / negative-result gating
problem_type: best_practice
component: development_workflow
related_components: [testing_framework, documentation]
severity: high
applies_when:
  - "Writing up a null or negative result from a model run"
  - "A run's validity is checked by asking whether a warm start / fit / config was present"
  - "A claim rests on one unit's score level (an anchored wallet, a held-out case, a single label)"
  - "A summary artifact reports means and maxima but no dispersion"
tags: [negative-results, no-test, degenerate-scores, warm-start, provenance, case-study, evidence, guard-rails]
---

# A flat score series is a no-test, not a negative result

## Context

The Van Dyke case study is the project's only externally-labeled insider
episode — the one place a claim about detection can be checked against reality.
It was run twice.

The first run was cold-started and every wallet scored a flat P(Z) = 0.0500.
The existing guard caught it: `CaseStudySummary.is_cold_start`, read off the
replay sidecar's `warm_start: null`, refused to produce a claim.

A real VEM warm start was then fitted and the study re-run. The provenance now
looked healthy — `warm_start` pointed at a fitted artifact — and the
cluster-wide scores genuinely varied: 1,285 distinct values over 77,766 trades,
range 0.000–0.077. The cold-start guard correctly did not fire. The run was
written up as a **negative result**: "the model does not detect the one labeled
insider we have", anchored wallet mean P(Z) 0.050 against a 0.050 baseline,
elevation −0.000, rank 3347 of 6110.

`/finish`'s methods-critic found that this was analytically forced. The fitted
artifact had `estimate_betas: false`, so `beta_S = beta_Z = 0` and both the size
and persistence channels were gone. `sigma2_0 == sigma2_1` to machine precision
(the P10 order-constraint bind). And the anchored wallet was absent from the
training wallet index, so its `theta_w` sat at the Beta(1,19) prior mean.
`logit(pi_Z)` was therefore a **constant** for that wallet: its 13 in-window
trades scored 0.050000 with a spread of 7.26e-11. No data configuration could
have produced an elevation. The published number was not a measurement of the
detector at all.

## Guidance

**A null result requires proving the measurement was capable of a non-null.**
Otherwise "we looked and saw nothing" is indistinguishable from "we were never
able to look", and only the first is evidence.

Three concrete practices follow:

1. **Do not let a provenance check stand in for a capability check.** "Is a warm
   start present?" and "did the model have a channel through which it could
   respond?" are different questions. The first one passing is exactly what made
   the second easy to skip — the cold-start guard was doing its job, and its
   green light was read as a broader all-clear than it ever claimed.

2. **Check the unit under test, not the aggregate.** Cluster-wide score
   variation is what made the run look alive. It says nothing about whether the
   *specific* wallet the claim is about was scoreable. Dispersion has to be
   evaluated on the unit the claim rests on.

3. **Detect the degeneracy from the output, not from the input config.** The
   fix added `WalletRow.min_p_z` and `WalletRow.is_flat` (spread `< 1e-6`),
   which drive `CaseStudySummary.anchor_is_untested`, a report banner, the
   headline claim, the caveat list, and a CLI warning. Gating on the *scores*
   rather than on the warm-start artifact matters because the degeneracy has
   several possible causes — betas not estimated, an unseen wallet pinned at the
   `theta_w` prior, a bound `sigma2` order constraint — and the constant series
   is the one symptom common to all of them. It therefore catches causes nobody
   enumerated in advance.

A fourth, cheaper practice would have surfaced the whole thing on day one:
**report the spread wherever a claim rests on a level.** The tell was in the
artifact all along and no reader could see it, because `summary.json` carried
`mean_p_z` and `max_p_z` but not `min_p_z`. A statistic whose dispersion is not
reported cannot be sanity-checked.

## Why This Matters

A retracted negative result is more expensive than a missing one. This claim was
the project's headline finding about its only labeled episode; left standing, it
would have been quoted forward as a measured property of the detector and used
to justify model changes aimed at a failure that had not been observed.

The failure mode is also quiet in a specific way. Every surface downstream of
the run behaved correctly: the scores were real floats, the guard that existed
did not fire because its precondition genuinely was not met, the report
rendered, the bundle built. Nothing errored. The only visible trace was a number
that happened to equal the baseline — which reads as a finding rather than as an
absence of one.

## When to Apply

- Any writeup containing "the model does not detect", "no elevation", "no
  effect", or "consistent with the null" — before publishing, show that the
  measurement could have produced the alternative.
- Any run whose validity is established by a provenance or config field being
  non-null. Ask what that field being populated does *not* guarantee.
- Reading a committed summary artifact: if it reports a level without a spread,
  treat the level as unverified. Add the spread rather than trusting the mean.
- Reviewing a fitted-artifact hand-off: check whether the estimated parameters
  actually leave a live channel from the data to the score for the units in the
  claim, not just that the artifact loaded.

## Examples

The guard, keyed on output degeneracy rather than input configuration
(`src/analysis/case_study.py`):

```python
@property
def is_flat(self) -> bool:
    """Whether this wallet's in-window scores carry no dispersion at all."""
    return (self.max_p_z - self.min_p_z) < _FLAT_SCORE_TOL


@property
def anchor_is_untested(self) -> bool:
    """Whether the run carries no information about the anchored wallet."""
    return len(self.anchored_rows) == 1 and self.anchored_rows[0].is_flat
```

The artifact change that makes the degeneracy readable by a human — `min_p_z`
and `flat` alongside the levels that were already there:

```python
"mean_p_z": self.mean_p_z,
"max_p_z": self.max_p_z,
"min_p_z": self.min_p_z,   # added: level without spread is unverifiable
"elevation": self.elevation,
"flat": self.is_flat,
```

The claim before and after:

```text
# BEFORE — a measurement that was never taken.
The model does not detect the one labeled insider we have
(mean P(Z) 0.050 vs baseline 0.050, elevation -0.000, rank 3347/6110).

# AFTER — the run is reported as what it is.
UNTESTED ANCHOR: the anchored wallet's 13 in-window scores span 7.26e-11,
below the 1e-6 flatness tolerance. This run carries no information about
whether the model detects this trader.
```

## Related

- `src/analysis/case_study.py` — `WalletRow.min_p_z` / `is_flat`,
  `CaseStudySummary.anchor_is_untested`, and the banner/caveat ordering that
  puts a run-invalidating warning ahead of qualifying caveats.
- `scripts/case_study.py` — the CLI warning emitted for both the cold-start and
  untested-anchor branches.
- `tests/test_scripts.py` — the two tests pinning both branches of the flatness
  guard, including the one asserting the degeneracy does *not* require a cold
  start.
- `results/case_studies/van_dyke/summary.json` and `report.md` — the retracted
  claim and the banner that replaced it (commit `a105253`).
- `docs/solutions/best-practices/default-off-flag-makes-gate-evidence-vacuous.md`
  — the same failure family from the input side: evidence collected on a code
  path the feature was never on. Together: a gate must reach the code, and the
  code must have a live channel to the outcome, before a null means anything.
  Note that `estimate_betas: false` is the *same* flag in both stories.
- `agent_reference/STATUS.md` — current status of the Van Dyke study and the
  warm-start fit-quality caveats every score inherits.
