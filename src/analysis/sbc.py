"""Simulation-based calibration (SBC) and interval coverage for VEM + Laplace.

This module hosts **both halves** of the SBC pipeline:

  * *Replicate execution* (``run_replicate`` / ``run_sbc``) — draw ``phi`` from
    the prior, simulate a prior-predictive dataset, fit VEM + Laplace, and emit
    one result row per replicate into an append-only JSONL store. The row
    schema is pinned by ``SBC_SCHEMA_VERSION`` and spelled out below.
  * *Analysis* — read the accumulated JSONL back and turn it into the
    calibration evidence: per-component rank uniformity, nominal-90% interval
    coverage, failure accounting, and a JSON summary.

The halves are decoupled through the on-disk JSONL store (plan
``2026-07-23-003`` KTD4): a killed run resumes, and re-analysing a finished run
costs no fits.

Rank statistic (Talts et al. 2018): with ``L`` i.i.d. posterior draws per
replicate, ``rank = #{l : phi_draw_l < phi_true}`` lies in ``{0, ..., L}`` and
is discrete-uniform over those ``L + 1`` values if and only if the posterior is
calibrated. Deviations read as:

  * U-shape (mass piled at both ends) — posterior too narrow, *overconfident*.
  * inverted-U (mass in the middle)   — posterior too wide, *underconfident*.
  * monotone slope                    — posterior location *biased*.

On the *current* Laplace layer the first reading is not the likely explanation:
STATUS.md P8 records ``PhiPosterior`` using expected-complete-data (ECM)
curvature in place of observed information — 113x over-precise on one axis and
mis-centred — so until P8 lands a phi-rank U-shape is expected to be dominated
by that known proposal defect rather than by the inference under test (see
``docs/solutions/best-practices/em-fixed-point-is-not-a-posterior-mode.md``).

Two numeric verdicts back those shapes so nothing depends on eyeballing a plot:
a chi-square bin test (the decision, ``p_value``/``rejected``) and a
Dvoretzky-Kiefer-Wolfowitz simultaneous ECDF band (``band_violation``, and the
band drawn on the figure). A signed pair of z-scores (``bias_z``,
``dispersion_z``) names *which* of the three shapes dominates.

Both verdicts are corrected for testing all eight phi components at once (Holm
for the p-values, Bonferroni for the bands), each at family-wise ``alpha``, and
the coverage table is Bonferroni-corrected across its own rows. The three
families are *not* corrected against each other, so a perfectly calibrated run
raises some flag with probability up to ``3 * alpha`` — roughly one run in
seven at the default 0.05 (``ALPHA_FAMILY_NOTE``, surfaced in the summary JSON
as ``alpha_note``). A lone flag on one component is weak evidence; the failure
this analysis exists to catch shows up across components and as a coverage
miss too.

Coverage is judged on the *interval*, never on the point estimate: a row fails
only when its Wilson CI excludes the nominal level, and passes-but-inconclusive
when the CI overlaps the nominal level yet is wider than the R4 acceptance
band. A point-estimate gate would mostly measure the replicate count — at
n = 30 a perfectly calibrated 0.90 falls outside [0.85, 0.95] about a third of
the time — so the summary reports conclusiveness alongside the verdict.

Degenerate and failed replicates are never silently dropped (R5): failed rows
carry no ranks and are excluded from every statistic but counted in the failure
rate; degenerate rows (VEM non-convergence or a Laplace curvature fallback) are
excluded by default and reported both as a rate and as a sensitivity block that
re-runs coverage with them included.

Row schema (``schema_version = 2``), one JSON object per line::

    {"schema_version": 2, "seed": int, "phi_true": {component: float},
     "L": int, "ranks": {component: int in 0..L} | null,
     "hits90": {component: bool} | null, "theta_hits90": [bool, ...] | null,
     "z_auc": float | null,
     "flags": {"vem_converged": bool, "laplace_fallback": bool,
               "failed": bool, "error": str | null},
     "elapsed_s": float, "size": {"K": int, "T": int, "n_wallets": int},
     "prior": {hyperparameter: float}}

Every row therefore carries its whole generating regime — ``L``, ``size`` and
the ``prior`` fingerprint. That is what lets ``run_sbc`` refuse to append to a
store built under a different regime (and ``analyze`` refuse to pool one),
instead of silently mixing two incomparable sets of ranks. Schema 2 added
``prior`` for exactly that check; ``load_results`` rejects every other version
outright, and no v1 store was ever produced, so there is nothing to migrate.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import binom, chi2, norm

from config.default_params import InferenceConfig, ModelParams, PhiPrior
from src.data.synthetic import (
    SyntheticMarket,
    generate_prior_predictive_market,
    params_from_prior,
)
from src.utils.transforms import logit

log = logging.getLogger(__name__)

SBC_SCHEMA_VERSION = 2

# Canonical phi order. Deliberately restated rather than imported from
# `src.inference.diagnostics.PHI_PARAM_NAMES` or `src.inference.laplace`: both
# pull the PG/iPMCMC inference stack in transitively, and the analysis half of
# this module only ever reads JSON (the replicate half defers those imports into
# `run_replicate` so `--analyze` never pays for them). `tests/test_sbc.py`
# asserts this tuple still equals `PHI_PARAM_NAMES` and `PhiPosterior.dims`, so
# the duplication cannot drift unnoticed.
PHI_COMPONENTS = (
    "sigma2_0",
    "sigma2_1",
    "q_01",
    "q_10",
    "beta_S",
    "beta_Z",
    "tau2_0",
    "tau2_1",
)

THETA_W_KEY = "theta_w"

NOMINAL_COVERAGE = 0.90
COVERAGE_BAND = (0.85, 0.95)  # R4 acceptance window for nominal-90% intervals
DEFAULT_ALPHA = 0.05
FAILURE_RATE_FLAG = 0.05  # plan verification contract: failures must stay < 5%

# Rank-histogram binning. The chi-square approximation wants >= 5 expected
# counts per bin, and more than ~20 bins buys resolution nobody reads off a
# 2-inch paper panel.
MAX_RANK_BINS = 20
MIN_EXPECTED_PER_BIN = 5

RANK_INTERPRETATION_KEY = (
    "Interpretation: ranks piled at both ends (U-shaped histogram, dip-then-rise "
    "in the ECDF difference) = posterior too narrow, overconfident; ranks piled "
    "in the middle = too wide, underconfident; a monotone slope = biased "
    "location. The shaded band holds jointly over every panel, so a curve "
    "staying inside it is the calibrated case. Caveat: until STATUS.md P8 "
    "(observed information via the Louis identity, and a real posterior mode) "
    "lands, a phi-rank U-shape here is expected to be dominated by the known "
    "mis-calibration of the Laplace proposal itself rather than by the "
    "inference under test - see docs/solutions/best-practices/"
    "em-fixed-point-is-not-a-posterior-mode.md."
)

_THETA_DEPENDENCE_NOTE = (
    "Wallet intervals are pooled across replicates; wallets within a replicate "
    "share one fit, so this binomial CI ignores that dependence and is "
    "anti-conservative. Read the width as a lower bound."
)

# Three separately corrected families of tests are reported side by side. Each
# holds at family-wise `alpha` on its own; nothing corrects them against each
# other, so the honest bound on a calibrated run raising *some* flag is their
# union. Stated verbatim in the summary JSON so a reader quoting "one component
# flagged" cannot mistake it for a 5% event.
ALPHA_FAMILY_NOTE = (
    "Three test families are reported, each controlled at family-wise alpha on "
    "its own: (1) chi-square rank uniformity, Holm-adjusted across components; "
    "(2) the DKW simultaneous ECDF band, Bonferroni-widened across components; "
    "(3) interval coverage, Bonferroni-widened across the coverage table's "
    "rows. They are not corrected against each other, so `flagged` unions them "
    "and the overall false-alarm probability on a perfectly calibrated run is "
    "bounded by 3 * alpha (about 14% at alpha = 0.05), not alpha. Read a single "
    "flag accordingly; a real failure shows up in more than one family."
)

_COVERAGE_POWER_NOTE = (
    "A coverage row passes when its Wilson CI overlaps the nominal level and "
    "is conclusive when the CI also fits entirely inside the acceptance band, "
    "i.e. n was large enough to rule out a coverage error big enough to "
    "matter. At nominal 0.90, a [0.85, 0.95] band and the table-wide Bonferroni "
    "correction that needs roughly 400 replicates; a passing row below that is "
    "underpowered and means 'no evidence of miscalibration', not 'calibration "
    "demonstrated'."
)


# ---------------- Row I/O and selection ----------------


def prior_fingerprint(prior: PhiPrior) -> dict[str, float]:
    """Serialize the prior a replicate was simulated from and fitted against.

    Recorded verbatim (all six hyperparameters, sorted) rather than hashed: a
    conflicting store then reports *which* hyperparameter moved, which is the
    only actionable thing to say when a resume is refused. ``PhiPrior`` holds
    nothing but floats, so the dict is JSON-safe and compares exactly.

    Args:
        prior: The prior threaded into the generator, the M-step and the
            curvature alike (R1/KTD5).

    Returns:
        ``{hyperparameter: value}`` in sorted key order.
    """
    return {name: float(value) for name, value in sorted(asdict(prior).items())}


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Read an SBC results JSONL into a list of rows, de-duplicated by seed.

    Blank lines are skipped; anything else that is not a well-formed row of the
    expected schema raises, because a silently dropped replicate is exactly the
    failure mode SBC bookkeeping exists to prevent. Duplicate seeds (possible if
    a ``--resume`` run re-ran a completed replicate) keep the first occurrence
    and are logged, so re-analysis is deterministic in file order.

    Args:
        path: Path to the append-only JSONL store.

    Returns:
        Rows in file order, one per unique seed.

    Raises:
        ValueError: If a line is not valid JSON, carries a ``schema_version``
            other than ``SBC_SCHEMA_VERSION``, or has no ``seed``.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    n_duplicates = 0
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON ({exc})") from exc
            version = row.get("schema_version")
            if version != SBC_SCHEMA_VERSION:
                raise ValueError(
                    f"{path}:{lineno}: schema_version {version!r}, expected "
                    f"{SBC_SCHEMA_VERSION}",
                )
            seed = row.get("seed")
            if seed is None:
                raise ValueError(f"{path}:{lineno}: row has no 'seed'")
            if seed in seen:
                n_duplicates += 1
                continue
            seen.add(seed)
            rows.append(row)
    if n_duplicates:
        log.warning(
            "%s: dropped %d duplicate-seed row(s); kept the first of each",
            path,
            n_duplicates,
        )
    return rows


def row_failed(row: dict[str, Any]) -> bool:
    """Whether a replicate produced no usable ranks (crashed or flagged failed)."""
    flags = row.get("flags") or {}
    return bool(flags.get("failed", False)) or row.get("ranks") is None


def row_degenerate(row: dict[str, Any]) -> bool:
    """Whether a replicate fit is degenerate: non-converged VEM or Laplace fallback."""
    flags = row.get("flags") or {}
    return not bool(flags.get("vem_converged", True)) or bool(
        flags.get("laplace_fallback", False),
    )


def usable_rows(
    rows: Sequence[dict[str, Any]],
    *,
    include_degenerate: bool = False,
) -> list[dict[str, Any]]:
    """Select the replicates that contribute to rank and coverage statistics.

    Failed rows are always excluded — they carry ``ranks = null``. Degenerate
    rows are excluded by default (R5); note that conditioning on a successful,
    converged fit is itself an approximation, which is why ``analyze`` reports
    the coverage that *would* have been obtained with them included.

    Args:
        rows: Rows as returned by ``load_results``.
        include_degenerate: Keep non-converged / Laplace-fallback replicates.

    Returns:
        The retained rows, in input order.
    """
    kept = [r for r in rows if not row_failed(r)]
    if include_degenerate:
        return kept
    return [r for r in kept if not row_degenerate(r)]


def _single_L(rows: Sequence[dict[str, Any]]) -> int:
    """Return the common posterior-draw count L, or raise if replicates disagree."""
    values = sorted({int(r["L"]) for r in rows})
    if not values:
        raise ValueError("no replicates to analyse")
    if len(values) > 1:
        raise ValueError(
            f"replicates disagree on L (posterior draws per replicate): {values}; "
            "rank bins are only comparable at a single L",
        )
    return values[0]


def _distinct_priors(rows: Sequence[dict[str, Any]]) -> list[dict[str, float]]:
    """Distinct prior fingerprints across rows, in first-seen order."""
    seen: list[dict[str, float]] = []
    for row in rows:
        fingerprint = row.get("prior") or {}
        if fingerprint not in seen:
            seen.append(fingerprint)
    return seen


def _single_prior(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Return the common prior fingerprint, or raise if replicates disagree.

    A mixed prior is a harder error than a mixed ``L``: the rank statistic is
    uniform only when every replicate was simulated from *and* fitted against
    the one same density (R1/KTD5), so a histogram pooled over two priors tests
    nothing at all and must never be reported as calibration evidence.
    """
    values = _distinct_priors(rows)
    if not values:
        raise ValueError("no replicates to analyse")
    if len(values) > 1:
        raise ValueError(
            f"replicates disagree on the prior: {values}; SBC ranks are only "
            "comparable within one prior",
        )
    return values[0]


# ---------------- Rank uniformity ----------------


@dataclass(frozen=True, eq=False)
class UniformityResult:
    """Rank-uniformity verdict for one phi component.

    Attributes:
        component: Name of the phi component.
        n: Replicates contributing ranks.
        L: Posterior draws per replicate; ranks live in ``{0, ..., L}``.
        n_bins: Bins used by the chi-square test.
        bin_edges: Integer rank boundaries, shape ``(n_bins + 1,)``, running
            from 0 to ``L + 1``; bin ``j`` holds ranks in
            ``[bin_edges[j], bin_edges[j + 1])``.
        counts: Observed count per bin.
        expected: Expected count per bin under discrete uniformity. Bins are
            not exactly equiprobable when ``L + 1`` is not a multiple of
            ``n_bins``, so this is computed from the exact integer bin widths.
        band_lo: Per-bin lower binomial band, Bonferroni-adjusted across bins.
        band_hi: Per-bin upper binomial band.
        chi2_stat: Pearson chi-square statistic over the bins.
        dof: ``n_bins - 1`` — no parameters are estimated from the ranks.
        p_value: Raw upper-tail chi-square p-value for this component alone.
        p_value_adj: ``p_value`` after a Holm-Bonferroni step-down across the
            components tested together — the eight phi components are eight
            simultaneous tests, so an unadjusted 0.05 would flag a *perfectly
            calibrated* pipeline about a third of the time.
        rejected: ``p_value_adj < alpha``; the family-wise decision.
        pit: Continuity-corrected rank PIT values ``(rank + 0.5) / (L + 1)``,
            shape ``(n,)``. Not serialized.
        ks_stat: Two-sided Kolmogorov-Smirnov distance of ``pit`` from U(0, 1).
        band_half_width: DKW simultaneous half-width at ``alpha``.
        ks_floor: ``1 / (2 * (L + 1))`` — the KS distance a *perfectly*
            calibrated discrete PIT cannot go below, because the continuity-
            corrected ranks live on ``L + 1`` atoms while the band compares them
            against the continuous U(0, 1) CDF. ``band_half_width`` shrinks like
            ``1 / sqrt(n)`` but the floor does not, so a small ``L`` with a large
            ``n`` produces band violations from discretization alone; when
            ``band_half_width`` is not comfortably above ``ks_floor`` the band is
            testing the grid, not the posterior.
        band_violation: ``ks_stat > band_half_width``.
        mean_pit: Sample mean of ``pit``; 0.5 under calibration.
        var_pit: Sample variance of ``pit``; 1/12 under calibration.
        bias_z: Standardized ``mean_pit - 0.5``; the slope/location signal.
        dispersion_z: Standardized ``var_pit - 1/12``; positive is U-shaped
            (overconfident), negative inverted-U (underconfident).
        shape_hint: Human-readable naming of the dominant deviation.
        alpha: Family-wise significance level for the decision and the bands.
        band_alpha: Per-component level the bands were drawn at, i.e.
            ``alpha / n_components``, so the panels hold jointly at ``alpha``.
    """

    component: str
    n: int
    L: int
    n_bins: int
    bin_edges: np.ndarray
    counts: np.ndarray
    expected: np.ndarray
    band_lo: np.ndarray
    band_hi: np.ndarray
    chi2_stat: float
    dof: int
    p_value: float
    p_value_adj: float
    rejected: bool
    pit: np.ndarray
    ks_stat: float
    band_half_width: float
    ks_floor: float
    band_violation: bool
    mean_pit: float
    var_pit: float
    bias_z: float
    dispersion_z: float
    shape_hint: str
    alpha: float
    band_alpha: float

    @property
    def flagged(self) -> bool:
        """Whether either numeric verdict says the ranks are non-uniform.

        Note:
            This is a *union* of two families that are each controlled at
            ``alpha`` separately (Holm over the chi-square p-values, Bonferroni
            over the DKW bands) and are not corrected against each other. The
            union bound on flagging a perfectly calibrated component is
            therefore ``2 * alpha``, and adding the coverage family takes the
            whole-report bound to ``3 * alpha`` — see ``ALPHA_FAMILY_NOTE``,
            which ships in the summary JSON as ``alpha_note``.
        """
        return self.rejected or self.band_violation

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view; the per-replicate ``pit`` array is omitted."""
        return {
            "component": self.component,
            "n": self.n,
            "L": self.L,
            "n_bins": self.n_bins,
            "bin_edges": [int(e) for e in self.bin_edges],
            "counts": [int(c) for c in self.counts],
            "expected": [float(e) for e in self.expected],
            "band_lo": [float(v) for v in self.band_lo],
            "band_hi": [float(v) for v in self.band_hi],
            "chi2_stat": self.chi2_stat,
            "dof": self.dof,
            "p_value": self.p_value,
            "p_value_adj": self.p_value_adj,
            "rejected": self.rejected,
            "ks_stat": self.ks_stat,
            "band_half_width": self.band_half_width,
            "ks_floor": self.ks_floor,
            "band_violation": self.band_violation,
            "mean_pit": self.mean_pit,
            "var_pit": self.var_pit,
            "bias_z": self.bias_z,
            "dispersion_z": self.dispersion_z,
            "shape_hint": self.shape_hint,
            "flagged": self.flagged,
            "alpha": self.alpha,
            "band_alpha": self.band_alpha,
        }


def rank_bin_edges(L: int, n_bins: int) -> np.ndarray:
    """Integer bin boundaries partitioning the rank support ``{0, ..., L}``.

    Edge ``j`` is ``ceil(j * (L + 1) / n_bins)`` evaluated in exact integer
    arithmetic — floating-point ``linspace`` edges land a ULP either side of an
    integer and would move whole ranks between bins.

    Args:
        L: Posterior draws per replicate.
        n_bins: Number of bins; must be at least 1.

    Returns:
        Non-decreasing integer array of shape ``(n_bins + 1,)`` from 0 to
        ``L + 1``.
    """
    j = np.arange(n_bins + 1, dtype=np.int64)
    return -((-j * (L + 1)) // n_bins)


def default_n_bins(n: int, L: int) -> int:
    """Choose a rank-histogram bin count for ``n`` replicates and support size L+1.

    Keeps the chi-square approximation honest (>= ``MIN_EXPECTED_PER_BIN``
    expected per bin) while never exceeding ``MAX_RANK_BINS`` or the number of
    distinct ranks available.

    The ``max(2, ...)`` floor deliberately *overrides* the expected-count
    guidance rather than returning a single meaningless bin: a one-bin
    chi-square has zero degrees of freedom and tests nothing. Below
    ``2 * MIN_EXPECTED_PER_BIN`` replicates that floor is therefore reached by
    violating the guidance, which is logged loudly — the chi-square p-value is
    not trustworthy there and the run needs more replicates, not a different
    bin count.

    Args:
        n: Replicate count contributing ranks.
        L: Posterior draws per replicate.

    Returns:
        Bin count, at least 2.
    """
    n_bins = max(2, min(MAX_RANK_BINS, n // MIN_EXPECTED_PER_BIN, L + 1))
    expected_per_bin = n / n_bins if n_bins else 0.0
    if expected_per_bin < MIN_EXPECTED_PER_BIN:
        log.warning(
            "rank histogram has only %.1f expected counts per bin at n=%d over "
            "%d bins (floor is %d): the chi-square approximation is unreliable "
            "and its p_value should not be quoted as evidence",
            expected_per_bin,
            n,
            n_bins,
            MIN_EXPECTED_PER_BIN,
        )
    return n_bins


def _shape_hint(bias_z: float, dispersion_z: float, threshold: float) -> str:
    """Name the dominant rank-histogram deviation from the two signed z-scores.

    Args:
        bias_z: Standardized departure of the mean PIT from 0.5.
        dispersion_z: Standardized departure of the PIT variance from 1/12.
        threshold: ``|z|`` above which a departure is called dominant. Tied to
            the same corrected level as the tests, so a component reported as
            uniform is not simultaneously labelled "biased".

    Returns:
        A one-phrase description of the dominant deviation.
    """
    if abs(bias_z) < threshold and abs(dispersion_z) < threshold:
        return "no dominant deviation"
    if abs(dispersion_z) >= abs(bias_z):
        if dispersion_z > 0.0:
            return "U-shape: posterior too narrow (overconfident)"
        return "inverted-U: posterior too wide (underconfident)"
    if bias_z > 0.0:
        return "slope: posterior biased low (truth ranks high among draws)"
    return "slope: posterior biased high (truth ranks low among draws)"


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment across simultaneously tested components.

    Args:
        p_values: Raw p-value per component.

    Returns:
        Adjusted p-values, keyed as the input; comparing them to ``alpha``
        controls the family-wise error rate across the whole set.
    """
    m = len(p_values)
    running = 0.0
    adjusted: dict[str, float] = {}
    for i, name in enumerate(sorted(p_values, key=p_values.__getitem__)):
        running = max(running, min(1.0, (m - i) * p_values[name]))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def _component_uniformity(
    ranks: np.ndarray,
    *,
    component: str,
    L: int,
    n_bins: int,
    alpha: float,
    band_alpha: float,
) -> UniformityResult:
    """Run the chi-square bin test and the DKW ECDF band for one component."""
    n = int(ranks.size)
    edges = rank_bin_edges(L, n_bins)
    widths = np.diff(edges).astype(float)
    probs = widths / float(L + 1)
    # searchsorted over the *interior* edges maps rank r to the unique bin j
    # with edges[j] <= r < edges[j+1], matching how `widths` counts integers.
    bins = np.searchsorted(edges[1:-1], ranks, side="right")
    counts = np.bincount(bins, minlength=n_bins).astype(np.int64)
    expected = n * probs
    chi2_stat = float(np.sum((counts - expected) ** 2 / expected))
    dof = n_bins - 1
    p_value = float(chi2.sf(chi2_stat, dof))

    # Bonferroni across bins turns the pointwise binomial intervals into a
    # (conservative) simultaneous band, so a single bin poking out is evidence
    # rather than the expected one-in-twenty excursion. `band_alpha` already
    # carries the across-component correction.
    bin_alpha = band_alpha / n_bins
    band_lo = np.asarray(binom.ppf(bin_alpha / 2.0, n, probs), dtype=float)
    band_hi = np.asarray(binom.ppf(1.0 - bin_alpha / 2.0, n, probs), dtype=float)

    # Continuity correction maps the discrete rank onto (0, 1) without stacking
    # mass on the endpoints, which is what an ECDF band assumes.
    pit = (ranks + 0.5) / float(L + 1)
    ordered = np.sort(pit)
    steps = np.arange(1, n + 1, dtype=float) / n
    ks_stat = float(max(np.max(steps - ordered), np.max(ordered - (steps - 1.0 / n))))
    # DKW: P(sup|F_n - F| > eps) <= 2 exp(-2 n eps^2), so this half-width is a
    # distribution-free band holding simultaneously over the whole curve.
    band_half_width = math.sqrt(math.log(2.0 / band_alpha) / (2.0 * n))
    # Discretization floor: `pit` sits on the L+1 midpoints k/(L+1) + 1/(2(L+1))
    # while the band compares it against the *continuous* U(0, 1) CDF, so even an
    # exactly calibrated rank distribution leaves a KS distance of 1/(2(L+1))
    # (half a grid step, at the atoms). The band half-width falls off as
    # 1/sqrt(n) and this floor does not, so at small L with large n the band
    # eventually flags the grid rather than the inference - `ks_floor` is
    # reported so that regime is visible instead of read as miscalibration.
    ks_floor = 1.0 / (2.0 * float(L + 1))

    mean_pit = float(np.mean(pit))
    var_pit = float(np.var(pit))
    # Under U(0,1): Var(mean) = 1/(12n); Var(sample variance) ~ (mu4 - s^4)/n
    # = (1/80 - 1/144)/n = 1/(180 n).
    bias_z = (mean_pit - 0.5) / math.sqrt(1.0 / (12.0 * n))
    dispersion_z = (var_pit - 1.0 / 12.0) / math.sqrt(1.0 / (180.0 * n))

    return UniformityResult(
        component=component,
        n=n,
        L=L,
        n_bins=n_bins,
        bin_edges=edges,
        counts=counts,
        expected=expected,
        band_lo=band_lo,
        band_hi=band_hi,
        chi2_stat=chi2_stat,
        dof=dof,
        p_value=p_value,
        # Placeholders: `rank_uniformity` owns the across-component adjustment
        # because only it knows how many components were tested together.
        p_value_adj=p_value,
        rejected=p_value < alpha,
        pit=pit,
        ks_stat=ks_stat,
        band_half_width=band_half_width,
        ks_floor=ks_floor,
        band_violation=ks_stat > band_half_width,
        mean_pit=mean_pit,
        var_pit=var_pit,
        bias_z=bias_z,
        dispersion_z=dispersion_z,
        shape_hint=_shape_hint(
            bias_z,
            dispersion_z,
            float(norm.isf(band_alpha / 2.0)),
        ),
        alpha=alpha,
        band_alpha=band_alpha,
    )


def rank_uniformity(
    rows: Sequence[dict[str, Any]],
    *,
    components: Sequence[str] = PHI_COMPONENTS,
    n_bins: int | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, UniformityResult]:
    """Assess rank uniformity per phi component over already-selected replicates.

    Callers pass rows filtered by ``usable_rows``; this function does no
    filtering of its own so the exclusion policy stays in one place.

    ``alpha`` is a *family-wise* level across ``components``: the p-values are
    Holm-adjusted and the bands Bonferroni-widened, so "no component flagged"
    means what a reader assumes it means. Reading eight unadjusted 0.05 tests
    would flag a perfectly calibrated pipeline roughly one run in three, which
    is exactly the false alarm that would derail the paper's claim.

    Args:
        rows: Usable replicate rows, each carrying a full ``ranks`` mapping.
        components: Phi components to assess, in report order.
        n_bins: Chi-square bin count; ``None`` picks ``default_n_bins``.
        alpha: Family-wise significance level for the tests and both bands.

    Returns:
        Mapping from component name to its ``UniformityResult``.

    Raises:
        ValueError: If ``rows`` is empty, replicates disagree on ``L``, a row is
            missing a component's rank, or a rank falls outside ``{0, ..., L}``.
    """
    if not rows:
        raise ValueError("no replicates to analyse")
    if not components:
        raise ValueError("components is empty; nothing to assess")
    L = _single_L(rows)
    n_bins = default_n_bins(len(rows), L) if n_bins is None else int(n_bins)
    band_alpha = alpha / len(components)

    results: dict[str, UniformityResult] = {}
    for component in components:
        try:
            raw = [row["ranks"][component] for row in rows]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"a replicate is missing rank {component!r}") from exc
        ranks = np.asarray(raw, dtype=np.int64)
        if ranks.min() < 0 or ranks.max() > L:
            raise ValueError(
                f"rank for {component!r} outside {{0, ..., {L}}}: "
                f"[{ranks.min()}, {ranks.max()}]",
            )
        results[component] = _component_uniformity(
            ranks,
            component=component,
            L=L,
            n_bins=n_bins,
            alpha=alpha,
            band_alpha=band_alpha,
        )

    adjusted = _holm_adjust({k: v.p_value for k, v in results.items()})
    return {
        name: replace(
            result,
            p_value_adj=adjusted[name],
            rejected=adjusted[name] < alpha,
        )
        for name, result in results.items()
    }


# ---------------- Coverage ----------------


def wilson_interval(
    n_hits: int,
    n: int,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald interval: it stays inside [0, 1] and keeps its
    nominal level at the small replicate counts and near-1 rates this analysis
    lives at (a 0.90 hit rate over ~200 replicates).

    Args:
        n_hits: Number of successes.
        n: Number of trials; 0 yields ``(0.0, 1.0)``.
        alpha: Two-sided level, e.g. 0.05 for a 95% interval.

    Returns:
        The ``(lo, hi)`` bounds.
    """
    if n <= 0:
        return (0.0, 1.0)
    z = float(norm.isf(alpha / 2.0))
    p = n_hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z / denom * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class CoverageRow:
    """One line of the coverage table, decided on the interval not the point.

    The verdict reads the Wilson CI, never ``rate`` against ``band`` directly.
    An empirical coverage rate is a binomial proportion, and at the replicate
    counts SBC actually runs at it is far too noisy to compare with a +/- 0.05
    window: at ``n = 30`` a *perfectly calibrated* 0.90 lands outside
    [0.85, 0.95] about a third of the time, so a point-estimate gate mostly
    measures ``n``. The two questions are therefore separated:

      * ``in_range`` — the pass/fail decision. True when the CI *overlaps*
        ``nominal``, i.e. the data are consistent with correct coverage. A row
        fails only when the CI excludes ``nominal``, which is real evidence of
        miscalibration rather than a small-sample artefact.
      * ``conclusive`` — whether the CI also fits entirely inside ``band``, i.e.
        whether ``n`` was large enough to *rule out* a coverage error big enough
        to matter. ``in_range and not conclusive`` is the underpowered verdict:
        no evidence against calibration, and not enough evidence for it either.

    Attributes:
        name: Component name, or ``theta_w`` for the pooled latent row.
        n: Interval draws contributing (replicates, or wallet intervals). Kept
            on the row because the verdict is only readable next to it.
        n_hits: How many contained the truth.
        rate: Empirical coverage ``n_hits / n``.
        ci_lo: Wilson lower bound at ``row_alpha``.
        ci_hi: Wilson upper bound at ``row_alpha``.
        nominal: Nominal interval level the hits were computed at.
        band: R4 acceptance window the CI is checked against for
            conclusiveness.
        row_alpha: Two-sided level this row's CI was built at — the table's
            ``alpha`` Bonferroni-divided by the number of rows, so "no coverage
            row failed" holds jointly over the table.
        in_range: CI overlaps ``nominal``; the pass/fail decision.
        conclusive: CI lies entirely inside ``band``; the power check.
        note: Caveat attached to this row, empty when there is none.
    """

    name: str
    n: int
    n_hits: int
    rate: float
    ci_lo: float
    ci_hi: float
    nominal: float
    band: tuple[float, float]
    row_alpha: float
    in_range: bool
    conclusive: bool
    note: str = ""

    @property
    def verdict(self) -> str:
        """One-token reading of ``(in_range, conclusive)`` for report tables."""
        if not self.in_range:
            return "fail"
        return "pass (conclusive)" if self.conclusive else "pass (underpowered)"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view of the row."""
        return {
            "name": self.name,
            "n": self.n,
            "n_hits": self.n_hits,
            "rate": self.rate,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "nominal": self.nominal,
            "band": list(self.band),
            "row_alpha": self.row_alpha,
            "in_range": self.in_range,
            "conclusive": self.conclusive,
            "verdict": self.verdict,
            "note": self.note,
        }


def _coverage_row(
    name: str,
    hits: Sequence[bool],
    *,
    nominal: float,
    band: tuple[float, float],
    row_alpha: float,
    note: str = "",
) -> CoverageRow:
    """Build one coverage row, deciding pass/fail and power from the Wilson CI."""
    n = len(hits)
    n_hits = int(sum(bool(h) for h in hits))
    rate = n_hits / n if n else 0.0
    lo, hi = wilson_interval(n_hits, n, alpha=row_alpha)
    # An empty row carries no interval at all (`wilson_interval` returns the
    # whole unit interval), so it can be neither a failure nor a pass anyone
    # should read: report it as failing to keep it visible in `flagged`.
    overlaps = bool(n) and lo <= nominal <= hi
    return CoverageRow(
        name=name,
        n=n,
        n_hits=n_hits,
        rate=rate,
        ci_lo=lo,
        ci_hi=hi,
        nominal=nominal,
        band=band,
        row_alpha=row_alpha,
        in_range=overlaps,
        conclusive=overlaps and band[0] <= lo and hi <= band[1],
        note=note,
    )


def _theta_hits(rows: Sequence[dict[str, Any]]) -> list[list[bool]]:
    """Per-replicate wallet-interval hit lists, skipping replicates without any."""
    out: list[list[bool]] = []
    for row in rows:
        hits = row.get("theta_hits90")
        if hits:
            out.append([bool(h) for h in hits])
    return out


def coverage_table(
    rows: Sequence[dict[str, Any]],
    *,
    components: Sequence[str] = PHI_COMPONENTS,
    nominal: float = NOMINAL_COVERAGE,
    band: tuple[float, float] = COVERAGE_BAND,
    alpha: float = DEFAULT_ALPHA,
) -> list[CoverageRow]:
    """Build the nominal-interval coverage table over already-selected replicates.

    One row per phi component, plus a pooled ``theta_w`` row when any replicate
    recorded wallet intervals. Each row's verdict is read off its Wilson CI (see
    ``CoverageRow``); nothing is dropped.

    ``alpha`` is *family-wise across the table*: every row's CI is built at
    ``alpha / n_rows``, matching how ``rank_uniformity`` treats its bands. Nine
    independent 95% decisions would flag a perfectly calibrated pipeline about
    37% of the time, which is precisely the n-blind false alarm this
    interval-based gate exists to remove.

    Args:
        rows: Usable replicate rows, each carrying a full ``hits90`` mapping.
        components: Phi components to tabulate, in report order.
        nominal: Nominal interval level the harness recorded hits at.
        band: Acceptance window used for the conclusiveness check (R4).
        alpha: Family-wise two-sided level for the Wilson CIs.

    Returns:
        Coverage rows in ``components`` order, ``theta_w`` last when present.

    Raises:
        ValueError: If a row is missing a component's hit indicator.
    """
    per_replicate = _theta_hits(rows)
    # Bonferroni denominator has to be known before the first row is built, so
    # the theta_w row's presence is settled up front rather than appended later.
    n_rows = len(components) + (1 if per_replicate else 0)
    row_alpha = alpha / n_rows if n_rows else alpha

    table: list[CoverageRow] = []
    for component in components:
        try:
            hits = [bool(row["hits90"][component]) for row in rows]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"a replicate is missing hit {component!r}") from exc
        table.append(
            _coverage_row(
                component,
                hits,
                nominal=nominal,
                band=band,
                row_alpha=row_alpha,
            ),
        )

    if per_replicate:
        pooled = [h for replicate in per_replicate for h in replicate]
        table.append(
            _coverage_row(
                THETA_W_KEY,
                pooled,
                nominal=nominal,
                band=band,
                row_alpha=row_alpha,
                note=_THETA_DEPENDENCE_NOTE,
            ),
        )
    return table


def _theta_block(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Summarize wallet-interval coverage, including its between-replicate spread."""
    per_replicate = _theta_hits(rows)
    if not per_replicate:
        return None
    means = np.asarray([np.mean(r) for r in per_replicate], dtype=float)
    return {
        "n_replicates": len(per_replicate),
        "n_wallet_intervals": int(sum(len(r) for r in per_replicate)),
        "mean_of_replicate_rates": float(np.mean(means)),
        # The honest uncertainty scale for a clustered rate: how much the
        # per-replicate coverage moves between fits.
        "between_replicate_sd": float(np.std(means, ddof=1)) if means.size > 1 else 0.0,
        "note": _THETA_DEPENDENCE_NOTE,
    }


# ---------------- Failure accounting ----------------


@dataclass(frozen=True)
class FailureAccounting:
    """Replicate-health bookkeeping (R5).

    Attributes:
        n_replicates: Rows read from the store.
        n_failed: Rows that produced no ranks.
        n_scored: ``n_replicates - n_failed``.
        n_degenerate: Scored rows with a non-converged VEM or Laplace fallback.
        n_vem_nonconverged: Scored rows whose VEM did not converge.
        n_laplace_fallback: Scored rows whose Laplace curvature fell back.
        failure_rate: ``n_failed / n_replicates``.
        degenerate_rate: ``n_degenerate / n_scored``.
        error_counts: Error string to occurrence count, for failed rows.
    """

    n_replicates: int
    n_failed: int
    n_scored: int
    n_degenerate: int
    n_vem_nonconverged: int
    n_laplace_fallback: int
    failure_rate: float
    degenerate_rate: float
    error_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view of the accounting."""
        return {
            "n_replicates": self.n_replicates,
            "n_failed": self.n_failed,
            "n_scored": self.n_scored,
            "n_degenerate": self.n_degenerate,
            "n_vem_nonconverged": self.n_vem_nonconverged,
            "n_laplace_fallback": self.n_laplace_fallback,
            "failure_rate": self.failure_rate,
            "degenerate_rate": self.degenerate_rate,
            "error_counts": dict(self.error_counts),
        }


def failure_accounting(rows: Sequence[dict[str, Any]]) -> FailureAccounting:
    """Count failed and degenerate replicates over *all* rows read from the store.

    Degeneracy is counted among scored rows only: a replicate that crashed has
    no meaningful convergence status to report.

    Args:
        rows: Every row from ``load_results``, unfiltered.

    Returns:
        The populated ``FailureAccounting``.
    """
    n_replicates = len(rows)
    failed = [r for r in rows if row_failed(r)]
    scored = [r for r in rows if not row_failed(r)]
    n_nonconverged = sum(
        1 for r in scored if not bool((r.get("flags") or {}).get("vem_converged", True))
    )
    n_fallback = sum(
        1 for r in scored if bool((r.get("flags") or {}).get("laplace_fallback", False))
    )
    n_degenerate = sum(1 for r in scored if row_degenerate(r))

    error_counts: dict[str, int] = {}
    for row in failed:
        error = (row.get("flags") or {}).get("error") or "unspecified"
        error_counts[error] = error_counts.get(error, 0) + 1

    return FailureAccounting(
        n_replicates=n_replicates,
        n_failed=len(failed),
        n_scored=len(scored),
        n_degenerate=n_degenerate,
        n_vem_nonconverged=n_nonconverged,
        n_laplace_fallback=n_fallback,
        failure_rate=len(failed) / n_replicates if n_replicates else 0.0,
        degenerate_rate=n_degenerate / len(scored) if scored else 0.0,
        error_counts=error_counts,
    )


# ---------------- Summary ----------------


@dataclass(frozen=True, eq=False)
class SBCSummary:
    """Everything the analysis pass produces from one results store.

    Attributes:
        n_replicates: Rows read.
        n_analysed: Rows contributing to ranks and coverage.
        L: Posterior draws per replicate.
        sizes: Distinct per-replicate ``{"K", "T", "n_wallets"}`` sizes seen.
        prior: Fingerprint of the prior every analysed replicate shares; the
            summary is meaningless without it, since the prior *is* the density
            the ranks test the posterior against.
        alpha: Significance level threaded through every test and band.
        nominal: Nominal interval level.
        band: R4 acceptance window used for the coverage conclusiveness check.
        include_degenerate: Whether degenerate rows entered the statistics.
        uniformity: Per-component rank verdicts.
        coverage: The coverage table, ``theta_w`` last when present.
        coverage_power: Conclusive / underpowered accounting over ``coverage``;
            a passing table at small ``n`` proves nothing on its own.
        failures: Failure and degeneracy accounting over all rows.
        z_auc: Pooled ``{"n", "mean", "sd"}`` for the latent-Z health metric.
        theta_w: Wallet-interval block, or None when none were recorded.
        sensitivity: Coverage rates recomputed with degenerate rows included,
            or None when that would change nothing.
        flagged: Human-readable names of every check that did not pass.
    """

    n_replicates: int
    n_analysed: int
    L: int
    sizes: list[dict[str, int]]
    prior: dict[str, float]
    alpha: float
    nominal: float
    band: tuple[float, float]
    include_degenerate: bool
    uniformity: dict[str, UniformityResult]
    coverage: list[CoverageRow]
    coverage_power: dict[str, Any]
    failures: FailureAccounting
    z_auc: dict[str, Any]
    theta_w: dict[str, Any] | None
    sensitivity: dict[str, Any] | None
    flagged: list[str]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view of the whole summary."""
        return {
            "schema_version": SBC_SCHEMA_VERSION,
            "n_replicates": self.n_replicates,
            "n_analysed": self.n_analysed,
            "L": self.L,
            "sizes": self.sizes,
            "prior": self.prior,
            "alpha": self.alpha,
            "nominal_coverage": self.nominal,
            "coverage_band": list(self.band),
            "include_degenerate": self.include_degenerate,
            "coverage": [row.to_dict() for row in self.coverage],
            "coverage_power": self.coverage_power,
            "uniformity": {k: v.to_dict() for k, v in self.uniformity.items()},
            "theta_w": self.theta_w,
            "z_auc": self.z_auc,
            "failures": self.failures.to_dict(),
            "sensitivity": self.sensitivity,
            "flagged": list(self.flagged),
            "alpha_note": ALPHA_FAMILY_NOTE,
            "interpretation_key": RANK_INTERPRETATION_KEY,
        }


def _distinct_sizes(rows: Sequence[dict[str, Any]]) -> list[dict[str, int]]:
    """Distinct replicate sizes, sorted, so the summary is self-describing."""
    seen: dict[tuple[int, int, int], dict[str, int]] = {}
    for row in rows:
        size = row.get("size") or {}
        key = (
            int(size.get("K", 0)),
            int(size.get("T", 0)),
            int(size.get("n_wallets", 0)),
        )
        seen.setdefault(key, {"K": key[0], "T": key[1], "n_wallets": key[2]})
    return [seen[k] for k in sorted(seen)]


def _z_auc_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pooled mean and spread of the per-replicate Z discrimination AUC."""
    values = np.asarray(
        [r["z_auc"] for r in rows if r.get("z_auc") is not None],
        dtype=float,
    )
    if values.size == 0:
        return {"n": 0, "mean": None, "sd": None}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    }


def _coverage_power(coverage: Sequence[CoverageRow]) -> dict[str, Any]:
    """Report how much of the coverage table is conclusive rather than just passing.

    Args:
        coverage: The table as returned by ``coverage_table``.

    Returns:
        Counts per verdict, the names of the underpowered rows, the smallest
        contributing ``n``, and ``_COVERAGE_POWER_NOTE``.
    """
    underpowered = [row.name for row in coverage if row.in_range and not row.conclusive]
    return {
        "n_rows": len(coverage),
        "n_pass_conclusive": sum(1 for row in coverage if row.conclusive),
        "n_pass_underpowered": len(underpowered),
        "n_fail": sum(1 for row in coverage if not row.in_range),
        "underpowered": underpowered,
        "min_n": min((row.n for row in coverage), default=0),
        "row_alpha": coverage[0].row_alpha if coverage else None,
        "note": _COVERAGE_POWER_NOTE,
    }


def _collect_flags(
    uniformity: dict[str, UniformityResult],
    coverage: Sequence[CoverageRow],
    failures: FailureAccounting,
) -> list[str]:
    """List every check that did not pass, in report order."""
    flags: list[str] = []
    for name, result in uniformity.items():
        if result.rejected:
            flags.append(
                f"{name}: rank uniformity rejected (p={result.p_value:.3g}, "
                f"Holm-adjusted {result.p_value_adj:.3g}; {result.shape_hint})",
            )
        elif result.band_violation:
            flags.append(
                f"{name}: ECDF leaves the simultaneous band "
                f"(D={result.ks_stat:.3f} > {result.band_half_width:.3f})",
            )
    for row in coverage:
        # Only a CI that *excludes* the nominal level is evidence. Underpowered
        # rows (CI overlaps nominal but is wider than the band) are reported in
        # the summary's `coverage_power` block instead of as failures, so a
        # short run reads as "not yet conclusive" rather than "miscalibrated".
        if not row.in_range:
            flags.append(
                f"{row.name}: coverage {row.rate:.3f} (n={row.n}), "
                f"CI [{row.ci_lo:.3f}, {row.ci_hi:.3f}] excludes the nominal "
                f"{row.nominal:.2f}",
            )
    if failures.failure_rate > FAILURE_RATE_FLAG:
        flags.append(
            f"failure rate {failures.failure_rate:.1%} exceeds "
            f"{FAILURE_RATE_FLAG:.0%}",
        )
    return flags


def analyze(
    rows: Sequence[dict[str, Any]],
    *,
    components: Sequence[str] = PHI_COMPONENTS,
    n_bins: int | None = None,
    alpha: float = DEFAULT_ALPHA,
    nominal: float = NOMINAL_COVERAGE,
    band: tuple[float, float] = COVERAGE_BAND,
    include_degenerate: bool = False,
) -> SBCSummary:
    """Turn raw replicate rows into the full calibration summary.

    Args:
        rows: Every row from ``load_results``, unfiltered — the failure
            accounting needs the excluded ones.
        components: Phi components to assess, in report order.
        n_bins: Chi-square bin count; ``None`` picks ``default_n_bins``.
        alpha: Significance level for tests, bands, and CIs.
        nominal: Nominal interval level the harness recorded hits at.
        band: R4 acceptance window; a coverage row is *conclusive* when its CI
            fits inside it (the pass/fail decision is CI-vs-``nominal``).
        include_degenerate: Fold non-converged / fallback replicates into the
            headline statistics instead of only the sensitivity block.

    Returns:
        The populated ``SBCSummary``.

    Raises:
        ValueError: If no replicate is usable, the rows disagree on ``L`` or on
            the prior they were generated under, or the rows are malformed (see
            ``rank_uniformity`` and ``coverage_table``).
    """
    selected = usable_rows(rows, include_degenerate=include_degenerate)
    if not selected:
        raise ValueError(
            f"no usable replicates among {len(rows)} rows "
            f"(include_degenerate={include_degenerate}); nothing to analyse",
        )
    # Refuse a store that pools two priors before computing anything from it:
    # unlike a mixed L (which `_single_L` catches on the way out) a mixed prior
    # produces perfectly well-formed, perfectly meaningless statistics.
    single_prior = _single_prior(selected)
    uniformity = rank_uniformity(
        selected,
        components=components,
        n_bins=n_bins,
        alpha=alpha,
    )
    coverage = coverage_table(
        selected,
        components=components,
        nominal=nominal,
        band=band,
        alpha=alpha,
    )
    failures = failure_accounting(rows)

    # Conditioning the headline table on "the fit worked" is itself an
    # assumption; recomputing the rates with the degenerate replicates folded
    # back in shows how much of the answer rests on it.
    sensitivity: dict[str, Any] | None = None
    if not include_degenerate and failures.n_degenerate > 0:
        with_degenerate = usable_rows(rows, include_degenerate=True)
        rates = coverage_table(
            with_degenerate,
            components=components,
            nominal=nominal,
            band=band,
            alpha=alpha,
        )
        sensitivity = {
            "n_analysed": len(with_degenerate),
            "coverage_rates": {row.name: row.rate for row in rates},
            "note": (
                "Coverage recomputed with the "
                f"{failures.n_degenerate} degenerate replicate(s) included; "
                "large moves mean the headline table depends on excluding them."
            ),
        }

    return SBCSummary(
        n_replicates=len(rows),
        n_analysed=len(selected),
        L=_single_L(selected),
        sizes=_distinct_sizes(rows),
        prior=single_prior,
        alpha=alpha,
        nominal=nominal,
        band=band,
        include_degenerate=include_degenerate,
        uniformity=uniformity,
        coverage=coverage,
        coverage_power=_coverage_power(coverage),
        failures=failures,
        z_auc=_z_auc_block(selected),
        theta_w=_theta_block(selected),
        sensitivity=sensitivity,
        flagged=_collect_flags(uniformity, coverage, failures),
    )


def write_summary(
    summary: SBCSummary,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``summary`` to ``path`` as indented JSON, creating parent dirs.

    Args:
        summary: Summary produced by ``analyze``.
        path: Destination file, typically ``results/sbc/summary.json``.
        extra: Provenance the analysis functions cannot know (source store,
            figure paths); merged into the top level of the payload.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.to_dict()
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------- Replicate execution ----------------

# VEM fit budget per replicate. Convergence is *declared* by comparing
# `n_iter_run` against this cap, so it has to be pinned here rather than left to
# `variational_em`'s own default: a replicate that merely ran out of iterations
# must be flagged degenerate (R5), not counted as a converged fit.
#
# 200, not the 50 used elsewhere in the repo: measured on prior-predictive data
# at the default (K=3, T=400, 30 wallets) size, a replicate needs ~86 iterations
# to meet `_VEM_TOL`, and stopping at 50 leaves the log-marginal ~800 nats short
# of the optimum. Ranking the truth against the Laplace curvature at a
# *non-optimal* point is not an approximation of the posterior at all, so the
# cap must be loose enough that hitting it is the exception R5 treats it as.
_VEM_MAX_ITER = 200
_VEM_TOL = 1e-3

# Wallets whose theta_w interval is scored, taken as the *first* indices of the
# concatenated wallet axis (i.e. the first wallets of the first market — see
# `run_replicate` for the per-market wallet offsetting). A fixed rule rather
# than a random subsample: a seed-dependent selection would make the pooled
# wallet coverage depend on the draw as well as on the fit.
_THETA_SUBSAMPLE = 10


@dataclass(frozen=True)
class ReplicateSize:
    """Simulation size for one SBC replicate.

    Attributes:
        K: Markets simulated per replicate.
        T: Trades per simulated market.
        n_wallets: Wallets *per market*. Wallet ids are offset per market when
            the markets are pooled for the fit, so a replicate's fit carries
            ``K * n_wallets`` distinct wallets (see ``run_replicate``).
    """

    K: int = 3
    T: int = 400
    n_wallets: int = 30

    def to_dict(self) -> dict[str, int]:
        """Row-schema view of the size, as written to the ``size`` field."""
        return {"K": self.K, "T": self.T, "n_wallets": self.n_wallets}


def default_sbc_prior() -> PhiPrior:
    """Return the prior the SBC draws and the fits are both scored against.

    SBC is only valid when the density the data is simulated from is exactly
    the density inference assumes (plan ``2026-07-23-003`` R1/KTD5), so this
    single seam supplies the prior to the generator, to the VEM M-step, and to
    the Laplace curvature alike.

    Note:
        The shipped ``PhiPrior()`` defaults put IG(1e-9, 1e-9) on ``tau2`` — a
        deliberately vanishing placeholder that regularizes the M-step without
        perturbing it (STATUS.md P11). It cannot be *sampled*: draws span
        hundreds of orders of magnitude and overflow. Until P11 replaces it with
        a proper prior, ``run_sbc`` raises the ``params_from_prior`` ValueError
        naming P11 rather than producing meaningless replicates.

    Returns:
        The ``PhiPrior`` every SBC replicate simulates from and fits against.
    """
    return PhiPrior()


def _phi_true(params: ModelParams) -> dict[str, float]:
    """Extract the eight sampled phi components from a prior draw, by name."""
    return {name: float(getattr(params, name)) for name in PHI_COMPONENTS}


def _to_market_data(market: SyntheticMarket, *, wallet_offset: int) -> Any:
    """Convert one synthetic market to inference input, shifting its wallet ids.

    Each ``generate_market`` call draws its *own* ``theta_w`` vector over ids
    ``0 .. n_wallets - 1``, so pooling K markets under the raw ids would give one
    wallet K contradictory truths and the pooled model would no longer be the
    model the data came from — which invalidates SBC. Offsetting market ``k``'s
    ids by ``k * n_wallets`` makes every wallet appear in exactly one market,
    restoring the exact prior-predictive correspondence.

    Args:
        market: The simulated market.
        wallet_offset: Added to every wallet id of this market.

    Returns:
        A ``MarketData`` (typed loosely to keep the inference import deferred).
    """
    # Deferred for the same reason as the fitting stack in `run_replicate`:
    # `particle_gibbs` pulls the numba kernels the analysis path never needs.
    from src.inference.particle_gibbs import MarketData

    return MarketData(
        Y=market.Y,
        delta=market.delta,
        log_size_ratio=np.log(market.S / market.S_bar),
        wallet_ids=market.wallet_ids + wallet_offset,
    )


def _phi_ranks_and_hits(
    draws: np.ndarray,
    phi_true: dict[str, float],
    dims: Sequence[str],
) -> tuple[dict[str, int], dict[str, bool]]:
    """Rank each true component among the posterior draws and test its interval.

    The rank is ``#{l : phi_draw_l < phi_true}`` in ``{0, ..., L}`` (Talts et
    al. 2018) and the interval is the central ``NOMINAL_COVERAGE`` empirical
    quantile pair of the *same* draws, so a coverage miss and a rank extreme are
    two readings of one posterior sample rather than two approximations of it.

    Args:
        draws: Constrained posterior draws, shape ``(L, 8)`` in ``dims`` order.
        phi_true: The true value per component.
        dims: Component order of ``draws``' columns.

    Returns:
        ``(ranks, hits90)``, both keyed by component name.
    """
    tail = (1.0 - NOMINAL_COVERAGE) / 2.0
    index = {name: i for i, name in enumerate(dims)}
    ranks: dict[str, int] = {}
    hits: dict[str, bool] = {}
    for name in PHI_COMPONENTS:
        column = np.asarray(draws[:, index[name]], dtype=float)
        truth = phi_true[name]
        ranks[name] = int(np.count_nonzero(column < truth))
        lo, hi = np.quantile(column, [tail, 1.0 - tail])
        hits[name] = bool(lo <= truth <= hi)
    return ranks, hits


def _theta_interval_hits(
    logit_mean: np.ndarray,
    logit_var: np.ndarray,
    theta_true: np.ndarray,
) -> list[bool]:
    """Test the logit-normal wallet intervals against the simulated truths.

    The VEM M-step reports each wallet as a Gaussian on the logit scale (mode
    and inverse curvature), so the central ``NOMINAL_COVERAGE`` interval is
    ``mean +/- z * sd`` there and the truth is compared after the same
    transform — equivalent to, and better conditioned than, back-transforming
    the interval to the probability scale.

    Args:
        logit_mean: Per-wallet logit-scale posterior mode, shape ``(W,)``.
        logit_var: Per-wallet logit-scale posterior variance, shape ``(W,)``.
        theta_true: Simulated wallet propensities, shape ``(W,)``.

    Returns:
        Hit indicators for the first ``_THETA_SUBSAMPLE`` wallets.
    """
    n = min(_THETA_SUBSAMPLE, theta_true.size)
    z = float(norm.isf((1.0 - NOMINAL_COVERAGE) / 2.0))
    # A zero/negative variance means the wallet block never got a finite
    # curvature; the interval then collapses to a point and honestly misses,
    # which is what the coverage table should see rather than a NaN.
    half = z * np.sqrt(np.maximum(np.asarray(logit_var, dtype=float)[:n], 0.0))
    centre = np.asarray(logit_mean, dtype=float)[:n]
    truth = np.asarray(logit(np.asarray(theta_true, dtype=float)[:n]), dtype=float)
    return [bool(v) for v in np.abs(truth - centre) <= half]


def _pooled_z_auc(z_true: np.ndarray, z_prob: np.ndarray) -> float | None:
    """Pooled discrimination of q(Z) against the simulated insider indicators.

    Args:
        z_true: Concatenated true ``Z_t`` over all markets, shape ``(sum T_k,)``.
        z_prob: Concatenated ``q(Z_t = 1)`` from the fit, same shape.

    Returns:
        The rank-sum AUC, or None when one class is absent — an undefined AUC
        must not be pooled into the summary as the 0.5 sentinel.
    """
    from src.analysis.results import roc_auc

    n_pos = int(np.count_nonzero(z_true))
    if n_pos == 0 or n_pos == z_true.size:
        return None
    return float(roc_auc(z_true, z_prob))


def _replicate_row(
    *,
    seed: int,
    phi_true: dict[str, float],
    L: int,
    size: ReplicateSize,
    prior: PhiPrior,
    elapsed_s: float,
    ranks: dict[str, int] | None = None,
    hits90: dict[str, bool] | None = None,
    theta_hits90: list[bool] | None = None,
    z_auc: float | None = None,
    vem_converged: bool = False,
    laplace_fallback: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    """Assemble one schema-v2 result row.

    Both flags are written explicitly on *every* row: ``row_degenerate`` reads a
    missing ``vem_converged`` as True, so an omitted key would silently
    understate the degeneracy rate the analysis exists to report. ``L``, ``size``
    and ``prior`` are likewise written on every row, failed ones included: they
    are the regime stamp ``run_sbc`` and ``analyze`` compare against, and a row
    without it could be appended to any store at all.

    Args:
        seed: Replicate seed; the store's unique key.
        phi_true: The prior draw, by component; empty if the draw itself failed.
        L: Posterior draws ranked against the truth.
        size: Simulation size of this replicate.
        prior: Prior this replicate simulated from and fitted against.
        elapsed_s: Wall-clock seconds for the whole replicate.
        ranks: Per-component ranks, or None for a failed replicate.
        hits90: Per-component interval hits, or None for a failed replicate.
        theta_hits90: Wallet interval hits, or None for a failed replicate.
        z_auc: Pooled Z AUC, or None when undefined or failed.
        vem_converged: Whether the EM stopped on its tolerance, not the cap.
        laplace_fallback: Whether any Laplace block used the R3 fallback ladder.
        error: ``repr`` of the exception when the replicate failed, else None.

    Returns:
        The JSON-serializable row.
    """
    return {
        "schema_version": SBC_SCHEMA_VERSION,
        "seed": int(seed),
        "phi_true": phi_true,
        "L": int(L),
        "ranks": ranks,
        "hits90": hits90,
        "theta_hits90": theta_hits90,
        "z_auc": z_auc,
        "flags": {
            "vem_converged": bool(vem_converged),
            "laplace_fallback": bool(laplace_fallback),
            "failed": error is not None,
            "error": error,
        },
        "elapsed_s": float(elapsed_s),
        "size": size.to_dict(),
        "prior": prior_fingerprint(prior),
    }


def run_replicate(
    sim_seed: int,
    size: ReplicateSize,
    prior: PhiPrior,
    L: int,
) -> dict[str, Any]:
    """Run one SBC replicate: prior draw, simulate, fit, and score.

    The pipeline is ``phi ~ prior`` -> prior-predictive dataset -> VEM (fit
    single-threaded; the parallelism is over replicates, never inside one) ->
    Laplace posterior -> ``L`` i.i.d. draws -> ranks, 90% interval hits,
    wallet-interval hits, and pooled Z AUC. ``prior`` is threaded into the
    generator *and* the M-step *and* the curvature, which is the condition that
    makes the rank statistic uniform under a correct posterior.

    Everything is seeded off ``sim_seed`` alone through one generator, so a
    replicate is reproducible and independent of how many workers ran it or in
    what order — the property ``--resume`` and ``--n-jobs`` both rely on.

    Any exception is captured into a failed row rather than propagated: one bad
    draw (an extreme untruncated Cauchy ``beta``, a singular fit) must not kill
    a 200-replicate run, and R5 requires the failure be counted, not dropped.

    Args:
        sim_seed: Seed for this replicate; also its key in the results store.
        size: Simulation size (K markets, T trades, wallets per market).
        prior: The prior to simulate from and to fit against.
        L: Posterior draws to rank the truth against (i.i.d., no thinning).

    Returns:
        One schema-v1 row, always — including on failure.
    """
    # Deferred imports (CODE_QUALITY §4 rule 6): the fitting stack pulls the
    # numba-backed PG kernels in transitively and costs seconds to import, while
    # the analysis half of this module and `scripts/sbc.py --analyze` only ever
    # read JSON. Keeping them function-local also lets tests monkeypatch the fit.
    from src.inference.laplace import laplace_from_vem
    from src.inference.variational_em import variational_em

    started = time.perf_counter()
    rng = np.random.default_rng(sim_seed)
    phi_true: dict[str, float] = {}
    try:
        params = params_from_prior(prior, rng)
        phi_true = _phi_true(params)
        simulated = [
            generate_prior_predictive_market(
                params,
                rng=rng,
                n_trades=size.T,
                n_wallets=size.n_wallets,
            )
            for _ in range(size.K)
        ]
        markets = [
            _to_market_data(market, wallet_offset=k * size.n_wallets)
            for k, market in enumerate(simulated)
        ]
        theta_true = np.concatenate([m.theta_w for m in simulated])

        vem = variational_em(
            markets,
            InferenceConfig(),
            n_wallets=size.K * size.n_wallets,
            n_iter=_VEM_MAX_ITER,
            tol=_VEM_TOL,
            n_jobs=1,
            prior=prior,
            # SBC ranks beta_S and beta_Z, so they must actually be estimated:
            # with the default `estimate_betas=False` the beta block carries no
            # Fisher information, the Laplace layer substitutes the Cauchy prior
            # curvature, and every replicate would report a fallback and rank
            # the truth against the prior instead of the posterior.
            estimate_betas=True,
        )
        posterior = laplace_from_vem(vem, markets, prior)
        draws = posterior.sample(rng, L)
        ranks, hits90 = _phi_ranks_and_hits(draws, phi_true, posterior.dims)
        row = _replicate_row(
            seed=sim_seed,
            phi_true=phi_true,
            L=L,
            size=size,
            prior=prior,
            elapsed_s=time.perf_counter() - started,
            ranks=ranks,
            hits90=hits90,
            theta_hits90=_theta_interval_hits(
                vem.theta_w_logit_mean,
                vem.theta_w_logit_var,
                theta_true,
            ),
            z_auc=_pooled_z_auc(
                np.concatenate([m.Z for m in simulated]),
                np.concatenate(vem.Z_prob),
            ),
            # The EM loop breaks out on its tolerance check, so a run that used
            # the full budget never met it.
            vem_converged=vem.n_iter_run < _VEM_MAX_ITER,
            laplace_fallback=posterior.curvature_fallback,
        )
    except Exception as exc:  # noqa: BLE001 - a failed replicate is data, not a crash
        log.warning("replicate seed=%d failed: %r", sim_seed, exc)
        row = _replicate_row(
            seed=sim_seed,
            phi_true=phi_true,
            L=L,
            size=size,
            prior=prior,
            elapsed_s=time.perf_counter() - started,
            error=repr(exc),
        )
    return row


def _prior_conflict(
    stored: Sequence[dict[str, float]],
    wanted: dict[str, float],
) -> str:
    """Spell out how a store's prior fingerprints differ from the current one.

    Args:
        stored: Distinct fingerprints found in the store.
        wanted: Fingerprint of the prior this run would use.

    Returns:
        A one-line description naming each differing hyperparameter (union of
        both key sets, so a fingerprint written by a different ``PhiPrior``
        shape is reported rather than silently matching).
    """
    parts: list[str] = []
    for fingerprint in stored:
        keys = sorted(set(fingerprint) | set(wanted))
        diffs = [
            f"{k}={fingerprint.get(k)!r} in store vs {wanted.get(k)!r} in this run"
            for k in keys
            if fingerprint.get(k) != wanted.get(k)
        ]
        parts.append(", ".join(diffs) if diffs else "identical")
    return "prior: " + "; ".join(parts)


def _regime_conflicts(
    rows: Sequence[dict[str, Any]],
    *,
    size: ReplicateSize,
    prior: PhiPrior,
    L: int,
) -> list[str]:
    """Name every way an existing store's regime differs from this run's.

    Ranks pool only within one regime: ``L`` fixes the rank support, ``size``
    fixes how much data each posterior saw, and ``prior`` *is* the density SBC
    tests the posterior against. Skipping completed seeds therefore is not
    enough for a resume to be sound — the rows already on disk must have come
    from the same regime, or the finished store is two half-runs of two
    different experiments and no statistic over it means anything.

    Args:
        rows: Rows already in the store, as returned by ``load_results``.
        size: Simulation size this run would use.
        prior: Prior this run would use.
        L: Posterior draws per replicate this run would use.

    Returns:
        One line per conflicting field, empty when the store matches.
    """
    conflicts: list[str] = []
    stored_L = sorted({int(row["L"]) for row in rows})
    if stored_L != [int(L)]:
        conflicts.append(f"L: store has {stored_L}, this run uses {int(L)}")
    stored_sizes = _distinct_sizes(rows)
    if stored_sizes != [size.to_dict()]:
        conflicts.append(
            f"size: store has {stored_sizes}, this run uses {size.to_dict()}",
        )
    wanted = prior_fingerprint(prior)
    stored_priors = _distinct_priors(rows)
    if stored_priors != [wanted]:
        conflicts.append(_prior_conflict(stored_priors, wanted))
    return conflicts


def completed_seeds(path: str | Path) -> set[int]:
    """Read back the seeds a results store already holds.

    Args:
        path: The append-only JSONL store; a missing file is not an error.

    Returns:
        The seeds present, empty when the file does not exist.

    Raises:
        ValueError: If the store is malformed (see ``load_results``) — resuming
            onto a file this module cannot read would double-count replicates.
    """
    path = Path(path)
    if not path.exists():
        return set()
    return {int(row["seed"]) for row in load_results(path)}


def run_sbc(
    seeds: Sequence[int],
    *,
    size: ReplicateSize,
    prior: PhiPrior,
    L: int,
    out_path: str | Path,
    n_jobs: int = 1,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run replicates over ``seeds``, appending one row each to a JSONL store.

    Rows are written by *this* process as workers return them, never by the
    workers themselves — concurrent appends to one file interleave partial
    lines — and flushed per row, so a killed run loses at most the in-flight
    replicates and ``resume`` picks up the rest.

    Args:
        seeds: Replicate seeds; each is an independent, reproducible stream.
        size: Simulation size passed to every replicate.
        prior: Prior to simulate from and fit against (one object, R1/KTD5).
        L: Posterior draws per replicate; must match any rows already in the
            store, since the analysis bins ranks at a single L.
        out_path: JSONL store to append to; parent directories are created.
        n_jobs: joblib workers over replicates. Each replicate fits
            single-threaded, so this is the only level of parallelism.
        resume: Skip seeds already present in ``out_path``.

    Returns:
        The rows computed by *this* call, in seed order; already-present seeds
        are not re-read.

    Raises:
        ValueError: If ``prior`` cannot be sampled (the P11 improper-tau2
            default), or if ``out_path`` already holds rows from a different
            ``(L, size, prior)`` regime. Both are raised before any replicate
            runs.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fail fast on an unsamplable prior: `run_replicate` would otherwise capture
    # the same error once per seed and burn the whole run writing failed rows.
    params_from_prior(prior, np.random.default_rng(0))

    # Fail fast on a regime clash, before a single replicate is dispatched.
    # Checked whenever the store already holds rows, not only under `resume`:
    # appending a second regime is exactly as wrong when the seeds happen not to
    # overlap, and the resulting store cannot be analysed either way.
    stored = load_results(out_path) if out_path.exists() else []
    conflicts = _regime_conflicts(stored, size=size, prior=prior, L=L) if stored else []
    if conflicts:
        raise ValueError(
            f"{out_path} was written under a different SBC regime: "
            + "; ".join(conflicts)
            + ". Ranks pool only within one (L, size, prior) regime - write this "
            "run to a new store instead.",
        )

    pending = list(seeds)
    if resume:
        done = {int(row["seed"]) for row in stored}
        pending = [s for s in pending if s not in done]
        log.info(
            "resume: %d seed(s) already in %s, %d to run",
            len(done),
            out_path,
            len(pending),
        )
    if not pending:
        log.info("nothing to run")
        return []

    tasks = (delayed(run_replicate)(seed, size, prior, L) for seed in pending)
    rows: list[dict[str, Any]] = []
    # `return_as="generator"` streams results back in submission order as they
    # complete, which is what lets the store grow incrementally instead of only
    # at the end of the run.
    with out_path.open("a", encoding="utf-8") as handle:
        for row in Parallel(n_jobs=n_jobs, return_as="generator")(tasks):
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.append(row)
            log.info(
                "seed=%d %s in %.1fs (%d/%d)",
                row["seed"],
                "FAILED" if row["flags"]["failed"] else "ok",
                row["elapsed_s"],
                len(rows),
                len(pending),
            )
    return rows
