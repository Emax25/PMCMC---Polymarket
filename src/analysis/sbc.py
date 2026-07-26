"""Simulation-based calibration (SBC) and interval coverage for VEM + Laplace.

This module hosts **both halves** of the SBC pipeline:

  * *Replicate execution* — draw ``phi`` from the prior, simulate a
    prior-predictive dataset, fit, and emit one result row. That half lands
    with the harness unit; the row schema it must produce is pinned by
    ``SBC_SCHEMA_VERSION`` and spelled out below so the two halves can be
    built independently.
  * *Analysis* (this file today) — read the accumulated JSONL back and turn it
    into the calibration evidence: per-component rank uniformity, nominal-90%
    interval coverage, failure accounting, and a JSON summary.

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

Two numeric verdicts back those shapes so nothing depends on eyeballing a plot:
a chi-square bin test (the decision, ``p_value``/``rejected``) and a
Dvoretzky-Kiefer-Wolfowitz simultaneous ECDF band (``band_violation``, and the
band drawn on the figure). A signed pair of z-scores (``bias_z``,
``dispersion_z``) names *which* of the three shapes dominates.

Both verdicts are corrected for testing all eight phi components at once (Holm
for the p-values, Bonferroni for the bands), each at family-wise ``alpha``.
They are not corrected *against each other*, so a perfectly calibrated run
raises some flag with probability roughly ``2 * alpha`` — about one run in ten
at the default 0.05. A lone flag on one component is weak evidence; the failure
this analysis exists to catch shows up across components and as a coverage
miss too.

Degenerate and failed replicates are never silently dropped (R5): failed rows
carry no ranks and are excluded from every statistic but counted in the failure
rate; degenerate rows (VEM non-convergence or a Laplace curvature fallback) are
excluded by default and reported both as a rate and as a sensitivity block that
re-runs coverage with them included.

Row schema (``schema_version = 1``), one JSON object per line::

    {"schema_version": 1, "seed": int, "phi_true": {component: float},
     "L": int, "ranks": {component: int in 0..L} | null,
     "hits90": {component: bool} | null, "theta_hits90": [bool, ...] | null,
     "z_auc": float | null,
     "flags": {"vem_converged": bool, "laplace_fallback": bool,
               "failed": bool, "error": str | null},
     "elapsed_s": float, "size": {"K": int, "T": int, "n_wallets": int}}
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import binom, chi2, norm

log = logging.getLogger(__name__)

SBC_SCHEMA_VERSION = 1

# Canonical phi order. Deliberately restated rather than imported from
# `src.inference.diagnostics.PHI_PARAM_NAMES` or `src.inference.laplace`: both
# pull the PG/iPMCMC inference stack in transitively, and the analysis path here
# only ever reads JSON. `tests/test_sbc.py` asserts this tuple still equals
# `PHI_PARAM_NAMES`, so the duplication cannot drift unnoticed.
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
    "staying inside it is the calibrated case."
)

_THETA_DEPENDENCE_NOTE = (
    "Wallet intervals are pooled across replicates; wallets within a replicate "
    "share one fit, so this binomial CI ignores that dependence and is "
    "anti-conservative. Read the width as a lower bound."
)


# ---------------- Row I/O and selection ----------------


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
        """Whether either numeric verdict says the ranks are non-uniform."""
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

    Args:
        n: Replicate count contributing ranks.
        L: Posterior draws per replicate.

    Returns:
        Bin count, at least 2.
    """
    return max(2, min(MAX_RANK_BINS, n // MIN_EXPECTED_PER_BIN, L + 1))


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
    """One line of the coverage table.

    Attributes:
        name: Component name, or ``theta_w`` for the pooled latent row.
        n: Interval draws contributing (replicates, or wallet intervals).
        n_hits: How many contained the truth.
        rate: Empirical coverage ``n_hits / n``.
        ci_lo: Wilson lower bound at the table's alpha.
        ci_hi: Wilson upper bound.
        nominal: Nominal interval level the hits were computed at.
        in_range: Whether ``rate`` lies inside the R4 acceptance band.
        note: Caveat attached to this row, empty when there is none.
    """

    name: str
    n: int
    n_hits: int
    rate: float
    ci_lo: float
    ci_hi: float
    nominal: float
    in_range: bool
    note: str = ""

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
            "in_range": self.in_range,
            "note": self.note,
        }


def _coverage_row(
    name: str,
    hits: Sequence[bool],
    *,
    nominal: float,
    band: tuple[float, float],
    alpha: float,
    note: str = "",
) -> CoverageRow:
    """Build one coverage row from a flat sequence of hit indicators."""
    n = len(hits)
    n_hits = int(sum(bool(h) for h in hits))
    rate = n_hits / n if n else 0.0
    lo, hi = wilson_interval(n_hits, n, alpha=alpha)
    return CoverageRow(
        name=name,
        n=n,
        n_hits=n_hits,
        rate=rate,
        ci_lo=lo,
        ci_hi=hi,
        nominal=nominal,
        in_range=bool(n) and band[0] <= rate <= band[1],
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
    recorded wallet intervals. Rows outside ``band`` are flagged via
    ``CoverageRow.in_range``; nothing is dropped.

    Args:
        rows: Usable replicate rows, each carrying a full ``hits90`` mapping.
        components: Phi components to tabulate, in report order.
        nominal: Nominal interval level the harness recorded hits at.
        band: Acceptance window for the empirical rate (R4).
        alpha: Two-sided level for the Wilson CI.

    Returns:
        Coverage rows in ``components`` order, ``theta_w`` last when present.

    Raises:
        ValueError: If a row is missing a component's hit indicator.
    """
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
                alpha=alpha,
            ),
        )

    per_replicate = _theta_hits(rows)
    if per_replicate:
        pooled = [h for replicate in per_replicate for h in replicate]
        table.append(
            _coverage_row(
                THETA_W_KEY,
                pooled,
                nominal=nominal,
                band=band,
                alpha=alpha,
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
        alpha: Significance level threaded through every test and band.
        nominal: Nominal interval level.
        band: R4 acceptance window for the empirical coverage.
        include_degenerate: Whether degenerate rows entered the statistics.
        uniformity: Per-component rank verdicts.
        coverage: The coverage table, ``theta_w`` last when present.
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
    alpha: float
    nominal: float
    band: tuple[float, float]
    include_degenerate: bool
    uniformity: dict[str, UniformityResult]
    coverage: list[CoverageRow]
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
            "alpha": self.alpha,
            "nominal_coverage": self.nominal,
            "coverage_band": list(self.band),
            "include_degenerate": self.include_degenerate,
            "coverage": [row.to_dict() for row in self.coverage],
            "uniformity": {k: v.to_dict() for k, v in self.uniformity.items()},
            "theta_w": self.theta_w,
            "z_auc": self.z_auc,
            "failures": self.failures.to_dict(),
            "sensitivity": self.sensitivity,
            "flagged": list(self.flagged),
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
        if not row.in_range:
            flags.append(
                f"{row.name}: coverage {row.rate:.3f} outside the acceptance band",
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
        band: R4 acceptance window for empirical coverage.
        include_degenerate: Fold non-converged / fallback replicates into the
            headline statistics instead of only the sensitivity block.

    Returns:
        The populated ``SBCSummary``.

    Raises:
        ValueError: If no replicate is usable, or the rows are malformed (see
            ``rank_uniformity`` and ``coverage_table``).
    """
    selected = usable_rows(rows, include_degenerate=include_degenerate)
    if not selected:
        raise ValueError(
            f"no usable replicates among {len(rows)} rows "
            f"(include_degenerate={include_degenerate}); nothing to analyse",
        )
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
        alpha=alpha,
        nominal=nominal,
        band=band,
        include_degenerate=include_degenerate,
        uniformity=uniformity,
        coverage=coverage,
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
