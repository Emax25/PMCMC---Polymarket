"""CLI: simulation-based calibration and interval coverage for the VEM fast path.

The command has two modes over one append-only JSONL store
(``results/sbc/sbc_results.jsonl``):

  * **run** (default) — the replicate harness: draw ``phi`` from the prior,
    simulate a prior-predictive dataset, fit VEM + Laplace, and append one row
    per replicate. Resumable via ``--resume``, parallel via ``--n-jobs``. *Not
    yet implemented — its flags are declared here so the analysis mode and the
    harness share one argument surface.*
  * **analyze** (``--analyze``) — read the store back and emit the calibration
    evidence: per-component rank-uniformity verdicts, the nominal-90% coverage
    table, failure accounting, a JSON summary, and the paper figures.

Splitting the two at the JSONL boundary is what makes a killed 200-replicate run
cheap to recover and free to re-analyse (plan ``2026-07-23-003`` KTD4).

All statistics live in ``src/analysis/sbc.py`` and all drawing in
``src/analysis/plots.py``; this file only wires arguments, orchestration,
reporting and I/O.

Reading the output: the two verdicts per component are a chi-square bin test
(``p``) and a simultaneous ECDF band, and coverage must land inside
[0.85, 0.95] for nominal-90% intervals. A U-shaped rank histogram means the
posterior is too narrow (overconfident) — the failure mode this whole exercise
exists to detect. Read the ``FLAGGED`` block and the failure rate before
quoting any single number.

Examples:
    # Analyse a completed run into figures + summary.json
    python -m scripts.sbc --analyze

    # Analyse a specific store into a scratch directory
    python -m scripts.sbc --analyze --in results/sbc/gate.jsonl \\
        --fig-dir /tmp/figs --summary /tmp/summary.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from src.analysis.plots import figure_sbc_ranks, save_paper_figure, set_paper_style
from src.analysis.sbc import (
    DEFAULT_ALPHA,
    RANK_INTERPRETATION_KEY,
    SBCSummary,
    analyze,
    load_results,
    write_summary,
)

log = logging.getLogger("sbc")

DEFAULT_RESULTS_PATH = Path("results/sbc/sbc_results.jsonl")
DEFAULT_FIG_DIR = Path("results/figures/sbc")
DEFAULT_SUMMARY_PATH = Path("results/sbc/summary.json")

# Posterior draws per replicate. i.i.d. Gaussian draws from the Laplace
# posterior, so no thinning: Talts-style thinning decorrelates MCMC draws and
# would only coarsen the rank statistic here.
DEFAULT_POSTERIOR_DRAWS = 999


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the sbc argument parser covering both the run and analyze modes."""
    p = argparse.ArgumentParser(
        description=(
            "Simulation-based calibration and interval coverage for the "
            "VEM + Laplace pipeline: run replicates into a JSONL store, then "
            "analyse that store into rank-uniformity figures, a coverage "
            "table, and a JSON summary."
        ),
    )
    p.add_argument(
        "--analyze",
        action="store_true",
        help="Analyse an existing results store instead of running replicates.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level name (default: INFO).",
    )

    run = p.add_argument_group("run mode")
    run.add_argument(
        "--n-sims",
        type=int,
        default=200,
        help="Replicates to run (default: 200; the plan's minimum for SBC).",
    )
    run.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="joblib workers over replicates (default: 1).",
    )
    run.add_argument(
        "--sim-K",
        type=int,
        default=3,
        help="Markets per simulated replicate (default: 3).",
    )
    run.add_argument(
        "--sim-T",
        type=int,
        default=400,
        help="Trades per simulated market (default: 400).",
    )
    run.add_argument(
        "--sim-wallets",
        type=int,
        default=30,
        help="Wallets per simulated replicate (default: 30).",
    )
    run.add_argument(
        "--posterior-draws",
        type=int,
        default=DEFAULT_POSTERIOR_DRAWS,
        help=(
            "L, the i.i.d. posterior draws ranked against the truth "
            f"(default: {DEFAULT_POSTERIOR_DRAWS}; no thinning)."
        ),
    )
    run.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="First replicate seed; replicate i uses seed-base + i (default: 0).",
    )
    run.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"JSONL results store to append to (default: {DEFAULT_RESULTS_PATH}).",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="Skip seeds already present in --out instead of re-running them.",
    )

    analyse = p.add_argument_group("analyze mode")
    analyse.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"JSONL results store to analyse (default: {DEFAULT_RESULTS_PATH}).",
    )
    analyse.add_argument(
        "--fig-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help=f"Directory for the rank figures (default: {DEFAULT_FIG_DIR}).",
    )
    analyse.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"JSON summary path (default: {DEFAULT_SUMMARY_PATH}).",
    )
    analyse.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="Rank-histogram bins for the chi-square test (default: automatic).",
    )
    analyse.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=(
            "Significance level for the uniformity test, the bands, and the "
            f"coverage CIs (default: {DEFAULT_ALPHA})."
        ),
    )
    analyse.add_argument(
        "--include-degenerate",
        action="store_true",
        help=(
            "Fold non-converged / Laplace-fallback replicates into the headline "
            "statistics. Off by default; they are always counted and always "
            "reported as a sensitivity block."
        ),
    )
    return p.parse_args(argv)


# ---------------- Figures and report ----------------


def _write_figures(summary: SBCSummary, *, fig_dir: Path) -> list[str]:
    """Render and save the ECDF-difference and rank-histogram figures."""
    set_paper_style()
    paths: list[Path] = []
    for kind, name in (("ecdf", "sbc_rank_ecdf"), ("hist", "sbc_rank_hist")):
        fig = figure_sbc_ranks(summary.uniformity, kind=kind)
        paths.extend(save_paper_figure(fig, name, directory=fig_dir))
        plt.close(fig)
    return [str(p) for p in paths]


def _format_report(summary: SBCSummary) -> str:
    """Build a human-readable summary of the calibration analysis.

    Flagged checks and the failure rate print *before* the per-component tables,
    and nowhere else, so a reader cannot quote a coverage number without first
    meeting the caveat that qualifies it.
    """
    failures = summary.failures
    sizes = ", ".join(
        f"K={s['K']} T={s['T']} wallets={s['n_wallets']}" for s in summary.sizes
    )
    lines = [
        "=== SBC and coverage analysis ===",
        f"Replicates: {summary.n_replicates} read, {summary.n_analysed} analysed, "
        f"L={summary.L} posterior draws  (alpha={summary.alpha:g})",
        f"Sizes: {sizes}",
    ]
    if summary.flagged:
        lines.append("")
        lines.append("!!! FLAGGED !!!")
        lines.extend(f"  {text}" for text in summary.flagged)
        lines.append(
            "  (the uniformity test and the ECDF band are each controlled at "
            f"alpha={summary.alpha:g} across components but not against each "
            "other, so a calibrated run raises some flag about "
            f"{2.0 * summary.alpha:.0%} of the time; a single isolated flag is "
            "weak evidence)",
        )

    lines.extend(
        [
            "",
            f"Failures: {failures.n_failed}/{failures.n_replicates} "
            f"({failures.failure_rate:.1%})   degenerate "
            f"{failures.n_degenerate}/{failures.n_scored} "
            f"({failures.degenerate_rate:.1%}: "
            f"vem_nonconverged={failures.n_vem_nonconverged}, "
            f"laplace_fallback={failures.n_laplace_fallback})",
        ],
    )
    for error, count in sorted(failures.error_counts.items()):
        lines.append(f"    {count} x {error}")

    lines.extend(
        [
            "",
            f"Rank uniformity (chi-square bin test, family-wise "
            f"alpha={summary.alpha:g} over {len(summary.uniformity)} components; "
            "p_adj is Holm-adjusted and drives the decision):",
        ],
    )
    for name, result in summary.uniformity.items():
        lines.append(
            f"  {name:9s} chi2={result.chi2_stat:8.3f} dof={result.dof:3d} "
            f"p={result.p_value:.4f} p_adj={result.p_value_adj:.4f}  "
            f"rejected={str(result.rejected):5s} "
            f"band_violation={str(result.band_violation):5s}  {result.shape_hint}",
        )

    lines.extend(
        [
            "",
            f"Coverage of nominal-{summary.nominal:.0%} intervals "
            f"(target [{summary.band[0]:.2f}, {summary.band[1]:.2f}], "
            f"Wilson {1.0 - summary.alpha:.0%} CI):",
        ],
    )
    for row in summary.coverage:
        lines.append(
            f"  {row.name:9s} {row.rate:.3f} [{row.ci_lo:.3f}, {row.ci_hi:.3f}] "
            f"n={row.n:<6d} in_range={row.in_range}",
        )
    if summary.theta_w is not None:
        lines.append(
            f"    theta_w pooled over {summary.theta_w['n_replicates']} replicates "
            f"({summary.theta_w['n_wallet_intervals']} wallet intervals); "
            f"between-replicate sd "
            f"{summary.theta_w['between_replicate_sd']:.3f}",
        )
        lines.append(f"    caveat: {summary.theta_w['note']}")

    z_auc = summary.z_auc
    if z_auc["n"]:
        lines.extend(
            [
                "",
                f"Pooled Z AUC: mean={z_auc['mean']:.4f} sd={z_auc['sd']:.4f} "
                f"(n={z_auc['n']})",
            ],
        )
    if summary.sensitivity is not None:
        lines.extend(["", "Sensitivity (degenerate replicates included):"])
        for name, rate in summary.sensitivity["coverage_rates"].items():
            lines.append(f"  {name:9s} {rate:.3f}")
    lines.extend(["", RANK_INTERPRETATION_KEY])
    return "\n".join(lines)


# ---------------- Entrypoint ----------------


def _run_analysis(args: argparse.Namespace) -> int:
    """Analyse an existing results store into a report, figures, and summary JSON."""
    rows = load_results(args.in_path)
    log.info("read %d replicate row(s) from %s", len(rows), args.in_path)
    summary = analyze(
        rows,
        n_bins=args.n_bins,
        alpha=args.alpha,
        include_degenerate=args.include_degenerate,
    )
    figures = _write_figures(summary, fig_dir=args.fig_dir)
    for path in figures:
        log.info("wrote %s", path)

    print(_format_report(summary))

    write_summary(
        summary,
        args.summary,
        extra={"source": str(args.in_path), "figures": figures},
    )
    log.info("wrote %s", args.summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the SBC harness, or analyse an existing results store.

    Args:
        argv: Argument list passed to argparse; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code 0 on success.

    Raises:
        NotImplementedError: In run mode — the replicate harness lands in a
            follow-up unit; its flags are declared so the surface is stable.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.analyze:
        raise NotImplementedError("harness lands in U2")
    return _run_analysis(args)


if __name__ == "__main__":
    sys.exit(main())
