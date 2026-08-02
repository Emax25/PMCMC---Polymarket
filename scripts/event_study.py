"""CLI: no-lookahead event study — does elevated ``P(Z)`` precede the terminal move?

Thin CLI over `src.analysis.event_study`, which owns the statistic, the
permutation schemes, the exclusion policy and the figure. This file wires
arguments, reporting and I/O only.

Two modes:

  * **study** (default) — read a `score_stream.py --replay` scores JSONL plus
    the markets' resolution metadata, run the pre-registered test per market,
    and emit a report, a JSON summary and the paper figure.
  * **calibrate** (``--calibrate``) — re-run the synthetic calibration that
    fixed the locked window (KTD4). Nothing about a real-data run depends on
    it; it exists so the locked ``W`` is auditable rather than asserted.

Reading the output. ``p_value`` is *the* result: mean ``P(Z)`` elevation over
``[t_close - W, t_close - w)`` against a within-market time-shifted-window
permutation null. ``p_max`` and ``p_cross`` are **labelled robustness checks**
— they reuse the same scores and window, so agreement is not a second piece of
evidence and disagreement only says the primary verdict is fragile. There is no
Kendall-tau criterion anywhere in this command, by design.

Replay-mode scores only. The no-lookahead property is inherited from
`score_stream.py --replay`, whose per-trade state is a function of trades
``0..t``; a scores file without a ``mode == "replay"`` provenance sidecar is
refused rather than analysed with a caveat.

Examples:
    # The real-data run: locked window, 999 permutations
    python -m scripts.event_study --scores results/streaming/scores.jsonl \\
        --resolutions data/processed --json-out results/event_study/summary.json

    # Reproduce the synthetic calibration that locked W
    python -m scripts.event_study --calibrate --n-replicates 60 --seed 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.analysis.event_study import (
    DAY_SECONDS,
    DEFAULT_ALPHA,
    DEFAULT_N_PERMUTATIONS,
    LOCKED_EMBARGO_S,
    LOCKED_WINDOW_S,
    EventStudySummary,
    ProvenanceError,
    WindowSpec,
    calibrate_window,
    load_resolutions,
    load_scores,
    read_replay_provenance,
    run_event_study,
    save_figures,
    write_summary,
)

log = logging.getLogger("event_study")

DEFAULT_FIG_DIR = Path("results/figures/event_study")
DEFAULT_SUMMARY_PATH = Path("results/event_study/summary.json")
# Calibrate mode gets its own default so a `--calibrate` run cannot overwrite a
# study summary with a window table: the two payloads have different schemas,
# and the study summary is the artifact a paper claim is read off.
DEFAULT_CALIBRATION_PATH = Path("results/event_study/window_calibration.json")

# Candidate windows the --calibrate grid sweeps, in days. Spans both sides of
# the locked 5 d so the plateau the lock sits in is visible in the output rather
# than taken on trust.
CALIBRATION_GRID_DAYS = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the event_study argument parser covering both modes."""
    p = argparse.ArgumentParser(
        description=(
            "No-lookahead event study over replayed per-trade insider scores: "
            "mean P(Z) elevation in the pre-close window against a "
            "within-market time-shifted-window permutation null."
        ),
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Re-run the synthetic window calibration instead of a study.",
    )
    p.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Scores JSONL from `score_stream.py --replay`. Its "
        "<scores>.meta.json sidecar must record mode=replay; live-mode or "
        "provenance-less scores are refused.",
    )
    p.add_argument(
        "--resolutions",
        type=Path,
        default=None,
        help="Market resolution metadata: a directory of *.meta.json sidecars, "
        "or a JSON object/array of records carrying close_time / end_date / "
        "close_ts. Markets missing from it are excluded and counted.",
    )
    p.add_argument(
        "--window",
        type=float,
        default=LOCKED_WINDOW_S / DAY_SECONDS,
        help="W in days, the event-window length before close (default: the "
        f"KTD4-locked {LOCKED_WINDOW_S / DAY_SECONDS:g}). Any other value is "
        "exploratory and is labelled as such in the summary.",
    )
    p.add_argument(
        "--embargo",
        type=float,
        default=LOCKED_EMBARGO_S / DAY_SECONDS,
        help="w in days: the window ends this far before close, and the "
        "terminal move is measured over the gap (default: the locked "
        f"{LOCKED_EMBARGO_S / DAY_SECONDS:g}).",
    )
    p.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help=f"Permutation draws per market (default: {DEFAULT_N_PERMUTATIONS}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Root RNG seed; a run replays exactly from it (default: 0).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Level for the significance count (default: {DEFAULT_ALPHA}).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help=f"JSON output path (default: {DEFAULT_SUMMARY_PATH} in study mode, "
        f"{DEFAULT_CALIBRATION_PATH} under --calibrate).",
    )
    p.add_argument(
        "--fig-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help=f"Directory for the event-study figure (default: {DEFAULT_FIG_DIR}).",
    )
    p.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip the figure; the report and summary JSON are unaffected.",
    )
    p.add_argument(
        "--n-replicates",
        type=int,
        default=20,
        help="Calibrate mode: replicates per arm (default: 20; the locked "
        "window was fixed at 60).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


# ---------------- Reports ----------------


def _format_report(summary: EventStudySummary) -> str:
    """Build the human-readable event-study report.

    The primary column leads and the robustness columns are explicitly labelled,
    so a reader cannot pick whichever of the three p-values is smallest and
    quote it as the result.
    """
    window = summary.window
    lines = [
        "=== No-lookahead event study ===",
        f"Window: W={window.W / DAY_SECONDS:g} d, embargo w="
        f"{window.w / DAY_SECONDS:g} d"
        + ("  (KTD4-locked)" if window.is_locked else "  (NOT LOCKED - EXPLORATORY)"),
        f"Markets: {summary.n_markets} scored, {len(summary.results)} analysed, "
        f"{len(summary.excluded)} excluded   "
        f"({summary.n_permutations} permutations, seed {summary.seed})",
    ]
    if summary.provenance:
        lines.append(
            f"Provenance: mode={summary.provenance.get('mode')!r} "
            f"input={summary.provenance.get('input')!r}",
        )
    if summary.exclusion_counts:
        lines.append("")
        lines.append("Excluded:")
        for reason, count in sorted(summary.exclusion_counts.items()):
            lines.append(f"  {count:3d} x {reason}")
        for row in summary.excluded:
            lines.append(f"      {row.market}: {row.reason}")

    lines.extend(
        [
            "",
            "Per market (PRIMARY = elevation + p; the two right-hand columns "
            "are labelled robustness checks, never confirmation):",
            f"  {'market':<24} {'n_win':>6} {'elev':>8} {'z':>7} {'p':>8}  |  "
            f"{'p_max':>8} {'p_cross':>8} {'move':>8}",
        ],
    )
    for row in summary.results:
        cross = "     n/a" if row.p_value_cross is None else f"{row.p_value_cross:8.4f}"
        lines.append(
            f"  {row.market[:24]:<24} {row.n_window:>6d} {row.elevation:>8.4f} "
            f"{row.z_score:>7.2f} {row.p_value:>8.4f}  |  {row.p_value_max:>8.4f} "
            f"{cross} {row.terminal_move:>8.3f}",
        )

    lines.extend(
        [
            "",
            f"Significant at alpha={summary.alpha:g}: {summary.n_significant}"
            f"/{len(summary.results)} markets "
            f"(expected {summary.alpha * len(summary.results):.1f} by chance)",
        ],
    )
    if summary.fisher_p is not None:
        lines.append(
            f"Fisher combined: X2={summary.fisher_stat:.2f} on "
            f"{2 * len(summary.results)} dof, p={summary.fisher_p:.4g}",
        )
    lines.extend(["", f"Note: {summary.to_dict()['robustness_note']}"])
    return "\n".join(lines)


def _format_calibration(report: dict) -> str:
    """Build the human-readable calibration table."""
    lines = [
        "=== Event-study window calibration (KTD4) ===",
        f"{report['n_replicates']} replicates per arm, "
        f"{report['n_permutations']} permutations, seed {report['seed']}, "
        f"alpha={report['alpha']:g}",
        "",
        f"  {'W (d)':>6} {'w (d)':>6} {'detection':>10} {'seam':>7} {'size':>7} "
        f"{'mean p (null)':>14}",
    ]
    for row in report["windows"]:
        lines.append(
            f"  {row['W_days']:>6.2f} {row['w_days']:>6.2f} "
            f"{row['detection_rate']:>10.3f} {row['seam_rejection_rate']:>7.3f} "
            f"{row['null_rejection_rate']:>7.3f} {row['mean_p']['null']:>14.3f}",
        )
    lines.extend(
        [
            "",
            report["arms_note"],
            "",
            f"Locked window: W={LOCKED_WINDOW_S / DAY_SECONDS:g} d, "
            f"w={LOCKED_EMBARGO_S / DAY_SECONDS:g} d. Do not re-lock it from a "
            "run that has already seen real-data p-values.",
        ],
    )
    return "\n".join(lines)


# ---------------- Entrypoints ----------------


def _run_calibration(args: argparse.Namespace) -> int:
    """Re-run the synthetic window calibration and print its table."""
    grid = [
        WindowSpec(W=days * DAY_SECONDS, w=args.embargo * DAY_SECONDS)
        for days in CALIBRATION_GRID_DAYS
        if days * DAY_SECONDS > args.embargo * DAY_SECONDS
    ]
    report = calibrate_window(
        grid,
        n_replicates=args.n_replicates,
        n_permutations=args.n_permutations,
        seed=args.seed,
        alpha=args.alpha,
    )
    print(_format_calibration(report))
    json_out = args.json_out if args.json_out is not None else DEFAULT_CALIBRATION_PATH
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", json_out)
    return 0


def _run_study(args: argparse.Namespace) -> int:
    """Run the event study over a replayed scores file.

    Args:
        args: Parsed namespace carrying the study-mode flags.

    Returns:
        Exit code 0 on success, 2 when the scores file carries no replay
        provenance, 3 when nothing could be analysed.
    """
    try:
        provenance = read_replay_provenance(args.scores)
    except ProvenanceError as err:
        log.error("%s", err)
        return 2

    scores = load_scores(args.scores)
    if not scores:
        log.error("%s holds no scored trades; nothing to analyse", args.scores)
        return 3
    resolutions = load_resolutions(args.resolutions)
    log.info(
        "read %d market(s) of scores and %d resolution record(s)",
        len(scores),
        len(resolutions),
    )

    summary = run_event_study(
        scores,
        resolutions,
        window=WindowSpec(
            W=args.window * DAY_SECONDS,
            w=args.embargo * DAY_SECONDS,
        ),
        n_permutations=args.n_permutations,
        seed=args.seed,
        alpha=args.alpha,
        provenance=provenance,
    )

    figures: list[str] = []
    if not args.no_figures and summary.results:
        figures = save_figures(summary, directory=args.fig_dir)
        for path in figures:
            log.info("wrote %s", path)

    print(_format_report(summary))
    json_out = args.json_out if args.json_out is not None else DEFAULT_SUMMARY_PATH
    write_summary(
        summary,
        json_out,
        extra={
            "scores": str(args.scores),
            "resolutions": str(args.resolutions),
            "figures": figures,
        },
    )
    log.info("wrote %s", json_out)
    if not summary.results:
        log.error("no market survived the exclusion checks; see the report above")
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the event study, or re-run the synthetic window calibration.

    Args:
        argv: Argument list passed to argparse; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: 0 on success, 1 when required study-mode arguments are
        missing, 2 when the scores file has no replay provenance, 3 when no
        market could be analysed.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.calibrate:
        return _run_calibration(args)
    if args.scores is None or args.resolutions is None:
        log.error("study mode needs both --scores and --resolutions")
        return 1
    return _run_study(args)


if __name__ == "__main__":
    sys.exit(main())
