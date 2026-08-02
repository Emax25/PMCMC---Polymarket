"""CLI: the Van Dyke / Maduro-cluster labeled-case study.

Thin CLI over `src.analysis.case_study`, which owns the manifest schema, the
window logic, the wallet table, the data-sufficiency verdict, the report and
the figure. This file wires arguments, the network pull and I/O only.

The chain is **manifest -> pull -> replay -> report**, and every step is driven
by the checked-in manifest (plan 2026-07-23-005 KTD5) so the bundle under
``results/case_studies/van_dyke/`` reproduces from it:

  1. ``--print-pull-command`` prints the documented `scripts.pull_data`
     invocation and exits. That is the batch pull, kept in `pull_data.py` where
     it belongs.
  2. ``--capture`` fetches the same markets' full histories in the
     `stream_trades.py` raw record shape, which is what
     ``score_stream.py --replay`` can actually consume — the batch processed
     format stores integer wallet ids and cannot be replayed or matched
     against a redacted wallet address. This is the only mode that touches the
     network.
  3. ``score_stream.py --replay`` is run separately, unchanged.
  4. The default mode reads that scores JSONL and writes the report, the
     summary JSON and the figure.

Reading the output. The **data-sufficiency section is the one to read first**:
the charged wallet has on the order of ten trades, far below the ~100 at which
this project's own `theta_w` posterior is meaningful (ARCHITECTURE.md 9.5), so
the wallet ranking is prior-dominated and the claim rests on per-trade ``P(Z)``
timing. The report says so in its own words; do not quote the rank without it.

Replay-mode scores only, on the same gate `src.analysis.event_study` uses: a
scores file without a ``mode == "replay"`` provenance sidecar is refused rather
than analysed with a caveat.

Examples:
    # 0. what to pull, and how (batch artifacts)
    python -m scripts.case_study --print-pull-command

    # 1. replayable capture of the cluster's full history (network)
    python -m scripts.case_study --capture \\
        --capture-out data/case_studies/van_dyke/trades.jsonl

    # 2. score it with no lookahead
    python -m scripts.score_stream --replay \\
        data/case_studies/van_dyke/trades.jsonl \\
        --warm-start results/validation/warm_start.json \\
        --output results/case_studies/van_dyke/scores.jsonl

    # 3. report
    python -m scripts.case_study \\
        --scores results/case_studies/van_dyke/scores.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from src.analysis.case_study import (
    DEFAULT_TOP_K,
    DEFAULT_TOP_TRADES,
    CaseManifest,
    ManifestError,
    format_report,
    load_manifest,
    load_scored_trades,
    raw_record,
    run_case_study,
    save_figures,
    write_capture,
    write_summary,
)
from src.analysis.event_study import ProvenanceError, read_replay_provenance

log = logging.getLogger("case_study")

DEFAULT_MANIFEST = Path("results/case_studies/van_dyke/markets.json")
DEFAULT_OUT_DIR = Path("results/case_studies/van_dyke")
DEFAULT_CAPTURE_OUT = Path("data/case_studies/van_dyke/trades.jsonl")

REPORT_NAME = "report.md"
SUMMARY_NAME = "summary.json"
FIGURE_SUBDIR = "figures"

# Exit codes, so a pipeline can tell the three failure modes apart.
EXIT_OK = 0
EXIT_BAD_MANIFEST = 1
EXIT_NO_PROVENANCE = 2
EXIT_NOTHING_TO_ANALYSE = 3


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the case_study argument parser covering all three modes."""
    p = argparse.ArgumentParser(
        description=(
            "Labeled-case study over replayed per-trade insider scores: where "
            "does the DOJ/CFTC-charged Van Dyke cluster rank, and when did the "
            "score move?"
        ),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Checked-in case manifest (default: {DEFAULT_MANIFEST}). It "
        "defines the cluster, the wallet anchor and the analysis window; "
        "nothing is inferred from the data.",
    )
    p.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Scores JSONL from `score_stream.py --replay`. Its "
        "<scores>.meta.json sidecar must record mode=replay.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Bundle directory for {REPORT_NAME}, {SUMMARY_NAME} and "
        f"{FIGURE_SUBDIR}/ (default: {DEFAULT_OUT_DIR}).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Wallets in the ranking table (default: {DEFAULT_TOP_K}). An "
        "anchored wallet below the cut is appended, never hidden.",
    )
    p.add_argument(
        "--top-trades",
        type=int,
        default=DEFAULT_TOP_TRADES,
        help=f"Individual trades in the timing table (default: "
        f"{DEFAULT_TOP_TRADES}).",
    )
    p.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip the figure; the report and summary JSON are unaffected.",
    )
    p.add_argument(
        "--print-pull-command",
        action="store_true",
        help="Print the manifest's documented pull_data command and exit.",
    )
    p.add_argument(
        "--capture",
        action="store_true",
        help="Fetch the cluster's full trade history into a replayable JSONL "
        "capture and exit. The only mode that hits the network.",
    )
    p.add_argument(
        "--capture-out",
        type=Path,
        default=DEFAULT_CAPTURE_OUT,
        help=f"Capture destination (default: {DEFAULT_CAPTURE_OUT}).",
    )
    p.add_argument(
        "--sleep-between",
        type=float,
        default=0.1,
        help="Seconds between paginated /trades calls during --capture.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def _run_capture(manifest: CaseManifest, args: argparse.Namespace) -> int:
    """Pull every cluster market's full history into one replayable capture.

    Per-market failure is *not* isolated here, unlike `scripts.pull_data`: a
    case study built on a partial cluster would quietly answer a different
    question than the manifest documents, so a failed market aborts and the
    operator re-runs.

    Args:
        manifest: The loaded case definition.
        args: Parsed namespace, read for the destination and rate limit.

    Returns:
        Exit code 0 on success, `EXIT_NOTHING_TO_ANALYSE` when no market
        returned a trade.
    """
    # Deferred: the API client pulls `requests`, which the report path (and its
    # tests) has no use for.
    from src.data.polymarket_api import fetch_trades_windowed

    records: list[dict[str, Any]] = []
    for market in manifest.markets:
        trades = fetch_trades_windowed(
            market.condition_id,
            sleep_between=args.sleep_between,
        )
        log.info("%-56s %6d trades", market.slug, len(trades))
        records.extend(raw_record(trade) for trade in trades)

    if not records:
        log.error("no trades pulled for any manifest market; nothing to capture")
        return EXIT_NOTHING_TO_ANALYSE
    path = write_capture(records, args.capture_out)
    log.info(
        "wrote %s (%d records across %d market(s), full history, no "
        "pre-resolution filter)",
        path,
        len(records),
        len(manifest.markets),
    )
    return EXIT_OK


def _run_report(manifest: CaseManifest, args: argparse.Namespace) -> int:
    """Read replayed scores and write the report, summary JSON and figure.

    Args:
        manifest: The loaded case definition.
        args: Parsed namespace carrying the report-mode flags.

    Returns:
        Exit code 0 on success, `EXIT_NO_PROVENANCE` when the scores file
        carries no replay provenance, `EXIT_NOTHING_TO_ANALYSE` when it holds
        no trade in the manifest's cluster.
    """
    try:
        provenance = read_replay_provenance(args.scores)
    except ProvenanceError as err:
        log.error("%s", err)
        return EXIT_NO_PROVENANCE

    trades = load_scored_trades(args.scores, condition_ids=manifest.condition_ids)
    if trades.n == 0:
        log.error(
            "%s holds no scored trade for any of the %d manifest market(s); "
            "nothing to analyse",
            args.scores,
            len(manifest.markets),
        )
        return EXIT_NOTHING_TO_ANALYSE

    summary = run_case_study(
        trades,
        manifest,
        top_k=args.top_k,
        top_trades=args.top_trades,
        provenance=provenance,
    )

    if summary.is_cold_start:
        log.warning(
            "%s was scored WITHOUT a warm start, so every P(Z) is the prior "
            "mean plus filter noise. The bundle is written (the pipeline ran) "
            "but it is not a result — re-score with `score_stream.py --replay "
            "--warm-start <fitted VEM artifact>`.",
            args.scores,
        )

    figures: list[str] = []
    if not args.no_figures:
        figures = save_figures(summary, directory=args.out_dir / FIGURE_SUBDIR)
        for path in figures:
            log.info("wrote %s", path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / REPORT_NAME
    report_path.write_text(format_report(summary), encoding="utf-8")
    log.info("wrote %s", report_path)

    summary_path = write_summary(
        summary,
        args.out_dir / SUMMARY_NAME,
        extra={
            "scores": str(args.scores),
            "report": str(report_path),
            "figures": figures,
        },
    )
    log.info("wrote %s", summary_path)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Print the pull command, build the capture, or write the case report.

    Args:
        argv: Argument list passed to argparse; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: 0 on success, 1 when the manifest is unusable or report mode
        was invoked without ``--scores``, 2 when the scores file has no replay
        provenance, 3 when nothing could be analysed.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as err:
        log.error("%s", err)
        return EXIT_BAD_MANIFEST

    if args.print_pull_command:
        print(manifest.pull.command)
        return EXIT_OK
    if args.capture:
        return _run_capture(manifest, args)
    if args.scores is None:
        log.error(
            "report mode needs --scores (or pass --capture / "
            "--print-pull-command)",
        )
        return EXIT_BAD_MANIFEST
    return _run_report(manifest, args)


if __name__ == "__main__":
    sys.exit(main())
