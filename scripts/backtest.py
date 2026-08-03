"""CLI: costed tradeability check on ``P(Z)`` — detection PoC, **not alpha**.

Thin CLI over `src.analysis.backtest`, which owns the cost model, the
pre-declared threshold grid, the purged walk-forward and the deflated Sharpe.
This file wires arguments, reporting and I/O only.

**What this command is.** A proof-of-concept check on whether the detector's
per-trade insider score survives a realistic spread and Kalshi's taker fee
(``0.07 * p * (1 - p)`` per contract, maximal mid-book where the signal is
weakest). One position per market, entered on the first trade whose ``P(Z)``
crosses a threshold from a grid declared in source, held to resolution.

**What this command is not.** It is not a validated trading strategy, it is not
evidence of alpha, and no line of its output is a trading recommendation. Every
artifact it writes carries that framing verbatim, and the headline number is a
*deflated* Sharpe — probabilistic Sharpe against the expected maximum over the
whole disclosed grid, computed from the empirical variance across the trial
Sharpes rather than from a raw trial count.

Replay-mode scores only. The no-lookahead property is inherited from
`score_stream.py --replay`; a scores file without a ``mode == "replay"``
provenance sidecar is refused rather than analysed with a caveat.

Examples:
    # The real-data run: declared grid, Kalshi fee, 2c spread, 4 folds
    python -m scripts.backtest --scores results/streaming/scores.jsonl \\
        --resolutions data/processed \\
        --json-out results/backtest/summary.json

    # Wider assumed spread, and outcomes from a separate settlements file
    python -m scripts.backtest --scores results/streaming/scores.jsonl \\
        --resolutions data/processed --outcomes data/settlements.json \\
        --spread 0.04
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

from src.analysis.backtest import (
    DAY_SECONDS,
    DECLARED_THRESHOLD_GRID,
    DEFAULT_EMBARGO_S,
    DEFAULT_N_SPLITS,
    DEFAULT_SPREAD,
    KALSHI_FEE_RATE,
    BacktestSummary,
    CostModel,
    build_panels,
    load_outcomes,
    run_backtest,
    save_figures,
    write_summary,
)
from src.analysis.event_study import (
    ProvenanceError,
    load_resolutions,
    load_scores,
    read_replay_provenance,
)

log = logging.getLogger("backtest")

DEFAULT_FIG_DIR = Path("results/figures/backtest")
DEFAULT_SUMMARY_PATH = Path("results/backtest/summary.json")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the backtest argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Costed tradeability check on the P(Z) detection signal: threshold "
            "entry, hold to resolution, spread + Kalshi taker fee, purged "
            "walk-forward, deflated Sharpe. A detection-signal proof of "
            "concept, NOT a validated alpha strategy and not a trading "
            "recommendation."
        ),
    )
    p.add_argument(
        "--scores",
        type=Path,
        required=True,
        help="Scores JSONL from `score_stream.py --replay`. Its "
        "<scores>.meta.json sidecar must record mode=replay; live-mode or "
        "provenance-less scores are refused.",
    )
    p.add_argument(
        "--resolutions",
        type=Path,
        required=True,
        help="Market resolution metadata: a directory of *.meta.json sidecars, "
        "or a JSON object/array of records carrying close_time / end_date / "
        "close_ts. Markets missing from it are excluded and counted.",
    )
    p.add_argument(
        "--outcomes",
        type=Path,
        default=None,
        help="Settled outcomes, same three shapes as --resolutions, read from "
        "result / outcome / settlement fields (default: --resolutions, which "
        "normally carries both).",
    )
    p.add_argument(
        "--spread",
        type=float,
        default=DEFAULT_SPREAD,
        help=f"Assumed bid-ask spread in probability units; entry pays half "
        f"(default: {DEFAULT_SPREAD:g}).",
    )
    p.add_argument(
        "--fee-rate",
        type=float,
        default=KALSHI_FEE_RATE,
        help=f"Taker-fee coefficient in fee = rate * p * (1 - p) (default: "
        f"Kalshi's {KALSHI_FEE_RATE:g}).",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Purged walk-forward folds (default: {DEFAULT_N_SPLITS}).",
    )
    p.add_argument(
        "--embargo",
        type=float,
        default=DEFAULT_EMBARGO_S / DAY_SECONDS,
        help="Embargo in days between a training market's resolution and the "
        f"test block's label span (default: {DEFAULT_EMBARGO_S / DAY_SECONDS:g}).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"JSON output path (default: {DEFAULT_SUMMARY_PATH}).",
    )
    p.add_argument(
        "--fig-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help=f"Directory for the PoC figure (default: {DEFAULT_FIG_DIR}).",
    )
    p.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip the figure; the report and summary JSON are unaffected.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


# ---------------- Report ----------------

_HEADER = "=== Costed tradeability check on P(Z) (DETECTION POC - NOT ALPHA) ==="

_DSR_READING = (
    "Reading the DSR: it is P(true Sharpe > the expected maximum over the "
    "disclosed grid), so a value near 0.5 or below says the result is "
    "indistinguishable from the best of that many correlated trials. The "
    "benchmark uses the empirical variance across the trial Sharpes, not the "
    "trial count alone. Whatever the number, it is a detection-signal PoC on a "
    "small market sample - not a validated strategy."
)


def _fmt(value: float, width: int = 8, digits: int = 4) -> str:
    """Fixed-width float, printing NaN as ``n/a`` rather than as a number."""
    if math.isnan(value):
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{digits}f}"


def _format_report(summary: BacktestSummary) -> str:
    """Build the human-readable report, PoC framing first and last.

    The deflated Sharpe leads and the undeflated PSR is shown next to it, so a
    reader sees the size of the multiple-testing correction rather than only its
    result. Every declared threshold is printed whether or not it was ever
    selected — that list *is* the disclosed trial count.
    """
    deflated = summary.deflated
    returns = summary.returns
    # `to_dict` serializes every fold and every position, so build it once and
    # reuse the framing note for both the opening and closing block.
    framing = summary.to_dict()["framing"]
    lines = [
        _HEADER,
        "",
        framing,
        "",
        f"Markets: {summary.n_markets} scored, {summary.panels} tradeable, "
        f"{len(summary.excluded)} excluded",
        f"Costs: spread={summary.cost_model.spread:g} "
        f"(half paid on entry), taker fee = "
        f"{summary.cost_model.fee_rate:g} * p * (1 - p) per contract",
        f"Walk-forward: {summary.n_splits} folds, embargo "
        f"{summary.embargo_s / DAY_SECONDS:g} d, purged label windows",
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

    lines.extend(
        [
            "",
            "Declared threshold grid (the disclosed trial count), each run "
            "fixed across the same folds:",
            f"  {'tau':>6} {'n_pos':>6} {'mean_ret':>10} {'hit':>7} {'sharpe':>8}",
        ],
    )
    for row in summary.trials:
        lines.append(
            f"  {row.threshold:>6.2f} {row.n_positions:>6d} "
            f"{_fmt(row.mean_return, 10)} {_fmt(row.hit_rate, 7, 3)} "
            f"{_fmt(row.sharpe)}",
        )

    lines.extend(
        [
            "",
            "Walk-forward folds (threshold selected in-sample, applied "
            "out-of-sample):",
            f"  {'fold':>4} {'n_tr':>5} {'n_te':>5} {'purged':>7} {'tau*':>6} "
            f"{'sr_train':>9} {'n_pos':>6} {'mean_ret':>10}",
        ],
    )
    for fold in summary.folds:
        fold_returns = fold.returns
        tau = "   n/a" if fold.threshold is None else f"{fold.threshold:>6.2f}"
        mean_ret = fold_returns.mean() if fold_returns.size else float("nan")
        lines.append(
            f"  {fold.fold:>4d} {fold.n_train:>5d} {fold.n_test:>5d} "
            f"{fold.n_purged:>7d} {tau} {_fmt(fold.train_sharpe, 9)} "
            f"{fold_returns.size:>6d} {_fmt(mean_ret, 10)}",
        )

    lines.extend(
        [
            "",
            f"Out-of-sample: {returns.size} position(s), "
            f"total net PnL {returns.sum():.4f} $/contract, "
            f"mean {_fmt(returns.mean() if returns.size else float('nan')).strip()}",
            f"Sharpe (per trade): {_fmt(deflated.sharpe).strip()}   "
            f"skew {_fmt(deflated.skewness).strip()}   "
            f"kurtosis {_fmt(deflated.kurtosis).strip()}",
            f"Trials disclosed: {deflated.n_trials} "
            f"({deflated.n_trial_sharpes} with a defined Sharpe), "
            f"variance across trial Sharpes {deflated.trial_variance:.6f}",
            f"DSR benchmark SR0: {_fmt(deflated.sr_benchmark).strip()}",
            f"PSR vs 0 (undeflated): {_fmt(deflated.psr_zero).strip()}",
            f"DEFLATED SHARPE: {_fmt(deflated.dsr).strip()}",
            "",
            _DSR_READING,
            "",
            framing,
        ],
    )
    return "\n".join(lines)


# ---------------- Entrypoint ----------------


def main(argv: list[str] | None = None) -> int:
    """Run the costed backtest over a replayed scores file.

    Args:
        argv: Argument list passed to argparse; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: 0 on success, 2 when the scores file carries no replay
        provenance, 3 when no market was tradeable.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        provenance = read_replay_provenance(args.scores)
    except ProvenanceError as err:
        log.error("%s", err)
        return 2

    scores = load_scores(args.scores)
    if not scores:
        log.error("%s holds no scored trades; nothing to backtest", args.scores)
        return 3
    resolutions = load_resolutions(args.resolutions)
    outcomes = load_outcomes(
        args.outcomes if args.outcomes is not None else args.resolutions,
    )
    log.info(
        "read %d market(s) of scores, %d close time(s), %d settled outcome(s)",
        len(scores),
        len(resolutions),
        len(outcomes),
    )

    panels, excluded = build_panels(scores, resolutions, outcomes)
    summary = run_backtest(
        panels,
        thresholds=DECLARED_THRESHOLD_GRID,
        cost_model=CostModel(spread=args.spread, fee_rate=args.fee_rate),
        n_splits=args.n_splits,
        embargo_s=args.embargo * DAY_SECONDS,
        n_markets=len(scores),
        excluded=excluded,
        provenance=provenance,
    )

    figures: list[str] = []
    if not args.no_figures and panels:
        figures = save_figures(summary, directory=args.fig_dir)
        for path in figures:
            log.info("wrote %s", path)

    print(_format_report(summary))
    write_summary(
        summary,
        args.json_out,
        extra={
            "scores": str(args.scores),
            "resolutions": str(args.resolutions),
            "outcomes": str(args.outcomes or args.resolutions),
            "figures": figures,
        },
    )
    log.info("wrote %s", args.json_out)
    if not panels:
        log.error("no market was tradeable; see the exclusions above")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
