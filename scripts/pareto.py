"""CLI: headline Pareto figure — synthetic ROC AUC vs wall-clock from benchmarks.

Reads one or more JSON files produced by ``scripts.benchmark --json-out`` and
plots pooled synthetic gate AUC (y) against mean seconds per run (x, log scale).

Example:
    python -m scripts.pareto \\
        --bench-json results/bench_pg.json results/bench_vem.json \\
        --output results/figures/pareto.png \\
        --csv-out results/tables/pareto_summary.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.plots import pareto_plot, set_paper_style

log = logging.getLogger("pareto")

_CSV_COLUMNS = (
    "label",
    "method",
    "N",
    "n_iter",
    "K",
    "mean_sec_per_run",
    "ci_half_width",
    "pooled_auc",
    "gate_pass",
    "kendall_tau_vs_baseline",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for pareto."""
    p = argparse.ArgumentParser(
        description="Pareto plot: synthetic ROC AUC vs wall-clock from benchmark JSON.",
    )
    p.add_argument(
        "--bench-json",
        type=Path,
        nargs="+",
        required=True,
        help="Benchmark JSON file(s) from scripts.benchmark --json-out.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Display labels (same count as --bench-json); default derives from JSON.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/pareto.png"),
        help="Output PNG path (default: results/figures/pareto.png).",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional summary CSV path.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def _default_label(payload: dict[str, Any]) -> str:
    """Derive a short label from method and config keys in a benchmark payload."""
    method = payload.get("method", "?")
    cfg = payload.get("config") or {}
    n_particles = cfg.get("N", "?")
    n_iter = cfg.get("n_iter", "?")
    return f"{method} N={n_particles} it={n_iter}"


def _load_benchmark_row(
    path: Path,
    label: str,
) -> dict[str, Any] | None:
    """Parse one benchmark JSON into a summary row; None when gate is absent."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("gate")
    if gate is None:
        log.warning("skipping %s: gate is null (run with --gate)", path)
        return None

    timings = payload.get("timings") or {}
    config = payload.get("config") or {}
    inputs = payload.get("inputs") or {}

    tau = payload.get("kendall_tau_vs_baseline")
    return {
        "label": label,
        "method": payload.get("method", ""),
        "N": config.get("N"),
        "n_iter": config.get("n_iter"),
        "K": inputs.get("K"),
        "mean_sec_per_run": timings.get("mean_sec_per_run"),
        "ci_half_width": timings.get("ci_half_width_sec_per_run", 0.0) or 0.0,
        "pooled_auc": gate.get("pooled_auc"),
        "gate_pass": gate.get("gate_pass"),
        "kendall_tau_vs_baseline": tau if tau is not None else "",
    }


def _rows_to_csv_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a CSV-ready DataFrame with stable column order."""
    return pd.DataFrame(rows, columns=list(_CSV_COLUMNS))


def main(argv: list[str] | None = None) -> int:
    """Build the Pareto figure and optional summary CSV from benchmark JSON files.

    Args:
        argv: Argument list passed to argparse; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code (0 on success, 1 when no plottable rows remain).
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    paths = args.bench_json
    if args.labels is not None and len(args.labels) != len(paths):
        log.error(
            "--labels count (%d) must match --bench-json count (%d)",
            len(args.labels),
            len(paths),
        )
        return 1

    labels = args.labels or [
        _default_label(json.loads(p.read_text(encoding="utf-8"))) for p in paths
    ]

    rows: list[dict[str, Any]] = []
    for path, label in zip(paths, labels):
        row = _load_benchmark_row(path, label)
        if row is not None:
            rows.append(row)

    if not rows:
        log.error("no benchmark files with gate metrics; nothing to plot")
        return 1

    plot_df = pd.DataFrame(rows)
    set_paper_style()
    fig = pareto_plot(plot_df, title=args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", args.output)

    if args.csv_out is not None:
        csv_df = _rows_to_csv_frame(rows)
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        csv_df.to_csv(args.csv_out, index=False)
        log.info("wrote %s", args.csv_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
