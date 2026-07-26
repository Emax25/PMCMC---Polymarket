"""Tests for src.analysis.sbc and the scripts/sbc.py --analyze path.

Everything here is built against synthetic JSONL fixtures written to tmp_path,
so no inference runs: the analysis layer's contract is "given a results store,
produce the right verdicts", and that is exactly what these exercise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from scripts import sbc as sbc_cli
from src.analysis.plots import figure_sbc_ranks, save_paper_figure
from src.analysis.sbc import (
    COVERAGE_BAND,
    PHI_COMPONENTS,
    SBC_SCHEMA_VERSION,
    THETA_W_KEY,
    analyze,
    coverage_table,
    default_n_bins,
    failure_accounting,
    load_results,
    rank_bin_edges,
    rank_uniformity,
    usable_rows,
    wilson_interval,
    write_summary,
)
from src.inference.diagnostics import PHI_PARAM_NAMES

L_DRAWS = 99  # small support keeps fixtures fast; rank in {0, ..., 99}


# ---------------- Fixture builders ----------------


def _row(
    seed: int,
    *,
    ranks: dict[str, int] | None,
    hits: dict[str, bool] | None,
    theta_hits: list[bool] | None = None,
    z_auc: float | None = 0.8,
    vem_converged: bool = True,
    laplace_fallback: bool = False,
    failed: bool = False,
    error: str | None = None,
    L: int = L_DRAWS,
) -> dict[str, Any]:
    """Build one schema-v1 replicate row."""
    return {
        "schema_version": SBC_SCHEMA_VERSION,
        "seed": seed,
        "phi_true": {name: 0.5 for name in PHI_COMPONENTS},
        "L": L,
        "ranks": ranks,
        "hits90": hits,
        "theta_hits90": theta_hits,
        "z_auc": z_auc,
        "flags": {
            "vem_converged": vem_converged,
            "laplace_fallback": laplace_fallback,
            "failed": failed,
            "error": error,
        },
        "elapsed_s": 1.25,
        "size": {"K": 3, "T": 400, "n_wallets": 30},
    }


def _calibrated_rows(n: int = 400, seed: int = 20260723) -> list[dict[str, Any]]:
    """Rows whose ranks are uniform and whose 90% intervals hit at rate 0.9."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        ranks = {c: int(rng.integers(0, L_DRAWS + 1)) for c in PHI_COMPONENTS}
        hits = {c: bool(rng.random() < 0.9) for c in PHI_COMPONENTS}
        theta = [bool(v) for v in rng.random(5) < 0.9]
        rows.append(
            _row(
                i,
                ranks=ranks,
                hits=hits,
                theta_hits=theta,
                z_auc=float(rng.uniform(0.7, 0.95)),
            ),
        )
    return rows


def _miscalibrated_rows(n: int = 400, seed: int = 7) -> list[dict[str, Any]]:
    """Rows with ranks piled at both extremes and 90% intervals that rarely hit."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        ranks = {
            c: int(0 if rng.random() < 0.5 else L_DRAWS) for c in PHI_COMPONENTS
        }
        hits = {c: bool(rng.random() < 0.2) for c in PHI_COMPONENTS}
        rows.append(_row(i, ranks=ranks, hits=hits))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write rows as JSONL, one object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


# ---------------- Constants and small helpers ----------------


def test_phi_components_match_canonical_order():
    """The locally restated component tuple must not drift from the canonical one."""
    assert PHI_COMPONENTS == PHI_PARAM_NAMES


def test_rank_bin_edges_partition_the_support_exactly():
    """Edges are integers spanning 0..L+1 with widths summing to the support size."""
    for L, n_bins in ((99, 20), (999, 20), (10, 3), (6, 4)):
        edges = rank_bin_edges(L, n_bins)
        assert edges[0] == 0
        assert edges[-1] == L + 1
        widths = np.diff(edges)
        assert widths.sum() == L + 1
        assert np.all(widths > 0)
        # Widths differ by at most one rank, so no bin is a rounding artefact.
        assert widths.max() - widths.min() <= 1


def test_default_n_bins_respects_expected_count_floor():
    """Bin count never lets the expected count per bin fall below the floor."""
    assert default_n_bins(400, 99) == 20
    assert default_n_bins(40, 99) == 8
    assert default_n_bins(3, 99) == 2  # floor of 2 bins even for tiny n


def test_wilson_interval_brackets_the_point_estimate():
    """Wilson bounds bracket the rate and stay inside [0, 1] at the extremes."""
    lo, hi = wilson_interval(90, 100)
    assert lo < 0.9 < hi
    assert (0.0, 1.0) == wilson_interval(0, 0)
    lo, hi = wilson_interval(100, 100)
    assert 0.0 <= lo < 1.0
    assert hi <= 1.0


# ---------------- Calibrated fixture ----------------


def test_calibrated_fixture_passes_uniformity_and_coverage():
    """Uniform ranks are not rejected and 90% hits land inside the target band."""
    rows = _calibrated_rows()
    summary = analyze(rows)

    assert summary.n_replicates == 400
    assert summary.n_analysed == 400
    assert summary.L == L_DRAWS

    for name, result in summary.uniformity.items():
        assert not result.rejected, f"{name} p={result.p_value}"
        assert not result.band_violation, f"{name} D={result.ks_stat}"
        assert result.shape_hint == "no dominant deviation"
        assert result.counts.sum() == 400

    for row in summary.coverage:
        assert row.in_range, f"{row.name} rate={row.rate}"
        assert COVERAGE_BAND[0] <= row.rate <= COVERAGE_BAND[1]
        assert row.ci_lo <= row.rate <= row.ci_hi

    assert summary.flagged == []
    assert summary.failures.failure_rate == 0.0
    assert summary.z_auc["n"] == 400


def test_calibrated_fixtures_rarely_flag_across_seeds():
    """The fixture seed is not lucky: false flags stay near the corrected level.

    Two verdicts each controlled at alpha=0.05 across components put the
    expected false-flag rate near 10%; anything much above that would mean the
    multiplicity correction is not doing its job and the paper's "no component
    flagged" claim would be worth nothing.
    """
    flagged = [
        bool(analyze(_calibrated_rows(n=200, seed=1000 + s)).flagged)
        for s in range(20)
    ]
    assert sum(flagged) <= 5


def test_calibrated_fixture_reports_theta_and_z_blocks():
    """The theta_w aggregate row and its clustering caveat are both present."""
    summary = analyze(_calibrated_rows())
    names = [row.name for row in summary.coverage]
    assert names[-1] == THETA_W_KEY
    theta_row = summary.coverage[-1]
    assert theta_row.n == 400 * 5
    assert "anti-conservative" in theta_row.note

    assert summary.theta_w is not None
    assert summary.theta_w["n_replicates"] == 400
    assert summary.theta_w["n_wallet_intervals"] == 2000
    assert summary.theta_w["between_replicate_sd"] > 0.0


# ---------------- Miscalibrated fixture ----------------


def test_miscalibrated_fixture_is_rejected_and_flagged():
    """Ranks at the extremes reject uniformity, read as overconfident, fail coverage."""
    summary = analyze(_miscalibrated_rows())

    for name, result in summary.uniformity.items():
        assert result.rejected, f"{name} p={result.p_value}"
        assert result.band_violation, f"{name} D={result.ks_stat}"
        assert result.dispersion_z > 0.0
        assert "overconfident" in result.shape_hint

    for row in summary.coverage:
        assert not row.in_range
        assert row.rate < COVERAGE_BAND[0]

    assert len(summary.flagged) >= len(PHI_COMPONENTS)
    assert any("overconfident" in text for text in summary.flagged)


def test_rank_uniformity_rejects_a_location_bias():
    """Ranks concentrated at the top read as a slope, not as a dispersion problem."""
    rng = np.random.default_rng(3)
    rows = [
        _row(
            i,
            ranks={c: int(rng.integers(70, L_DRAWS + 1)) for c in PHI_COMPONENTS},
            hits={c: True for c in PHI_COMPONENTS},
        )
        for i in range(200)
    ]
    results = rank_uniformity(usable_rows(rows))
    for result in results.values():
        assert result.rejected
        assert result.bias_z > 0.0
        assert "biased low" in result.shape_hint


# ---------------- Failure and degeneracy accounting ----------------


def _mixed_rows() -> list[dict[str, Any]]:
    """5 clean rows, 3 failed rows, 2 degenerate-but-scored rows."""
    rng = np.random.default_rng(11)

    def _ok(seed: int, **kwargs: Any) -> dict[str, Any]:
        return _row(
            seed,
            ranks={c: int(rng.integers(0, L_DRAWS + 1)) for c in PHI_COMPONENTS},
            hits={c: True for c in PHI_COMPONENTS},
            **kwargs,
        )

    rows = [_ok(i) for i in range(5)]
    rows += [
        _row(
            100 + i,
            ranks=None,
            hits=None,
            theta_hits=None,
            z_auc=None,
            failed=True,
            error="LinAlgError" if i < 2 else "Timeout",
        )
        for i in range(3)
    ]
    rows.append(_ok(200, vem_converged=False))
    rows.append(_ok(201, laplace_fallback=True))
    return rows


def test_failed_rows_are_counted_but_never_scored():
    """Failed replicates leave the rank stats untouched yet drive the failure rate."""
    rows = _mixed_rows()
    accounting = failure_accounting(rows)

    assert accounting.n_replicates == 10
    assert accounting.n_failed == 3
    assert accounting.n_scored == 7
    assert accounting.failure_rate == pytest.approx(0.3)
    assert accounting.error_counts == {"LinAlgError": 2, "Timeout": 1}

    assert accounting.n_degenerate == 2
    assert accounting.n_vem_nonconverged == 1
    assert accounting.n_laplace_fallback == 1
    assert accounting.degenerate_rate == pytest.approx(2 / 7)


def test_degenerate_rows_excluded_by_default_and_reported_as_sensitivity():
    """Default selection keeps 5 rows; include_degenerate keeps 7; both are reported."""
    rows = _mixed_rows()
    assert len(usable_rows(rows)) == 5
    assert len(usable_rows(rows, include_degenerate=True)) == 7

    summary = analyze(rows)
    assert summary.n_replicates == 10
    assert summary.n_analysed == 5
    assert next(iter(summary.uniformity.values())).n == 5
    assert summary.coverage[0].n == 5
    assert summary.sensitivity is not None
    assert summary.sensitivity["n_analysed"] == 7
    assert set(summary.sensitivity["coverage_rates"]) == set(PHI_COMPONENTS)

    strict = analyze(rows, include_degenerate=True)
    assert strict.n_analysed == 7
    assert strict.sensitivity is None
    # The failure rate is a property of the store, not of the selection policy.
    assert strict.failures.failure_rate == summary.failures.failure_rate

    assert any("failure rate" in text for text in summary.flagged)


def test_analyze_raises_when_every_replicate_failed():
    """An all-failed store is an error, not an empty table that looks like a pass."""
    rows = [
        _row(i, ranks=None, hits=None, z_auc=None, failed=True, error="boom")
        for i in range(4)
    ]
    with pytest.raises(ValueError, match="no usable replicates"):
        analyze(rows)


# ---------------- Loading ----------------


def test_load_results_round_trips_and_drops_duplicate_seeds(tmp_path):
    """Blank lines are skipped and a repeated seed keeps only its first row."""
    rows = _calibrated_rows(n=6)
    path = tmp_path / "results.jsonl"
    _write_jsonl(path, rows + [rows[0]])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    loaded = load_results(path)
    assert len(loaded) == 6
    assert [r["seed"] for r in loaded] == list(range(6))


def test_load_results_rejects_wrong_schema_and_malformed_lines(tmp_path):
    """A store this analysis cannot interpret fails loudly with the line number."""
    bad_version = tmp_path / "v2.jsonl"
    row = _calibrated_rows(n=1)[0]
    row["schema_version"] = 2
    _write_jsonl(bad_version, [row])
    with pytest.raises(ValueError, match="schema_version"):
        load_results(bad_version)

    malformed = tmp_path / "broken.jsonl"
    malformed.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        load_results(malformed)


def test_rank_uniformity_rejects_out_of_range_ranks():
    """A rank outside {0, ..., L} is a harness bug and must not be silently binned."""
    rows = _calibrated_rows(n=10)
    rows[0]["ranks"]["beta_S"] = L_DRAWS + 5
    with pytest.raises(ValueError, match="outside"):
        rank_uniformity(usable_rows(rows))


def test_analyze_rejects_mixed_posterior_draw_counts():
    """Ranks from different L are not comparable; the mismatch must surface."""
    rows = _calibrated_rows(n=10)
    rows[0]["L"] = L_DRAWS + 1
    with pytest.raises(ValueError, match="disagree on L"):
        analyze(rows)


def test_coverage_table_rejects_a_missing_hit():
    """A usable row missing a component's hit indicator is an error, not a skip."""
    rows = usable_rows(_calibrated_rows(n=10))
    del rows[0]["hits90"]["q_01"]
    with pytest.raises(ValueError, match="missing hit"):
        coverage_table(rows)


# ---------------- Figures ----------------


@pytest.mark.parametrize("kind", ["ecdf", "hist"])
def test_sbc_rank_figures_render_to_disk(tmp_path, kind):
    """Both figure kinds render at fixture scale and land as non-empty files."""
    results = rank_uniformity(usable_rows(_calibrated_rows(n=60)))
    fig = figure_sbc_ranks(results, kind=kind)
    paths = save_paper_figure(fig, f"sbc_rank_{kind}", directory=tmp_path)
    assert len(paths) == 2
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_figure_sbc_ranks_validates_its_inputs():
    """Empty results and unknown panel kinds are rejected up front."""
    results = rank_uniformity(usable_rows(_calibrated_rows(n=20)))
    with pytest.raises(ValueError, match="empty"):
        figure_sbc_ranks({})
    with pytest.raises(ValueError, match="kind must be"):
        figure_sbc_ranks(results, kind="violin")


# ---------------- Summary I/O and CLI ----------------


def test_write_summary_is_json_serializable(tmp_path):
    """Every field of the summary survives a JSON round trip, extras included."""
    summary = analyze(_calibrated_rows(n=50))
    path = write_summary(summary, tmp_path / "sub" / "summary.json", extra={"src": "x"})
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["src"] == "x"
    assert payload["n_analysed"] == 50
    assert set(payload["uniformity"]) == set(PHI_COMPONENTS)
    assert payload["uniformity"]["beta_S"]["counts"]
    assert "pit" not in payload["uniformity"]["beta_S"]
    assert payload["coverage"][0]["name"] == PHI_COMPONENTS[0]
    assert "overconfident" in payload["interpretation_key"]


def test_cli_analyze_end_to_end(tmp_path, capsys):
    """--analyze reads a store and writes the report, both figures, and summary.json."""
    store = _write_jsonl(tmp_path / "sbc.jsonl", _calibrated_rows(n=80))
    fig_dir = tmp_path / "figs"
    summary_path = tmp_path / "summary.json"

    code = sbc_cli.main(
        [
            "--analyze",
            "--in",
            str(store),
            "--fig-dir",
            str(fig_dir),
            "--summary",
            str(summary_path),
        ],
    )
    assert code == 0

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["source"] == str(store)
    assert payload["n_analysed"] == 80
    assert len(payload["figures"]) == 4  # two figures x (pdf, png)
    for name in payload["figures"]:
        assert Path(name).stat().st_size > 0

    out = capsys.readouterr().out
    assert "SBC and coverage analysis" in out
    assert "Rank uniformity" in out
    assert "Coverage of nominal-90% intervals" in out
    assert "Failures:" in out


def test_cli_analyze_prints_flagged_block_for_a_bad_store(tmp_path, capsys):
    """A miscalibrated store surfaces its FLAGGED block above the tables."""
    store = _write_jsonl(tmp_path / "bad.jsonl", _miscalibrated_rows(n=100))
    code = sbc_cli.main(
        [
            "--analyze",
            "--in",
            str(store),
            "--fig-dir",
            str(tmp_path / "figs"),
            "--summary",
            str(tmp_path / "summary.json"),
        ],
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "!!! FLAGGED !!!" in out
    assert out.index("!!! FLAGGED !!!") < out.index("Rank uniformity")


def test_cli_run_mode_is_not_implemented_yet():
    """The run path declares its flags but defers to the harness unit."""
    with pytest.raises(NotImplementedError, match="U2"):
        sbc_cli.main(["--n-sims", "2"])


def test_cli_declares_the_run_mode_flag_surface():
    """The harness flags exist now so the harness unit only fills in behaviour."""
    args = sbc_cli._parse_args([])
    for flag in (
        "n_sims",
        "n_jobs",
        "sim_K",
        "sim_T",
        "sim_wallets",
        "posterior_draws",
        "out",
        "resume",
        "seed_base",
    ):
        assert hasattr(args, flag), flag
    assert args.posterior_draws == 999
    assert args.resume is False
    assert args.analyze is False
