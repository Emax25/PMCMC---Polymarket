"""Tests for the no-lookahead event study (`src.analysis.event_study`).

The unit under test is a *pre-registered* procedure, so these tests pin the
pre-registration as much as the arithmetic:

  * the primary statistic is mean ``P(Z)`` elevation over
    ``[t_close - W, t_close - w)`` against a **within-market time-shifted**
    permutation null, with an add-one p-value (R6);
  * ``(W, w)`` is fixed by synthetic calibration *before* any real-data run, and
    anything else is labelled exploratory in the output (KTD4);
  * replay provenance is a gate, not a caveat — a scores file that cannot be
    shown to come from ``score_stream.py --replay`` is refused;
  * no Kendall-tau criterion exists anywhere in this feature.

The two tests at the bottom are the statistical ones: power against a planted
late burst, and rough uniformity of the null arm's p-values across repeated
generation. They drive the real generator and streaming scorer, so they are the
slowest here — but at ~2 s for 25 simulated markets they are nowhere near the
`slow` marker's "long-running inference" bar, and they stay in the fast suite
where the claims they guard are actually checked. Everything above them runs on
hand-built score paths in milliseconds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts import event_study as event_study_cli
from scripts import score_stream
from src.analysis.event_study import (
    ARM_NULL,
    ARM_PLANTED,
    DAY_SECONDS,
    LOCKED_EMBARGO_S,
    LOCKED_WINDOW_S,
    REASON_EMPTY_WINDOW,
    REASON_HISTORY_TOO_SHORT,
    REASON_NO_RESOLUTION,
    REASON_NO_TRADES_BEFORE_CLOSE,
    REASON_TOO_FEW_NULL,
    ExcludedMarket,
    MarketScores,
    ProvenanceError,
    WindowSpec,
    analyze_market,
    load_resolutions,
    load_scores,
    read_replay_provenance,
    run_event_study,
    simulate_market_scores,
    write_summary,
)

T0 = 1_700_000_000.0
HOUR = 3600.0

# The locked window, as every test that is not explicitly exploring uses it.
LOCKED = WindowSpec()


# ---------------- Hand-built score paths ----------------


def _flat_scores(
    market: str = "m0",
    *,
    n: int = 600,
    gap: float = HOUR,
    level: float = 0.05,
    seed: int = 0,
) -> MarketScores:
    """A market with no timing signal: ``p_z`` jitters around ``level``.

    600 hourly trades span 25 days, so the locked 5-day window leaves 20 days of
    earlier history for the time-shift null to place comparison windows in.
    """
    rng = np.random.default_rng(seed)
    ts = T0 + gap * np.arange(n, dtype=float)
    p_z = np.clip(rng.normal(level, 0.01, n), 0.0, 1.0)
    x_mean = np.cumsum(rng.normal(0.0, 0.01, n))
    return MarketScores(market=market, ts=ts, p_z=p_z, x_mean=x_mean)


def _plant(
    scores: MarketScores,
    close_ts: float,
    *,
    window: WindowSpec = LOCKED,
    level: float = 0.6,
) -> MarketScores:
    """Raise ``p_z`` to ``level`` inside the event window and nowhere else."""
    inside = (scores.ts >= close_ts - window.W) & (scores.ts < close_ts - window.w)
    p_z = scores.p_z.copy()
    p_z[inside] = level
    return replace(scores, p_z=p_z)


def _write_scores(path: Path, markets: dict[str, MarketScores]) -> Path:
    """Write per-market scores as a `score_stream.py`-shaped JSONL."""
    lines = []
    for market, scores in markets.items():
        for i in range(scores.n):
            lines.append(
                json.dumps(
                    {
                        "ts": float(scores.ts[i]),
                        "tx_hash": f"0x{i:06d}",
                        "market": market,
                        "wallet": "0xw0",
                        "p_z": float(scores.p_z[i]),
                        "p_v": 0.1,
                        "x_mean": float(scores.x_mean[i]),
                    },
                ),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _write_sidecar(scores_path: Path, mode: str = "replay") -> Path:
    """Write a `score_stream.py`-shaped provenance sidecar next to a scores file."""
    sidecar = scores_path.with_name(scores_path.name + ".meta.json")
    sidecar.write_text(
        json.dumps({"mode": mode, "input": "capture.jsonl"}),
        encoding="utf-8",
    )
    return sidecar


# ---------------- WindowSpec / KTD4 ordering ----------------


def test_window_spec_rejects_a_window_that_cannot_hold_a_trade():
    """W <= w leaves [t_close - W, t_close - w) empty, and a negative w is nonsense."""
    with pytest.raises(ValueError, match="must exceed the embargo"):
        WindowSpec(W=DAY_SECONDS, w=DAY_SECONDS)
    with pytest.raises(ValueError, match="non-negative"):
        WindowSpec(W=5 * DAY_SECONDS, w=-1.0)


def test_default_window_is_the_calibrated_locked_pair():
    """The default (W, w) is the KTD4-locked 5 d / 1 d, and says so."""
    assert LOCKED.W == 5.0 * DAY_SECONDS == LOCKED_WINDOW_S
    assert LOCKED.w == 1.0 * DAY_SECONDS == LOCKED_EMBARGO_S
    assert LOCKED.is_locked
    assert LOCKED.to_dict()["locked"] is True
    assert LOCKED.to_dict()["W_days"] == pytest.approx(5.0)


def test_non_locked_window_is_flagged_exploratory_in_output_and_log(caplog):
    """KTD4: a window the calibration did not fix cannot pass as the committed one."""
    scores = _flat_scores()
    close_ts = float(scores.ts[-1])
    window = WindowSpec(W=3.0 * DAY_SECONDS, w=DAY_SECONDS)
    assert not window.is_locked

    with caplog.at_level(logging.WARNING, logger="src.analysis.event_study"):
        summary = run_event_study(
            {"m0": scores},
            {"m0": close_ts},
            window=window,
            n_permutations=49,
        )

    assert summary.to_dict()["window"]["locked"] is False
    assert "NOT the KTD4-locked" in caplog.text
    # The CLI report must say it in words, not only in a JSON boolean.
    assert "NOT LOCKED - EXPLORATORY" in event_study_cli._format_report(summary)


def test_no_kendall_tau_criterion_anywhere_in_the_feature():
    """Hard rule: the event study has no Kendall-tau acceptance criterion.

    The word itself is allowed to appear — the CLI docstring says the criterion
    is deliberately absent — so this looks for the callable, which is the only
    way one could actually be applied.
    """
    from src.analysis import event_study as module

    for path in (Path(module.__file__), Path(event_study_cli.__file__)):
        text = path.read_text(encoding="utf-8").lower()
        assert "kendalltau" not in text
        assert "kendall_tau" not in text


# ---------------- Provenance gate ----------------


def test_read_replay_provenance_matches_the_sidecar_score_stream_writes(tmp_path):
    """Pin the gate against the real writer, not against a remembered constant."""
    capture = tmp_path / "trades.jsonl"
    capture.write_text(
        "".join(
            json.dumps(
                {
                    "timestamp": T0 + 3 * i,
                    "price": 0.4 + 0.01 * i,
                    "size": 12.0,
                    "wallet": f"0xw{i % 3}",
                    "side": "BUY",
                    "transaction_hash": f"0x{i:06d}",
                    "condition_id": "0xcond",
                },
            )
            + "\n"
            for i in range(6)
        ),
        encoding="utf-8",
    )
    out = tmp_path / "scores.jsonl"
    assert (
        score_stream.main(
            ["--replay", str(capture), "--output", str(out), "--log-level", "WARNING"],
        )
        == 0
    )

    payload = read_replay_provenance(out)

    assert payload["mode"] == "replay"
    assert payload["input"] == str(capture)
    # And the scores that sidecar describes load through the study's own reader.
    loaded = load_scores(out)
    assert set(loaded) == {"0xcond"}
    assert loaded["0xcond"].n == 6


def test_read_replay_provenance_refuses_missing_live_and_malformed(tmp_path):
    """Every way of failing to prove replay provenance is refused, not caveated."""
    scores = _write_scores(tmp_path / "s.jsonl", {"m0": _flat_scores(n=10)})

    with pytest.raises(ProvenanceError, match="no s.jsonl.meta.json"):
        read_replay_provenance(scores)

    _write_sidecar(scores, mode="live")
    with pytest.raises(ProvenanceError, match="mode='live'"):
        read_replay_provenance(scores)

    scores.with_name("s.jsonl.meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="not readable JSON"):
        read_replay_provenance(scores)


def test_cli_refuses_scores_without_the_replay_flag(tmp_path):
    """Prefix invariance is inherited from replay, so the CLI demands the flag."""
    scores = _write_scores(tmp_path / "s.jsonl", {"m0": _flat_scores(n=10)})
    resolutions = tmp_path / "res.json"
    resolutions.write_text(json.dumps({"m0": T0}), encoding="utf-8")
    argv = [
        "--scores",
        str(scores),
        "--resolutions",
        str(resolutions),
        "--json-out",
        str(tmp_path / "summary.json"),
        "--no-figures",
        "--log-level",
        "ERROR",
    ]

    assert event_study_cli.main(argv) == 2  # no sidecar at all

    _write_sidecar(scores, mode="live")
    assert event_study_cli.main(argv) == 2  # live-mode scores

    assert not (tmp_path / "summary.json").exists()


# ---------------- Loading ----------------


def test_load_scores_groups_by_market_sorts_and_rejects_malformed(tmp_path):
    """Records group and time-sort; a bad line names itself and stops the run."""
    path = tmp_path / "s.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"ts": 20.0, "market": "a", "p_z": 0.2, "x_mean": 1.0}),
                "",
                json.dumps({"ts": 10.0, "market": "a", "p_z": 0.1, "x_mean": 0.0}),
                json.dumps({"ts": 15.0, "market": "b", "p_z": 0.9, "x_mean": 2.0}),
            ],
        ),
        encoding="utf-8",
    )

    loaded = load_scores(path)

    assert set(loaded) == {"a", "b"}
    assert loaded["a"].ts.tolist() == [10.0, 20.0]
    assert loaded["a"].p_z.tolist() == pytest.approx([0.1, 0.2])
    assert loaded["b"].n == 1

    path.write_text('{"ts": 1.0,\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        load_scores(path)


def test_load_scores_on_an_empty_file_is_empty_not_an_error(tmp_path):
    """An empty capture is a legitimate (if useless) input; the CLI reports it."""
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert load_scores(path) == {}

    _write_sidecar(path)
    resolutions = tmp_path / "res.json"
    resolutions.write_text(json.dumps({"m0": T0}), encoding="utf-8")
    assert (
        event_study_cli.main(
            [
                "--scores",
                str(path),
                "--resolutions",
                str(resolutions),
                "--json-out",
                str(tmp_path / "summary.json"),
                "--no-figures",
                "--log-level",
                "ERROR",
            ],
        )
        == 3
    )


def test_load_resolutions_accepts_mapping_array_and_sidecar_directory(tmp_path):
    """All three shapes the pull/preprocess steps leave behind resolve to t_close."""
    mapping = tmp_path / "map.json"
    mapping.write_text(
        json.dumps({"a": 1000.0, "b": {"close_ts": 2000.0}}),
        encoding="utf-8",
    )
    assert load_resolutions(mapping) == {"a": 1000.0, "b": 2000.0}

    array = tmp_path / "arr.json"
    array.write_text(
        json.dumps([{"ticker": "K1", "close_time": "2026-01-02T03:04:05Z"}]),
        encoding="utf-8",
    )
    assert list(load_resolutions(array)) == ["K1"]

    directory = tmp_path / "processed"
    directory.mkdir()
    (directory / "mkt.meta.json").write_text(
        json.dumps({"condition_id": "0xabc", "end_date": "2026-03-04"}),
        encoding="utf-8",
    )
    assert list(load_resolutions(directory)) == ["0xabc"]

    with pytest.raises(FileNotFoundError):
        load_resolutions(tmp_path / "nope.json")


def test_load_resolutions_reports_a_malformed_sidecar_by_path(tmp_path):
    """A directory of sidecars fails the same documented way a single file does."""
    directory = tmp_path / "processed"
    directory.mkdir()
    (directory / "bad.meta.json").write_text("{oops", encoding="utf-8")

    with pytest.raises(ValueError, match="bad.meta.json: malformed JSON"):
        load_resolutions(directory)


def test_load_resolutions_drops_records_with_no_parseable_close_time(tmp_path, caplog):
    """A record with no close time is dropped loudly, never guessed at."""
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"a": {"slug": "no-dates"}}), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="src.analysis.event_study"):
        assert load_resolutions(path) == {}

    assert "carries no close time" in caplog.text


# ---------------- The primary statistic ----------------


def test_planted_window_elevation_beats_the_time_shift_null():
    """Power, deterministically: a burst confined to the window hits the floor p."""
    scores = _flat_scores()
    close_ts = float(scores.ts[-1])
    planted = _plant(scores, close_ts)

    result = analyze_market(
        planted,
        close_ts,
        window=LOCKED,
        n_permutations=199,
        rng=np.random.default_rng(0),
    )

    assert not isinstance(result, ExcludedMarket)
    assert result.n_window == 96  # 4 days of hourly trades
    assert result.elevation > 0.4
    # Nothing in the earlier history reaches the burst, so the add-one p-value
    # sits at its floor: 1 / (1 + n_null), never 0.
    assert result.p_value == pytest.approx(1.0 / (1 + result.n_null))
    assert result.p_value > 0.0
    # z is a reported effect size, not the test, and it understates the
    # separation here: shifts smaller than W - w slide the null window over part
    # of the burst, which widens the null's spread. That direction is
    # conservative — a contaminated null can only make the p-value larger.
    assert result.z_score > 3.0


def test_flat_scores_give_a_p_value_at_the_top_of_the_range():
    """No timing signal: the observed window is indistinguishable from any shift."""
    n = 600
    ts = T0 + HOUR * np.arange(n, dtype=float)
    constant = MarketScores(
        market="m0",
        ts=ts,
        p_z=np.full(n, 0.2),
        x_mean=np.zeros(n),
    )

    result = analyze_market(
        constant,
        float(ts[-1]),
        window=LOCKED,
        n_permutations=99,
        rng=np.random.default_rng(1),
    )

    assert not isinstance(result, ExcludedMarket)
    # Every shifted window has exactly the observed mean, so all of them tie.
    assert result.p_value == 1.0
    assert result.elevation == pytest.approx(0.0)


def test_the_null_is_shifted_within_the_market_not_pooled_across_markets():
    """R6: the null must be this market's own history, so a rich neighbour
    cannot move its p-value."""
    lonely = _plant(_flat_scores("m0", seed=3), T0 + HOUR * 599)
    close = {"m0": T0 + HOUR * 599, "m1": T0 + HOUR * 599}
    loud = _flat_scores("m1", seed=4, level=0.95)

    alone = run_event_study({"m0": lonely}, close, n_permutations=99, seed=7)
    with_neighbour = run_event_study(
        {"m0": lonely, "m1": loud},
        close,
        n_permutations=99,
        seed=7,
    )

    m0_alone = alone.results[0]
    m0_paired = next(r for r in with_neighbour.results if r.market == "m0")
    assert m0_paired.p_value == m0_alone.p_value
    assert m0_paired.elevation == pytest.approx(m0_alone.elevation)
    # Only the labelled cross-market robustness variant sees the neighbour.
    assert m0_alone.p_value_cross is None
    assert m0_paired.p_value_cross is not None


def test_robustness_variants_are_labelled_and_never_lead_the_output():
    """The max and cross-market variants live under `robustness`, with a note."""
    scores = _plant(_flat_scores(), T0 + HOUR * 599)
    summary = run_event_study(
        {"m0": scores, "m1": _flat_scores("m1", seed=9)},
        {"m0": T0 + HOUR * 599, "m1": T0 + HOUR * 599},
        n_permutations=99,
    )

    payload = summary.to_dict()
    market = payload["markets"][0]
    assert set(market["robustness"]) == {
        "window_max",
        "max_elevation",
        "p_value_max",
        "p_value_cross_market",
    }
    assert "p_value_max" not in market  # variants never sit at the top level
    assert "robustness checks" in payload["robustness_note"]
    assert "not independent confirmation" in payload["robustness_note"]
    assert "replay" in payload["no_lookahead_note"]


def test_terminal_move_is_measured_over_the_embargo_only():
    """The move the statistic is claimed to precede is the disjoint tail interval."""
    n = 600
    ts = T0 + HOUR * np.arange(n, dtype=float)
    close_ts = float(ts[-1])
    # Flat until the embargo starts, then a clean +2.0 ramp inside it.
    x_mean = np.zeros(n)
    embargo = ts >= close_ts - LOCKED.w
    x_mean[embargo] = np.linspace(0.0, 2.0, int(embargo.sum()))
    scores = MarketScores("m0", ts, np.full(n, 0.2), x_mean)

    result = analyze_market(
        scores,
        close_ts,
        window=LOCKED,
        n_permutations=49,
        rng=np.random.default_rng(2),
    )

    assert not isinstance(result, ExcludedMarket)
    assert result.n_terminal == 25  # 1 day of hourly trades, close inclusive
    assert result.terminal_move == pytest.approx(2.0)


# ---------------- Exclusions and edge cases ----------------


def test_market_with_no_trades_before_close_is_excluded():
    """A close time before the first score leaves nothing to average."""
    scores = _flat_scores(n=50)
    outcome = analyze_market(
        scores,
        float(scores.ts[0]) - 1.0,
        window=LOCKED,
        n_permutations=49,
        rng=np.random.default_rng(0),
    )
    assert outcome == ExcludedMarket("m0", REASON_NO_TRADES_BEFORE_CLOSE)


def test_market_with_zero_trades_inside_the_window_is_excluded():
    """Trading stopped well before close: the window exists but catches nothing."""
    scores = _flat_scores()
    outcome = analyze_market(
        scores,
        float(scores.ts[-1]) + 10.0 * DAY_SECONDS,
        window=LOCKED,
        n_permutations=49,
        rng=np.random.default_rng(0),
    )
    assert outcome == ExcludedMarket("m0", REASON_EMPTY_WINDOW)


def test_window_longer_than_the_market_history_is_excluded():
    """W beyond the first trade leaves no room to place a single shifted window."""
    scores = _flat_scores(n=48)  # 2 days of history, against a 5-day window
    outcome = analyze_market(
        scores,
        float(scores.ts[-1]),
        window=LOCKED,
        n_permutations=49,
        rng=np.random.default_rng(0),
    )
    assert outcome == ExcludedMarket("m0", REASON_HISTORY_TOO_SHORT)


def test_market_with_too_few_usable_shifted_windows_is_excluded():
    """A long silent gap makes most placements empty; a null that thin is refused."""
    close_ts = T0 + 40.0 * DAY_SECONDS
    # One ancient trade, then the whole book inside the event window.
    ts = np.concatenate(
        [
            [close_ts - 40.0 * DAY_SECONDS],
            np.linspace(close_ts - LOCKED.W, close_ts - LOCKED.w, 100, endpoint=False),
        ],
    )
    scores = MarketScores("m0", ts, np.full(ts.size, 0.3), np.zeros(ts.size))

    outcome = analyze_market(
        scores,
        close_ts,
        window=LOCKED,
        n_permutations=40,
        rng=np.random.default_rng(0),
    )

    assert outcome == ExcludedMarket("m0", REASON_TOO_FEW_NULL)


def test_market_without_resolution_metadata_is_excluded_warned_and_counted(caplog):
    """Resolution join: no t_close means no window, and the JSON says how many."""
    scores = {"m0": _flat_scores("m0"), "m1": _flat_scores("m1", seed=5)}

    with caplog.at_level(logging.WARNING, logger="src.analysis.event_study"):
        summary = run_event_study(
            scores,
            {"m0": float(scores["m0"].ts[-1])},  # m1 deliberately absent
            n_permutations=49,
        )

    assert "no resolution metadata" in caplog.text
    assert [row.market for row in summary.results] == ["m0"]
    assert summary.excluded == [ExcludedMarket("m1", REASON_NO_RESOLUTION)]
    payload = summary.to_dict()
    assert payload["exclusion_counts"] == {REASON_NO_RESOLUTION: 1}
    assert payload["n_markets"] == 2
    assert payload["n_analysed"] == 1
    assert payload["n_excluded"] == 1


def test_zero_permutations_is_rejected_and_one_is_allowed():
    """The permutation null *is* the test, so it cannot be switched off."""
    scores = _plant(_flat_scores(), T0 + HOUR * 599)
    close = {"m0": T0 + HOUR * 599}

    with pytest.raises(ValueError, match="n_permutations must be at least 1"):
        run_event_study({"m0": scores}, close, n_permutations=0)

    summary = run_event_study({"m0": scores}, close, n_permutations=1)
    row = summary.results[0]
    assert row.n_null == 1
    # One draw admits only two p-values, and neither of them is 0.
    assert row.p_value in (0.5, 1.0)
    assert np.isnan(row.z_score)  # no spread from a single null draw


def test_fisher_combination_needs_at_least_two_markets():
    """One market's p-value dressed up as a study is not a combined result."""
    scores = _flat_scores()
    close_ts = float(scores.ts[-1])
    one = run_event_study({"m0": scores}, {"m0": close_ts}, n_permutations=49)
    assert one.fisher_p is None and one.fisher_stat is None

    two = run_event_study(
        {"m0": scores, "m1": _flat_scores("m1", seed=6)},
        {"m0": close_ts, "m1": close_ts},
        n_permutations=49,
    )
    assert two.fisher_p is not None
    assert 0.0 < two.fisher_p <= 1.0


# ---------------- Reproducibility ----------------


def test_a_market_p_value_does_not_move_when_another_market_is_added():
    """Streams key on the market id, so the study replays across market sets."""
    scores = {
        "aaa": _flat_scores("aaa", seed=11),
        "bbb": _flat_scores("bbb", seed=12),
    }
    close = {m: float(s.ts[-1]) for m, s in scores.items()}
    close["ccc"] = float(scores["aaa"].ts[-1])

    before = run_event_study(scores, close, n_permutations=99, seed=3)
    scores["ccc"] = _flat_scores("ccc", seed=13)
    after = run_event_study(scores, close, n_permutations=99, seed=3)

    by_market = {row.market: row.p_value for row in after.results}
    for row in before.results:
        assert by_market[row.market] == row.p_value


def test_write_summary_round_trips_with_extra_provenance(tmp_path):
    """The summary JSON is self-describing and carries its inputs."""
    scores = _flat_scores()
    summary = run_event_study(
        {"m0": scores},
        {"m0": float(scores.ts[-1])},
        n_permutations=49,
        provenance={"mode": "replay", "input": "capture.jsonl"},
    )

    path = write_summary(
        summary,
        tmp_path / "out" / "summary.json",
        extra={"scores": "s"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["provenance"]["mode"] == "replay"
    assert payload["scores"] == "s"
    assert payload["window"]["locked"] is True


# ---------------- Statistical behaviour (slow) ----------------


@pytest.fixture(scope="module")
def simulated_arms():
    """Score 20 null-arm and 5 planted-arm synthetic markets once, for reuse.

    These come from the same generator + streaming scorer the KTD4 calibration
    uses, so the two tests below are a small replication of that evidence rather
    than a second, differently-shaped synthetic.
    """
    return {
        arm: [
            simulate_market_scores(
                f"{arm[0]}{i:03d}",
                arm=arm,
                rng=np.random.default_rng([4242, i, 0 if arm == ARM_NULL else 1]),
            )
            for i in range(n)
        ]
        for arm, n in ((ARM_NULL, 20), (ARM_PLANTED, 5))
    }


def _p_values(cases, *, seed: int) -> list[float]:
    """Primary p-values for a list of ``(scores, close_ts)`` simulation outputs."""
    out = []
    for i, (scores, close_ts) in enumerate(cases):
        result = analyze_market(
            scores,
            close_ts,
            window=LOCKED,
            n_permutations=199,
            rng=np.random.default_rng([seed, i]),
        )
        assert not isinstance(result, ExcludedMarket), result
        out.append(result.p_value)
    return out


def test_planted_insider_burst_is_detected_at_the_locked_window(simulated_arms):
    """Power: a late insider segment clears the within-market null every time.

    The calibration measured 60/60 detections at W = 5 d; five replicates here
    is a regression guard on that, not a re-measurement of the power curve.
    """
    p_values = _p_values(simulated_arms[ARM_PLANTED], seed=101)
    assert max(p_values) < 0.05


def test_null_arm_p_values_are_roughly_uniform(simulated_arms):
    """Size: with no planted signal the primary p-value carries no information.

    Coarse by design — 20 replicates cannot resolve a few points of size
    inflation, so this asserts the shape (spread out, few rejections, mean near
    1/2) that a badly-centred null would break loudly.
    """
    p_values = np.asarray(_p_values(simulated_arms[ARM_NULL], seed=202))

    assert float(np.mean(p_values < 0.05)) <= 0.15  # nominal 0.05 on 20 draws
    assert 0.25 < float(p_values.mean()) < 0.75
    assert float(p_values.min()) < 0.5 < float(p_values.max())
