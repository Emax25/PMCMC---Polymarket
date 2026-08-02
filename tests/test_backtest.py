"""Tests for the costed tradeability check (`src.analysis.backtest`).

The unit under test is a **detection-signal proof of concept, not a validated
alpha strategy**, and these tests pin that framing alongside the arithmetic:

  * the cost model matches Kalshi's ``0.07 * p * (1 - p)`` taker fee — maximal
    at mid-book, zero at the bounds — and charges the half-spread on whichever
    side is taken;
  * the walk-forward purge really does remove every training label window that
    overlaps a test block, including the embargo;
  * the deflated Sharpe uses the **empirical variance across the trial
    Sharpes**, not the raw trial count, and is monotone in the number of trials
    at fixed variance;
  * the degenerate cases a small-sample PoC will actually hit — zero trades, all
    winners, all losers, a single fold, zero trial-Sharpe variance — return a
    well-formed "undefined" rather than a number or a crash;
  * every artifact carries the PoC framing, and the replay-provenance gate is
    inherited unweakened from the event study.

Everything here runs on hand-built score paths in milliseconds; nothing drives
the inference stack.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts import backtest as backtest_cli
from src.analysis.backtest import (
    DECLARED_THRESHOLD_GRID,
    EULER_GAMMA,
    KALSHI_FEE_RATE,
    REASON_NO_OUTCOME,
    REASON_NO_RESOLUTION,
    SIDE_NO,
    SIDE_YES,
    BacktestSummary,
    CostModel,
    MarketPanel,
    build_panels,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    load_outcomes,
    open_position,
    probabilistic_sharpe_ratio,
    purged_walk_forward,
    run_backtest,
    sharpe_ratio,
    write_summary,
)
from src.analysis.event_study import MarketScores

T0 = 1_700_000_000.0
DAY = 86400.0


# ---------------- Fixtures ----------------


def _panel(
    market: str,
    *,
    p_z: list[float],
    price: list[float],
    outcome: float,
    t_start: float = T0,
    gap: float = 3600.0,
    close_ts: float | None = None,
) -> MarketPanel:
    """Build one hand-specified market panel."""
    ts = t_start + gap * np.arange(len(p_z), dtype=float)
    return MarketPanel(
        market=market,
        ts=ts,
        p_z=np.asarray(p_z, dtype=float),
        price=np.asarray(price, dtype=float),
        close_ts=float(ts[-1] + gap) if close_ts is None else close_ts,
        outcome=float(outcome),
    )


def _sequential_panels(
    n_markets: int,
    *,
    outcomes: list[float] | None = None,
    span: float = 5.0 * DAY,
    stride: float = 10.0 * DAY,
    prices: list[float] | None = None,
    n_trades: int = 6,
) -> list[MarketPanel]:
    """Build ``n_markets`` non-overlapping markets, one every ``stride``.

    Scores ramp from 0.4 to 0.95 inside each market so every declared threshold
    finds an entry, and the markets are laid out end to end so the purge has
    something to keep as well as something to drop.

    Entry prices vary across markets by default. A constant price would give
    every position an identical return, whose standard deviation is zero and
    whose Sharpe is therefore undefined — a real degenerate case (pinned
    separately) but not the one most of these tests are about.
    """
    panels = []
    for k in range(n_markets):
        start = T0 + k * stride
        gap = span / n_trades
        outcome = 1.0 if outcomes is None else outcomes[k]
        price = 0.55 + 0.02 * (k % 8) if prices is None else prices[k]
        panels.append(
            _panel(
                f"m{k:02d}",
                p_z=list(np.linspace(0.4, 0.95, n_trades)),
                price=[price] * n_trades,
                outcome=outcome,
                t_start=start,
                gap=gap,
            ),
        )
    return panels


# ---------------- Cost model ----------------


def test_taker_fee_is_maximal_at_a_half_and_zero_at_the_bounds():
    """Kalshi's fee peaks mid-book, exactly where the insider signal is weakest."""
    cost = CostModel(spread=0.0, fee_rate=KALSHI_FEE_RATE)
    assert cost.taker_fee(0.0) == 0.0
    assert cost.taker_fee(1.0) == 0.0
    assert cost.taker_fee(0.5) == pytest.approx(0.25 * KALSHI_FEE_RATE)

    grid = np.linspace(0.0, 1.0, 101)
    fees = np.asarray([cost.taker_fee(float(p)) for p in grid])
    assert fees.max() == pytest.approx(cost.taker_fee(0.5))
    assert int(np.argmax(fees)) == 50
    # Symmetric in p: the same fee is charged on the YES and the NO leg.
    assert fees == pytest.approx(fees[::-1])


def test_taker_fee_dollars_matches_the_kalshi_adapters_worked_example():
    """``ceil(0.07 * C * p * (1 - p))`` billed at cent granularity.

    The adapter's "~1.75c per contract at p = 0.5" is the *unrounded* rate; what
    the venue bills on one contract is that rounded up to 2c, and on 100
    contracts exactly $1.75. Pinning both is what stops the rounding from being
    silently applied at dollar granularity, where a one-contract fee would be
    a dollar.
    """
    cost = CostModel()
    assert cost.taker_fee(0.5) == pytest.approx(0.0175)
    assert cost.taker_fee_dollars(0.5, 1) == pytest.approx(0.02)
    assert cost.taker_fee_dollars(0.5, 100) == pytest.approx(1.75)
    # Rounding is *up*: 0.07 * 0.01 * 0.99 = 0.000693 -> one cent, never zero.
    assert cost.taker_fee_dollars(0.01, 1) == pytest.approx(0.01)


def test_entry_cost_applies_the_half_spread_on_whichever_side_is_taken():
    """Both legs pay up from mid; neither is rebated the spread."""
    cost = CostModel(spread=0.02, fee_rate=KALSHI_FEE_RATE)
    p = 0.7
    yes = cost.entry_cost(p, SIDE_YES)
    no = cost.entry_cost(p, SIDE_NO)

    assert yes == pytest.approx(p + 0.01 + cost.taker_fee(p))
    assert no == pytest.approx(1.0 - p + 0.01 + cost.taker_fee(p))
    assert yes > p and no > 1.0 - p
    # Buying both legs costs par plus the full spread plus two fees, which is
    # the arbitrage-free statement that the taker never collects the spread.
    assert yes + no == pytest.approx(1.0 + cost.spread + 2 * cost.taker_fee(p))


def test_cost_model_rejects_a_negative_spread_or_fee():
    """A negative cost would be a rebate the venue does not pay."""
    with pytest.raises(ValueError, match="spread"):
        CostModel(spread=-0.01)
    with pytest.raises(ValueError, match="fee_rate"):
        CostModel(fee_rate=-1.0)
    with pytest.raises(ValueError, match="side"):
        CostModel().entry_cost(0.5, "maybe")


# ---------------- Entry rule ----------------


def test_open_position_enters_on_the_first_crossing_and_takes_the_favoured_side():
    """The pre-declared rule: first ``p_z >= tau``, side the price favours."""
    cost = CostModel(spread=0.0, fee_rate=0.0)
    panel = _panel("m", p_z=[0.1, 0.8, 0.95], price=[0.3, 0.7, 0.2], outcome=1.0)

    position = open_position(panel, 0.5, cost)
    assert position is not None
    assert position.entry_ts == panel.ts[1]
    assert position.side == SIDE_YES  # price 0.7 favours YES
    assert position.payout == 1.0
    assert position.ret == pytest.approx(1.0 - 0.7)

    # A threshold nothing crosses takes no position at all.
    assert open_position(panel, 0.99, cost) is None


def test_open_position_takes_the_no_side_below_a_half_and_prices_it_correctly():
    """Below 0.5 the favoured side is NO, and the payout flips with the outcome."""
    cost = CostModel(spread=0.0, fee_rate=0.0)
    panel = _panel("m", p_z=[0.9], price=[0.25], outcome=0.0)
    position = open_position(panel, 0.5, cost)
    assert position.side == SIDE_NO
    assert position.cost == pytest.approx(0.75)
    assert position.payout == 1.0  # outcome 0 -> the NO leg settles at $1
    assert position.ret == pytest.approx(0.25)


def test_open_position_skips_a_contract_that_costs_a_dollar_or_more():
    """Paying $1 for a $1-max payout is never done, so no position is booked."""
    cost = CostModel(spread=0.10, fee_rate=KALSHI_FEE_RATE)
    panel = _panel("m", p_z=[0.9], price=[0.97], outcome=1.0)
    assert open_position(panel, 0.5, cost) is None


# ---------------- Purged / embargoed walk-forward ----------------


def test_no_training_label_window_overlaps_a_test_label_window():
    """The purge property, on a constructed fixture with deliberate overlap.

    Every third market is a long one whose label window runs into later folds;
    an unpurged split would leak its resolution into the training block.
    """
    starts, ends = [], []
    for k in range(12):
        start = T0 + k * 5.0 * DAY
        span = (30.0 if k % 3 == 0 else 4.0) * DAY
        starts.append(start)
        ends.append(start + span)
    label_start = np.asarray(starts)
    label_end = np.asarray(ends)

    embargo = 2.0 * DAY
    splits = purged_walk_forward(
        label_start, label_end, n_splits=3, embargo_s=embargo
    )
    assert len(splits) == 3
    assert sum(int(s.n_purged) for s in splits) > 0  # the fixture bites

    for split in splits:
        assert split.test.size > 0
        assert not set(split.train.tolist()) & set(split.test.tolist())
        for i in split.train:
            # Purge: the training label closes before the test span opens...
            assert label_end[i] <= split.test_start - embargo
            # ...which means it overlaps no test label window at all.
            for j in split.test:
                assert label_end[i] < label_start[j]


def test_a_single_fold_split_is_well_formed():
    """The degenerate ``n_splits=1`` case still trains before it tests."""
    label_start = T0 + np.arange(6, dtype=float) * 10.0 * DAY
    label_end = label_start + 2.0 * DAY
    splits = purged_walk_forward(label_start, label_end, n_splits=1, embargo_s=0.0)
    assert len(splits) == 1
    split = splits[0]
    assert split.train.size + split.test.size == 6
    assert label_end[split.train].max() <= split.test_start


def test_purged_walk_forward_validates_its_arguments_and_handles_no_data():
    """Bad fold counts raise; an empty panel yields no folds rather than an error."""
    label = T0 + np.arange(4, dtype=float) * DAY
    with pytest.raises(ValueError, match="n_splits"):
        purged_walk_forward(label, label, n_splits=0)
    with pytest.raises(ValueError, match="embargo_s"):
        purged_walk_forward(label, label, n_splits=2, embargo_s=-1.0)
    with pytest.raises(ValueError, match="same shape"):
        purged_walk_forward(label, label[:2], n_splits=2)
    assert purged_walk_forward(np.zeros(0), np.zeros(0), n_splits=2) == []


def test_a_long_embargo_can_empty_the_training_block_without_hiding_it():
    """A fold whose training set the purge wipes out is reported, not dropped."""
    label_start = T0 + np.arange(6, dtype=float) * DAY
    label_end = label_start + 0.5 * DAY
    splits = purged_walk_forward(
        label_start, label_end, n_splits=2, embargo_s=365.0 * DAY
    )
    assert splits, "folds must still be returned"
    assert all(split.train.size == 0 for split in splits)
    assert all(split.n_purged > 0 for split in splits)


# ---------------- Sharpe / PSR / DSR ----------------


def test_sharpe_is_undefined_rather_than_infinite_on_degenerate_returns():
    """No trades, one trade, or zero dispersion all give NaN, never +inf."""
    assert math.isnan(sharpe_ratio(np.zeros(0)))
    assert math.isnan(sharpe_ratio(np.asarray([0.3])))
    assert math.isnan(sharpe_ratio(np.full(10, 0.05)))
    assert sharpe_ratio(np.asarray([0.1, -0.1, 0.2, -0.2])) == pytest.approx(
        np.mean([0.1, -0.1, 0.2, -0.2]) / np.std([0.1, -0.1, 0.2, -0.2], ddof=1)
    )


def test_expected_max_sharpe_uses_the_trial_variance_not_the_count_alone():
    """The deflator scales with ``sqrt(Var[trial Sharpes])``.

    This is the review finding the unit exists to get right: halving the spread
    of the trial Sharpes must halve ``SR0``. A count-only deflator would return
    the same number for both.
    """
    n_trials = 20
    wide = expected_max_sharpe(0.04, n_trials)
    narrow = expected_max_sharpe(0.01, n_trials)
    assert wide == pytest.approx(2.0 * narrow)

    # And it is the published closed form, not an approximation of it.
    from scipy.stats import norm

    expected = math.sqrt(0.04) * (
        (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    assert wide == pytest.approx(expected)


def test_expected_max_sharpe_is_monotone_in_the_trial_count():
    """More trials searched -> a higher bar, at fixed trial dispersion."""
    benchmarks = [expected_max_sharpe(0.02, n) for n in (2, 3, 5, 10, 50, 500)]
    assert all(b < a for b, a in zip(benchmarks, benchmarks[1:]))
    assert expected_max_sharpe(0.02, 1) == 0.0  # nothing was selected


def test_deflated_sharpe_is_monotone_decreasing_in_the_trial_count():
    """The DSR itself falls as the disclosed trial count grows."""
    rng = np.random.default_rng(0)
    returns = 0.05 + 0.2 * rng.standard_normal(200)
    trials = list(0.1 + 0.05 * rng.standard_normal(8))

    dsrs = [
        deflated_sharpe_ratio(returns, trials, n_trials=n).dsr
        for n in (2, 5, 20, 100, 1000)
    ]
    assert all(math.isfinite(d) for d in dsrs)
    assert all(lo > hi for lo, hi in zip(dsrs, dsrs[1:]))
    # Deflation only ever costs you: the undeflated PSR is the ceiling.
    psr = probabilistic_sharpe_ratio(returns, 0.0)
    assert all(d <= psr + 1e-12 for d in dsrs)


def test_zero_trial_variance_leaves_nothing_to_deflate():
    """Identical trial Sharpes mean the search selected nothing.

    ``SR0`` is then 0 and the DSR collapses onto the undeflated PSR — the
    honest answer, since no trial was preferred over any other.
    """
    rng = np.random.default_rng(1)
    returns = 0.05 + 0.2 * rng.standard_normal(120)

    identical = deflated_sharpe_ratio(returns, [0.3] * 12)
    # `approx`, not `== 0.0`: the sample variance of twelve identical floats is
    # ~1e-33 rather than bit-zero, and the resulting SR0 is ~1e-16 — negligible
    # against any Sharpe, which is why no snapping threshold is needed.
    assert identical.trial_variance == pytest.approx(0.0, abs=1e-24)
    assert identical.sr_benchmark == pytest.approx(0.0, abs=1e-12)
    assert identical.dsr == pytest.approx(probabilistic_sharpe_ratio(returns, 0.0))

    # A single trial is the same situation seen from the other side.
    single = deflated_sharpe_ratio(returns, [0.3])
    assert single.n_trials == 1
    assert single.sr_benchmark == 0.0


def test_deflated_sharpe_counts_empty_trials_but_excludes_them_from_the_variance():
    """A searched-and-empty threshold still counts toward the disclosed trials."""
    rng = np.random.default_rng(2)
    returns = 0.05 + 0.2 * rng.standard_normal(80)
    result = deflated_sharpe_ratio(returns, [0.1, 0.4, math.nan, math.nan])
    assert result.n_trials == 4
    assert result.n_trial_sharpes == 2
    assert result.trial_variance == pytest.approx(np.var([0.1, 0.4], ddof=1))


def test_deflated_sharpe_of_an_empty_return_stream_is_undefined_not_zero():
    """A strategy that took no trade has no Sharpe and no DSR."""
    result = deflated_sharpe_ratio(np.zeros(0), [0.1, 0.2, 0.3])
    assert result.n_returns == 0
    assert math.isnan(result.sharpe)
    assert math.isnan(result.dsr)
    assert math.isnan(result.psr_zero)


# ---------------- End-to-end run ----------------


def test_a_strategy_that_takes_no_trade_reports_cleanly():
    """Every threshold above the scores' ceiling -> zero positions, no crash."""
    panels = _sequential_panels(8)
    summary = run_backtest(panels, thresholds=(0.99, 0.995), n_splits=2)
    assert summary.returns.size == 0
    assert all(row.n_positions == 0 for row in summary.trials)
    assert math.isnan(summary.deflated.dsr)
    assert all(fold.threshold is None for fold in summary.folds)
    # The grid is still disclosed in full, so the trial count is not silently 0.
    assert summary.to_dict()["n_trials_disclosed"] == 2


def test_an_all_winning_panel_books_positive_costed_returns():
    """Buy the favourite, every market resolves YES: edge survives the costs."""
    panels = _sequential_panels(12, outcomes=[1.0] * 12)
    summary = run_backtest(panels, n_splits=3)
    returns = summary.returns
    assert returns.size > 0
    assert np.all(returns > 0.0)
    cost = CostModel()
    for position in (p for fold in summary.folds for p in fold.positions):
        assert position.side == SIDE_YES
        assert position.ret == pytest.approx(
            1.0 - position.price - cost.half_spread - cost.taker_fee(position.price),
        )


def test_an_all_losing_panel_books_negative_costed_returns():
    """The mirror image: the favourite loses every time, so only costs remain."""
    panels = _sequential_panels(12, outcomes=[0.0] * 12)
    summary = run_backtest(panels, n_splits=3)
    returns = summary.returns
    assert returns.size > 0
    assert np.all(returns < 0.0)
    assert summary.to_dict()["out_of_sample"]["total_return"] < 0.0
    for position in (p for fold in summary.folds for p in fold.positions):
        assert position.payout == 0.0
        assert position.ret == pytest.approx(-position.cost)


def test_a_constant_price_panel_has_no_defined_sharpe_and_selects_nothing():
    """Zero-dispersion returns are undefined, not infinitely good.

    Every market entered at the same price with the same outcome pays the same
    return, so no threshold has a defined in-sample Sharpe and no fold can
    select one. The run must say so rather than pick arbitrarily.
    """
    panels = _sequential_panels(12, outcomes=[1.0] * 12, prices=[0.6] * 12)
    summary = run_backtest(panels, n_splits=3)
    assert all(fold.threshold is None for fold in summary.folds)
    assert all(math.isnan(fold.train_sharpe) for fold in summary.folds)
    assert summary.returns.size == 0
    assert math.isnan(summary.deflated.dsr)


def test_run_backtest_is_out_of_sample_and_discloses_the_full_grid():
    """Selection happens on training folds; the disclosed grid is the trial count."""
    rng = np.random.default_rng(3)
    outcomes = list(rng.integers(0, 2, size=24).astype(float))
    panels = _sequential_panels(24, outcomes=outcomes)
    summary = run_backtest(panels, n_splits=4)

    assert summary.thresholds == DECLARED_THRESHOLD_GRID
    assert summary.deflated.n_trials == len(DECLARED_THRESHOLD_GRID)
    assert len(summary.trials) == len(DECLARED_THRESHOLD_GRID)
    assert len(summary.folds) == 4

    # Out-of-sample means every reported position is in a test block.
    for fold, split_markets in zip(summary.folds, _fold_markets(summary, panels)):
        for position in fold.positions:
            assert position.market in split_markets
        if fold.threshold is not None:
            assert fold.threshold in DECLARED_THRESHOLD_GRID


def _fold_markets(summary: BacktestSummary, panels: list[MarketPanel]) -> list[set]:
    """Recompute each fold's test-market ids, for the out-of-sample assertion."""
    splits = purged_walk_forward(
        np.asarray([p.label_start for p in panels]),
        np.asarray([p.close_ts for p in panels]),
        n_splits=summary.n_splits,
        embargo_s=summary.embargo_s,
    )
    return [{panels[int(i)].market for i in split.test} for split in splits]


def test_run_backtest_rejects_an_empty_grid():
    """An empty family would make the disclosed trial count zero."""
    with pytest.raises(ValueError, match="at least one"):
        run_backtest(_sequential_panels(4), thresholds=())


def test_summary_json_carries_the_poc_framing_everywhere_it_matters(tmp_path):
    """The framing is a property of the artifact, not of the terminal output."""
    panels = _sequential_panels(10, outcomes=[1.0, 0.0] * 5)
    summary = run_backtest(panels, n_splits=2)
    path = write_summary(summary, tmp_path / "out" / "summary.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    framing = payload["framing"].lower()
    assert "not a validated alpha strategy" in framing
    assert "not evidence of alpha" in framing
    assert "nothing in this output is a trading recommendation" in framing
    assert payload["declared_threshold_grid"] == list(DECLARED_THRESHOLD_GRID)
    assert "empirical variance" in payload["deflated_sharpe"]["note"].lower()
    assert "trial count" in payload["deflated_sharpe"]["note"].lower()
    assert "p * (1 - p)" in payload["cost_model"]["note"]
    assert payload["schema_version"] == 1


# ---------------- Inputs ----------------


def test_build_panels_excludes_markets_with_no_close_or_no_outcome():
    """A market without a settled label is counted, never traded against a guess."""
    scores = {
        m: MarketScores(
            market=m,
            ts=T0 + np.arange(4, dtype=float) * 3600.0,
            p_z=np.full(4, 0.8),
            x_mean=np.zeros(4),
        )
        for m in ("ok", "no_close", "no_outcome", "all_after_close")
    }
    closes = {
        "ok": T0 + 10 * 3600.0,
        "no_outcome": T0 + 10 * 3600.0,
        "all_after_close": T0 - 3600.0,
    }
    outcomes = {"ok": 1.0, "no_close": 1.0, "all_after_close": 1.0}

    panels, excluded = build_panels(scores, closes, outcomes)
    assert [p.market for p in panels] == ["ok"]
    assert panels[0].price == pytest.approx(np.full(4, 0.5))  # sigmoid(0)
    reasons = {row.market: row.reason for row in excluded}
    assert reasons["no_close"] == REASON_NO_RESOLUTION
    assert reasons["no_outcome"] == REASON_NO_OUTCOME
    assert "no scored trades" in reasons["all_after_close"]


def test_load_outcomes_reads_the_venues_spellings_and_refuses_a_price(tmp_path):
    """``result: "yes"`` and a 0/1 payout settle; a 0.97 price does not."""
    path = tmp_path / "meta.json"
    path.write_text(
        json.dumps(
            {
                "kalshi": {"ticker": "KX-1", "result": "yes"},
                "poly": {"condition_id": "0xabc", "outcome": 0},
                "unsettled": {"market": "u", "result": None},
                "priced": {"market": "p", "outcome": 0.97},
                "bare": 1,
            },
        ),
        encoding="utf-8",
    )
    outcomes = load_outcomes(path)
    assert outcomes == {"KX-1": 1.0, "0xabc": 0.0, "bare": 1.0}


def test_load_outcomes_accepts_a_sidecar_directory(tmp_path):
    """Same three input shapes as the event study's resolution loader."""
    directory = tmp_path / "processed"
    directory.mkdir()
    (directory / "a.meta.json").write_text(
        json.dumps({"condition_id": "a", "result": "no"}), encoding="utf-8"
    )
    assert load_outcomes(directory) == {"a": 0.0}
    with pytest.raises(FileNotFoundError):
        load_outcomes(tmp_path / "nope.json")


# ---------------- CLI ----------------


def _write_scores(path: Path, panels: list[MarketPanel]) -> Path:
    """Write panels back out as a `score_stream.py` scores JSONL."""
    lines = []
    for panel in panels:
        for i in range(panel.ts.size):
            lines.append(
                json.dumps(
                    {
                        "ts": float(panel.ts[i]),
                        "tx_hash": f"0x{panel.market}{i}",
                        "market": panel.market,
                        "wallet": "w0",
                        "p_z": float(panel.p_z[i]),
                        "p_v": 0.0,
                        "x_mean": 0.0,
                    },
                ),
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_cli_refuses_scores_without_replay_provenance(tmp_path):
    """The no-lookahead gate is inherited from the event study, unweakened."""
    panels = _sequential_panels(4)
    scores = _write_scores(tmp_path / "s.jsonl", panels)
    resolutions = tmp_path / "res.json"
    resolutions.write_text(
        json.dumps(
            {p.market: {"close_ts": p.close_ts, "result": "yes"} for p in panels},
        ),
        encoding="utf-8",
    )
    argv = [
        "--scores",
        str(scores),
        "--resolutions",
        str(resolutions),
        "--json-out",
        str(tmp_path / "summary.json"),
        "--no-figures",
    ]
    assert backtest_cli.main(argv) == 2  # no sidecar at all

    sidecar = scores.with_name(scores.name + ".meta.json")
    sidecar.write_text(json.dumps({"mode": "live"}), encoding="utf-8")
    assert backtest_cli.main(argv) == 2  # live-mode scores
    assert not (tmp_path / "summary.json").exists()


def test_cli_end_to_end_writes_a_framed_summary(tmp_path):
    """The whole path: replay scores in, framed JSON and report out."""
    panels = _sequential_panels(16, outcomes=[1.0, 0.0] * 8)
    scores = _write_scores(tmp_path / "s.jsonl", panels)
    scores.with_name(scores.name + ".meta.json").write_text(
        json.dumps({"mode": "replay", "input": "capture.jsonl"}), encoding="utf-8"
    )
    resolutions = tmp_path / "res.json"
    resolutions.write_text(
        json.dumps(
            {
                p.market: {
                    "market": p.market,
                    "close_ts": p.close_ts,
                    "result": "yes" if p.outcome else "no",
                }
                for p in panels
            },
        ),
        encoding="utf-8",
    )
    json_out = tmp_path / "summary.json"
    code = backtest_cli.main(
        [
            "--scores",
            str(scores),
            "--resolutions",
            str(resolutions),
            "--n-splits",
            "3",
            "--json-out",
            str(json_out),
            "--no-figures",
        ],
    )
    assert code == 0

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["provenance"]["mode"] == "replay"
    assert payload["n_tradeable"] == 16
    assert len(payload["folds"]) == 3
    assert payload["n_trials_disclosed"] == len(DECLARED_THRESHOLD_GRID)
    assert "NOT A VALIDATED ALPHA STRATEGY" in payload["framing"]
    assert payload["walk_forward"]["n_splits"] == 3


def test_cli_report_leads_with_the_poc_framing_and_the_deflated_number():
    """The terminal report cannot be quoted as a strategy result by accident."""
    panels = _sequential_panels(12, outcomes=[1.0, 0.0] * 6)
    summary = run_backtest(panels, n_splits=3)
    report = backtest_cli._format_report(summary)

    assert "DETECTION POC - NOT ALPHA" in report
    assert "NOT A VALIDATED ALPHA STRATEGY" in report
    assert "DEFLATED SHARPE:" in report
    assert "Trials disclosed: 9" in report
    # Every declared threshold is printed, which is what "disclosed" means.
    for tau in DECLARED_THRESHOLD_GRID:
        assert f"{tau:>6.2f}" in report
