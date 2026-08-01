"""End-to-end smoke tests for the four CLI scripts.

All paths exercise the `--synthetic` mode of the runners + the offline
fixtures for pull_data.py, so the suite never hits Polymarket.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from config.default_params import ModelParams, OnlineScorerConfig
from scripts import (
    _runner,
    benchmark,
    eval_c4,
    make_figures,
    pareto,
    pull_data,
    pull_kalshi,
    run_ipmcmc,
    run_pg,
    score_stream,
    stream_trades,
    validate_vem,
)
from src.analysis.validation import PSIS_KHAT_KEY
from src.data import kalshi_api, trade_stream
from src.data.kalshi_api import KalshiAPIError, KalshiMarketMeta
from src.data.polymarket_api import MarketMeta, PolymarketAPIError, RawTrade
from src.inference import stream_scoring
from src.inference.ipmcmc import iPMCMCOutput
from src.inference.online_scorer import OnlineScorer
from src.inference.particle_gibbs import PGOutput
from src.inference.variational_em import VEMOutput
from src.utils.transforms import logit
from tests.test_rtds import LIVE_CONDITION_ID, FakeSocket, make_frame

FIXTURES = Path(__file__).parent / "fixtures"

# Dummy Gamma conditionId reused across the offline market fixtures below.
_COND_ID = "0xaaa000000000000000000000000000000000000000000000000000000000aa01"


# ---------------- _runner helpers ----------------


def test_build_config_dev_default():
    """Dev preset yields its default iteration/burn-in counts."""
    args = _runner.add_common_args.__globals__["argparse"].Namespace(
        config="dev",
        seed=None,
        n_iter=None,
        n_burnin=None,
        n_particles=None,
    )
    cfg = _runner.build_config(args)
    assert cfg.n_iter == 200 and cfg.n_burnin == 50


def test_build_config_overrides():
    """CLI flags override the preset's seed, iterations, and particle count."""
    args = _runner.add_common_args.__globals__["argparse"].Namespace(
        config="dev",
        seed=7,
        n_iter=12,
        n_burnin=3,
        n_particles=15,
    )
    cfg = _runner.build_config(args)
    assert cfg.seed == 7
    assert cfg.n_iter == 12 and cfg.n_burnin == 3 and cfg.N == 15


def test_make_synthetic_inputs_shapes():
    """Synthetic input builder honours requested K, T, and wallet count."""
    inputs = _runner.make_synthetic_inputs(K=2, T=40, n_wallets=8, seed=0)
    assert len(inputs.markets) == 2
    assert all(md.T == 40 for md in inputs.markets)
    assert inputs.wallet_index.n_wallets == 8
    assert inputs.is_synthetic is True


def test_pickle_and_load_run(tmp_path):
    """A pickled run round-trips its sampler, chain, and market metadata."""
    inputs = _runner.make_synthetic_inputs(K=1, T=30, n_wallets=5, seed=0)
    cfg = _runner.DEV_CONFIG
    fake_chain = "placeholder"
    out = tmp_path / "test_run.pkl"
    _runner.pickle_run(out, sampler="pg", config=cfg, chain=fake_chain, inputs=inputs)
    loaded = _runner.load_run(out)
    assert loaded["sampler"] == "pg"
    assert loaded["chain"] == "placeholder"
    assert loaded["is_synthetic"] is True
    assert len(loaded["market_objs"]) == 1


# ---------------- pull_data.py ----------------


def test_pull_data_main_with_mocked_api(tmp_path, monkeypatch):
    """End-to-end pull_data.py against canned API responses."""
    page1 = json.loads((FIXTURES / "data_trades_page1.json").read_text())
    gamma_market = {
        "id": "1",
        "conditionId": _COND_ID,
        "slug": "test-market",
        "question": "Test market for offline smoke test.",
        "volume": 100_000,
        "closed": True,
        "endDate": "2024-11-05",
        "tags": ["Politics"],
    }

    def fake_fetch_market_by_slug(slug, **kwargs):
        return MarketMeta.from_dict({**gamma_market, "slug": slug})

    fetch_count = {"n": 0}

    def fake_fetch_trades(condition_id, **kwargs):
        fetch_count["n"] += 1
        return [RawTrade.from_dict(d) for d in page1]

    monkeypatch.setattr(
        "scripts.pull_data.fetch_market_by_slug",
        fake_fetch_market_by_slug,
    )
    monkeypatch.setattr("scripts.pull_data.fetch_trades", fake_fetch_trades)

    rc = pull_data.main(
        [
            "--output-dir",
            str(tmp_path),
            "--slugs",
            "alpha",
            "beta",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    assert fetch_count["n"] == 2
    assert (tmp_path / "alpha.parquet").exists()
    assert (tmp_path / "beta.parquet").exists()
    assert (tmp_path / "wallet_index.json").exists()
    idx = json.loads((tmp_path / "wallet_index.json").read_text())
    # Same wallet set in both pages → shared index
    assert isinstance(idx, dict) and len(idx) >= 1


def test_pull_data_tail_truncates(tmp_path, monkeypatch):
    """--tail-trades trims a market down to its last N trades."""
    page1 = json.loads((FIXTURES / "data_trades_page1.json").read_text())
    gamma_market = {
        "id": "1",
        "conditionId": _COND_ID,
        "slug": "x",
        "question": "x",
        "volume": 100_000,
        "closed": True,
        "endDate": "2024-11-05",
    }
    monkeypatch.setattr(
        "scripts.pull_data.fetch_market_by_slug",
        lambda s, **k: MarketMeta.from_dict({**gamma_market, "slug": s}),
    )
    monkeypatch.setattr(
        "scripts.pull_data.fetch_trades",
        lambda *a, **k: [RawTrade.from_dict(d) for d in page1],
    )

    rc = pull_data.main(
        [
            "--output-dir",
            str(tmp_path),
            "--slugs",
            "alpha",
            "--tail-trades",
            "3",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    from src.data.preprocess import load_processed

    mkt = load_processed(tmp_path / "alpha.parquet")
    assert mkt.T == 3
    assert mkt.delta[0] == 0.0


def _mock_pull_data_api(monkeypatch, slug_meta_extra=None):
    """Point pull_data at canned Gamma/Data responses; returns recorded calls.

    The returned dict has a "trades" list (kwargs of every `fetch_trades` call)
    and a "windowed" list (kwargs of every `fetch_trades_windowed` call).
    """
    page1 = json.loads((FIXTURES / "data_trades_page1.json").read_text())
    gamma_market = {
        "id": "1",
        "conditionId": _COND_ID,
        "slug": "x",
        "question": "x",
        "volume": 100_000,
        "closed": True,
        "endDate": "2024-11-05",
        **(slug_meta_extra or {}),
    }
    calls: dict[str, list] = {"trades": [], "windowed": []}

    def record(key):
        def fake(condition_id, **kwargs):
            calls[key].append({"condition_id": condition_id, **kwargs})
            return [RawTrade.from_dict(d) for d in page1]

        return fake

    monkeypatch.setattr(
        "scripts.pull_data.fetch_market_by_slug",
        lambda s, **k: MarketMeta.from_dict({**gamma_market, "slug": s}),
    )
    monkeypatch.setattr("scripts.pull_data.fetch_trades", record("trades"))
    monkeypatch.setattr("scripts.pull_data.fetch_trades_windowed", record("windowed"))
    return calls


def test_pull_data_without_full_history_keeps_tail_call_pattern(tmp_path, monkeypatch):
    """Default (flag off) still hits fetch_trades with the unchanged kwargs."""
    calls = _mock_pull_data_api(monkeypatch)

    rc = pull_data.main(
        [
            "--output-dir",
            str(tmp_path),
            "--slugs",
            "alpha",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    assert calls["windowed"] == []
    assert calls["trades"] == [
        {"condition_id": _COND_ID, "max_pages": 200, "sleep_between": 0.1}
    ]


def test_pull_data_full_history_uses_windowed_fetch_from_epoch_one(
    tmp_path, monkeypatch
):
    """--full-history routes to the windowed fetcher with start_ts=1, not 0."""
    calls = _mock_pull_data_api(monkeypatch)

    rc = pull_data.main(
        [
            "--output-dir",
            str(tmp_path),
            "--slugs",
            "alpha",
            "--full-history",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    assert calls["trades"] == []
    assert len(calls["windowed"]) == 1
    assert calls["windowed"][0]["condition_id"] == _COND_ID
    # 0 would be dropped server-side as falsy, so the bound must be epoch 1.
    assert calls["windowed"][0]["start_ts"] == 1
    assert calls["windowed"][0]["max_pages"] == 20_000


def test_pull_data_full_history_survives_one_market_failing(
    tmp_path, monkeypatch, caplog
):
    """One market's API failure costs that market only, not the whole pull.

    A --full-history backfill is hours per market, so losing the markets already
    retrieved to a late failure is the expensive kind of bug. The run must keep
    the survivors *and* say which slugs are missing, or a partial directory is
    indistinguishable from a complete one.
    """
    _mock_pull_data_api(monkeypatch)
    page1 = json.loads((FIXTURES / "data_trades_page1.json").read_text())

    # The mocked metas all share one condition_id, so the failure is driven by
    # call order: the second market pulled ("beta") is the one that breaks.
    calls = {"n": 0}

    def fake_windowed(condition_id, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise PolymarketAPIError("500 Server Error from /trades")
        return [RawTrade.from_dict(d) for d in page1]

    monkeypatch.setattr("scripts.pull_data.fetch_trades_windowed", fake_windowed)

    with caplog.at_level("ERROR"):
        rc = pull_data.main(
            [
                "--output-dir",
                str(tmp_path),
                "--slugs",
                "alpha",
                "beta",
                "gamma",
                "--full-history",
                "--log-level",
                "ERROR",
            ]
        )

    assert rc == 0
    assert calls["n"] == 3  # the loop kept going after the failure
    assert (tmp_path / "alpha.parquet").exists()
    assert (tmp_path / "gamma.parquet").exists()
    assert not (tmp_path / "beta.parquet").exists()
    assert (tmp_path / "wallet_index.json").exists()
    assert "FAILED beta" in caplog.text
    # The closing line must name the gap, not just log it mid-run.
    assert "INCOMPLETE: 1 of 3" in caplog.text


def test_pull_data_full_history_applies_tail_after_retrieval(tmp_path, monkeypatch):
    """--tail-trades still caps the kept trades when --full-history is on."""
    calls = _mock_pull_data_api(monkeypatch)

    rc = pull_data.main(
        [
            "--output-dir",
            str(tmp_path),
            "--slugs",
            "alpha",
            "--full-history",
            "--tail-trades",
            "3",
            "--max-pages",
            "7",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    # The cap is a post-retrieval slice: the fetch itself is still unbounded
    # apart from the explicit --max-pages budget.
    assert calls["windowed"][0]["max_pages"] == 7
    assert "end_ts" not in calls["windowed"][0]

    from src.data.preprocess import load_processed

    mkt = load_processed(tmp_path / "alpha.parquet")
    assert mkt.T == 3
    assert mkt.delta[0] == 0.0


# ---------------- pull_kalshi.py ----------------


def _kalshi_meta(ticker: str, *, close_time: str = "2026-07-01T03:59:00Z"):
    """Canned Kalshi market metadata for the offline pull_kalshi smoke tests."""
    return KalshiMarketMeta.from_dict(
        {
            "ticker": ticker,
            "title": f"Offline smoke market {ticker}",
            "status": "finalized",
            "close_time": close_time,
            "volume_fp": "1234.50",
        }
    )


def _kalshi_trades(ticker: str, n: int = 8) -> list[RawTrade]:
    """n anonymous Kalshi-shaped trades, one per hour, ending well before close.

    Timestamps sit ~30 days before the canned close time so the default
    7-day pre-resolution filter keeps every row.
    """
    base = 1_780_000_000
    return [
        RawTrade(
            timestamp=base + 3600 * i,
            price=0.40 + 0.01 * i,
            size=10.0 + i,
            wallet=None,
            side="BUY" if i % 2 else "SELL",
            transaction_hash=f"{ticker}-trade-{i:03d}",
            condition_id=ticker,
            asset_id="",
        )
        for i in range(n)
    ]


def _mock_kalshi_api(monkeypatch) -> dict:
    """Point pull_kalshi at canned Kalshi responses; returns recorded calls."""
    calls: dict[str, list] = {"market": [], "trades": []}

    def fake_fetch_market(ticker, **kwargs):
        calls["market"].append(ticker)
        return _kalshi_meta(ticker)

    def fake_fetch_trades(ticker, **kwargs):
        calls["trades"].append({"ticker": ticker, **kwargs})
        return _kalshi_trades(ticker)

    monkeypatch.setattr("scripts.pull_kalshi.fetch_market", fake_fetch_market)
    monkeypatch.setattr("scripts.pull_kalshi.fetch_trades", fake_fetch_trades)
    return calls


def test_pull_kalshi_main_with_mocked_api(tmp_path, monkeypatch):
    """End-to-end pull_kalshi.py against canned GetTrades responses."""
    calls = _mock_kalshi_api(monkeypatch)

    rc = pull_kalshi.main(
        [
            "--tickers",
            "KXA-26JUL01",
            "KXB-26JUL01",
            "--output-dir",
            str(tmp_path),
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    assert calls["market"] == ["KXA-26JUL01", "KXB-26JUL01"]
    # Default pull is the newest-N tail, matching pull_data's budgeted default.
    assert calls["trades"][0]["max_trades"] == kalshi_api.DEFAULT_MAX_TRADES

    df = pd.read_parquet(tmp_path / "KXA-26JUL01.parquet")
    assert len(df) == 8
    assert list(df["timestamp"]) == sorted(df["timestamp"])
    # No-identity invariant survives the whole CLI path, not just the parser.
    assert df["wallet"].isna().all()

    meta = json.loads((tmp_path / "KXA-26JUL01.meta.json").read_text())
    assert meta["source"] == "kalshi"
    assert meta["mode"] == "anonymous"
    assert meta["n_trades"] == 8
    assert (tmp_path / "KXB-26JUL01.parquet").exists()


def test_pull_kalshi_full_history_lifts_the_row_budget(tmp_path, monkeypatch):
    """--full-history walks the cursor to the first trade (max_trades=None)."""
    calls = _mock_kalshi_api(monkeypatch)

    rc = pull_kalshi.main(
        [
            "--tickers",
            "KXA-26JUL01",
            "--output-dir",
            str(tmp_path),
            "--full-history",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    assert calls["trades"][0]["max_trades"] is None


def test_pull_kalshi_tail_and_pre_resolution_filter(tmp_path, monkeypatch):
    """--tail-trades slices post-retrieval; --pre-resolution-days drops the tail."""
    _mock_kalshi_api(monkeypatch)

    rc = pull_kalshi.main(
        [
            "--tickers",
            "KXA-26JUL01",
            "--output-dir",
            str(tmp_path),
            "--tail-trades",
            "3",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    kept = pd.read_parquet(tmp_path / "KXA-26JUL01.parquet")
    assert len(kept) == 3
    assert list(kept["transaction_hash"]) == [
        "KXA-26JUL01-trade-005",
        "KXA-26JUL01-trade-006",
        "KXA-26JUL01-trade-007",
    ]

    # A close time just after the last trade puts every row inside the 7-day
    # exclusion window, so nothing survives and the market is reported failed.
    monkeypatch.setattr(
        "scripts.pull_kalshi.fetch_market",
        lambda ticker, **k: _kalshi_meta(ticker, close_time="2026-05-30T00:00:00Z"),
    )
    rc = pull_kalshi.main(
        [
            "--tickers",
            "KXA-26JUL01",
            "--output-dir",
            str(tmp_path / "empty"),
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 1
    assert not (tmp_path / "empty" / "KXA-26JUL01.parquet").exists()


def test_pull_kalshi_survives_one_failing_ticker(tmp_path, monkeypatch):
    """One bad ticker is reported but does not discard the markets already pulled."""
    _mock_kalshi_api(monkeypatch)

    def flaky_fetch_market(ticker, **kwargs):
        if ticker == "KXBAD-26JUL01":
            raise KalshiAPIError("HTTP 404: not found")
        return _kalshi_meta(ticker)

    monkeypatch.setattr("scripts.pull_kalshi.fetch_market", flaky_fetch_market)

    rc = pull_kalshi.main(
        [
            "--tickers",
            "KXBAD-26JUL01",
            "KXA-26JUL01",
            "--output-dir",
            str(tmp_path),
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 0
    assert (tmp_path / "KXA-26JUL01.parquet").exists()
    assert not (tmp_path / "KXBAD-26JUL01.parquet").exists()


def test_pull_kalshi_wallet_mode_override_is_rejected(tmp_path, monkeypatch):
    """--mode wallet on an identity-free source fails loudly (KTD3 seam)."""
    _mock_kalshi_api(monkeypatch)

    with pytest.raises(ValueError, match="no trade carries a wallet"):
        pull_kalshi.main(
            [
                "--tickers",
                "KXA-26JUL01",
                "--output-dir",
                str(tmp_path),
                "--mode",
                "wallet",
                "--log-level",
                "WARNING",
            ]
        )


# ---------------- run_pg.py / run_ipmcmc.py ----------------


def test_run_pg_synthetic_writes_pickle(tmp_path):
    """run_pg.py --synthetic writes a PGOutput pickle with expected shapes."""
    out = tmp_path / "pg.pkl"
    rc = run_pg.main(
        [
            "--synthetic",
            "--synthetic-K",
            "2",
            "--synthetic-T",
            "40",
            "--synthetic-n-wallets",
            "5",
            "--config",
            "dev",
            "--n-iter",
            "8",
            "--n-burnin",
            "2",
            "--n-particles",
            "12",
            "--output",
            str(out),
            "--no-progress",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    payload = pickle.loads(out.read_bytes())
    assert payload["sampler"] == "pg"
    assert isinstance(payload["chain"], PGOutput)
    assert payload["chain"].sigma2_0.shape == (8,)
    assert payload["is_synthetic"] is True
    assert len(payload["market_objs"]) == 2


def test_run_pg_n_jobs_defaults_to_one_and_is_overridable():
    """--n-jobs defaults to 1 (sequential, bit-exact) and can be overridden."""
    args_default = run_pg._parse_args(["--synthetic", "--config", "dev"])
    assert args_default.n_jobs == 1

    args_override = run_pg._parse_args(
        ["--synthetic", "--config", "dev", "--n-jobs", "4"]
    )
    assert args_override.n_jobs == 4


def test_run_ipmcmc_synthetic_writes_pickle(tmp_path):
    """run_ipmcmc.py --synthetic writes an iPMCMCOutput with (n_iter, P) shape."""
    out = tmp_path / "ip.pkl"
    rc = run_ipmcmc.main(
        [
            "--synthetic",
            "--synthetic-K",
            "1",
            "--synthetic-T",
            "30",
            "--synthetic-n-wallets",
            "5",
            "--config",
            "dev",
            "--n-iter",
            "6",
            "--n-burnin",
            "2",
            "--n-particles",
            "10",
            "--M",
            "4",
            "--P",
            "2",
            "--output",
            str(out),
            "--no-progress",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    payload = pickle.loads(out.read_bytes())
    assert payload["sampler"] == "ipmcmc"
    assert isinstance(payload["chain"], iPMCMCOutput)
    assert payload["chain"].sigma2_0.shape == (6, 2)


def test_run_ipmcmc_rejects_p_gt_m(tmp_path):
    """P > M is rejected at the CLI with a SystemExit."""
    with pytest.raises(SystemExit):
        run_ipmcmc.main(
            [
                "--synthetic",
                "--config",
                "dev",
                "--M",
                "2",
                "--P",
                "4",
                "--n-iter",
                "4",
                "--n-burnin",
                "1",
                "--n-particles",
                "10",
                "--output",
                str(tmp_path / "ip.pkl"),
                "--no-progress",
                "--log-level",
                "WARNING",
            ]
        )


def test_run_pg_loads_real_inputs_from_disk(tmp_path, monkeypatch):
    """run_pg.py reads processed parquet + wallet_index.json that pull_data
    produced (here the pull is mocked)."""
    page1 = json.loads((FIXTURES / "data_trades_page1.json").read_text())
    gamma_market = {
        "id": "1",
        "conditionId": _COND_ID,
        "slug": "x",
        "question": "x",
        "volume": 100_000,
        "closed": True,
        "endDate": "2024-11-05",
    }
    monkeypatch.setattr(
        "scripts.pull_data.fetch_market_by_slug",
        lambda s, **k: MarketMeta.from_dict({**gamma_market, "slug": s}),
    )
    monkeypatch.setattr(
        "scripts.pull_data.fetch_trades",
        lambda *a, **k: [RawTrade.from_dict(d) for d in page1],
    )
    data_dir = tmp_path / "processed"
    pull_data.main(
        [
            "--output-dir",
            str(data_dir),
            "--slugs",
            "alpha",
            "--log-level",
            "WARNING",
        ]
    )

    out = tmp_path / "pg_real.pkl"
    rc = run_pg.main(
        [
            "--data-dir",
            str(data_dir),
            "--config",
            "dev",
            "--n-iter",
            "6",
            "--n-burnin",
            "2",
            "--n-particles",
            "10",
            "--output",
            str(out),
            "--no-progress",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    payload = pickle.loads(out.read_bytes())
    assert payload["is_synthetic"] is False
    assert payload["slugs"] == ["alpha"]


# ---------------- make_figures.py ----------------


def test_make_figures_end_to_end(tmp_path):
    """run a tiny PG synthetic run, then make_figures on its pickle."""
    pkl = tmp_path / "pg.pkl"
    run_pg.main(
        [
            "--synthetic",
            "--synthetic-K",
            "2",
            "--synthetic-T",
            "40",
            "--synthetic-n-wallets",
            "6",
            "--config",
            "dev",
            "--n-iter",
            "10",
            "--n-burnin",
            "3",
            "--n-particles",
            "12",
            "--output",
            str(pkl),
            "--no-progress",
            "--log-level",
            "WARNING",
        ]
    )

    figs = tmp_path / "figs"
    tabs = tmp_path / "tabs"
    rc = make_figures.main(
        [
            "--chain",
            str(pkl),
            "--figures-dir",
            str(figs),
            "--tables-dir",
            str(tabs),
            "--top-k-wallets",
            "5",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    # Per-market overview for each of the 2 synthetic markets
    overview_pdfs = list(figs.glob("pg_*_overview.pdf"))
    assert len(overview_pdfs) == 2
    assert (figs / "pg_chain_diagnostics.pdf").exists()
    assert (figs / "pg_wallet_ranking.pdf").exists()
    assert (figs / "pg_roc.pdf").exists()  # synthetic ⇒ ROC produced
    assert (tabs / "pg_chain_summary.csv").exists()
    assert (tabs / "pg_wallet_ranking.csv").exists()


def test_make_figures_skips_roc_on_real_data(tmp_path, monkeypatch):
    """No SyntheticMarket → no ROC figure written."""
    page1 = json.loads((FIXTURES / "data_trades_page1.json").read_text())
    gamma_market = {
        "id": "1",
        "conditionId": _COND_ID,
        "slug": "x",
        "question": "x",
        "volume": 100_000,
        "closed": True,
        "endDate": "2024-11-05",
    }
    monkeypatch.setattr(
        "scripts.pull_data.fetch_market_by_slug",
        lambda s, **k: MarketMeta.from_dict({**gamma_market, "slug": s}),
    )
    monkeypatch.setattr(
        "scripts.pull_data.fetch_trades",
        lambda *a, **k: [RawTrade.from_dict(d) for d in page1],
    )
    data_dir = tmp_path / "processed"
    pull_data.main(
        [
            "--output-dir",
            str(data_dir),
            "--slugs",
            "alpha",
            "--log-level",
            "WARNING",
        ]
    )
    pkl = tmp_path / "pg_real.pkl"
    run_pg.main(
        [
            "--data-dir",
            str(data_dir),
            "--config",
            "dev",
            "--n-iter",
            "6",
            "--n-burnin",
            "2",
            "--n-particles",
            "10",
            "--output",
            str(pkl),
            "--no-progress",
            "--log-level",
            "WARNING",
        ]
    )
    figs = tmp_path / "figs"
    tabs = tmp_path / "tabs"
    make_figures.main(
        [
            "--chain",
            str(pkl),
            "--figures-dir",
            str(figs),
            "--tables-dir",
            str(tabs),
            "--top-k-wallets",
            "3",
            "--log-level",
            "WARNING",
        ]
    )
    assert not (figs / "pg_roc.pdf").exists()
    assert (figs / "pg_alpha_overview.pdf").exists()


# ---------------- benchmark.py ----------------


_BENCHMARK_VEM_FILTER = [
    "--synthetic",
    "--synthetic-K",
    "2",
    "--synthetic-T",
    "40",
    "--synthetic-n-wallets",
    "8",
    "--config",
    "dev",
    "--n-iter",
    "10",
    "--n-burnin",
    "2",
    "--n-particles",
    "10",
    "--n-runs",
    "1",
    "--threads",
    "1",
    "--log-level",
    "WARNING",
]

_BENCHMARK_TINY = [
    "--synthetic",
    "--synthetic-K",
    "2",
    "--synthetic-T",
    "40",
    "--synthetic-n-wallets",
    "8",
    "--config",
    "dev",
    "--n-iter",
    "6",
    "--n-burnin",
    "2",
    "--n-particles",
    "10",
    "--n-runs",
    "2",
    "--threads",
    "1",
    "--log-level",
    "WARNING",
]


def test_benchmark_smoke(tmp_path):
    """benchmark.py runs on tiny synthetic scale and writes optional JSON."""
    json_path = tmp_path / "bench.json"
    rc = benchmark.main([*_BENCHMARK_TINY, "--json-out", str(json_path)])
    assert rc == 0
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["inputs"]["synthetic"] is True
    assert len(payload["timings"]["sec_per_run"]) == 2
    assert "profile_tottime" in payload


def test_benchmark_gate_smoke():
    """benchmark.py --gate computes synthetic metrics without error."""
    rc = benchmark.main([*_BENCHMARK_TINY, "--gate"])
    assert rc == 0


def test_benchmark_vem_smoke(tmp_path):
    """benchmark.py --method vem runs and writes method key to JSON."""
    json_path = tmp_path / "bench_vem.json"
    rc = benchmark.main(
        [
            *_BENCHMARK_VEM_FILTER,
            "--method",
            "vem",
            "--vem-iters",
            "10",
            "--json-out",
            str(json_path),
        ],
    )
    assert rc == 0
    payload = json.loads(json_path.read_text())
    assert payload["method"] == "vem"


def test_benchmark_filter_gate_smoke(tmp_path):
    """benchmark.py --method filter --gate runs and writes method key to JSON."""
    json_path = tmp_path / "bench_filter.json"
    rc = benchmark.main(
        [
            *_BENCHMARK_VEM_FILTER,
            "--method",
            "filter",
            "--gate",
            "--json-out",
            str(json_path),
        ],
    )
    assert rc == 0
    payload = json.loads(json_path.read_text())
    assert payload["method"] == "filter"


_BENCHMARK_IPMCMC = [
    "--synthetic",
    "--synthetic-K",
    "2",
    "--synthetic-T",
    "60",
    "--synthetic-n-wallets",
    "10",
    "--config",
    "dev",
    "--n-iter",
    "5",
    "--n-burnin",
    "1",
    "--n-particles",
    "20",
    "--n-runs",
    "1",
    "--threads",
    "2",
    "--log-level",
    "WARNING",
]


def test_benchmark_ipmcmc_gate_smoke(tmp_path):
    """benchmark.py --method ipmcmc --gate runs and records M/P in JSON config."""
    json_path = tmp_path / "bench_ipmcmc.json"
    rc = benchmark.main(
        [
            *_BENCHMARK_IPMCMC,
            "--method",
            "ipmcmc",
            "--gate",
            "--json-out",
            str(json_path),
        ],
    )
    assert rc == 0
    payload = json.loads(json_path.read_text())
    assert payload["method"] == "ipmcmc"
    assert payload["config"]["M"] == 8
    assert payload["config"]["P"] == 4
    assert payload["gate"] is not None


def test_benchmark_ipmcmc_rejects_p_gt_m():
    """benchmark.py --method ipmcmc exits clearly when P > M."""
    with pytest.raises(SystemExit):
        benchmark.main(
            [*_BENCHMARK_IPMCMC, "--method", "ipmcmc", "--M", "2", "--P", "4"]
        )


def test_artifacts_from_mcmc_chain_flattens_ipmcmc_theta():
    """iPMCMC theta_w (n_iter, P, n_wallets) pools conditional chains post-burn-in."""
    n_iter, M, P, n_wallets, T = 4, 3, 2, 3, 5
    rng = np.random.default_rng(0)
    theta_w = rng.uniform(size=(n_iter, P, n_wallets))
    param = np.ones((n_iter, P))
    latent = np.zeros((n_iter, P, T))
    chain = iPMCMCOutput(
        sigma2_0=param,
        sigma2_1=param,
        q_01=param,
        q_10=param,
        beta_S=param,
        beta_Z=param,
        tau2_0=param,
        tau2_1=param,
        theta_w=theta_w,
        X=[latent],
        V=[latent],
        Z=[rng.integers(0, 2, size=(n_iter, P, T)).astype(float)],
        log_marg=np.zeros((n_iter, M)),
        chain_indices=np.zeros((n_iter, P), dtype=int),
        acc_beta_S=np.ones((n_iter, P), dtype=bool),
        acc_beta_Z=np.ones((n_iter, P), dtype=bool),
        acc_tau2_0=np.ones((n_iter, P), dtype=bool),
        acc_tau2_1=np.ones((n_iter, P), dtype=bool),
        final_mh_step_beta_S=0.1,
        final_mh_step_beta_Z=0.1,
        final_mh_step_log_tau2_0=0.1,
        final_mh_step_log_tau2_1=0.1,
    )
    n_burnin = 1
    artifacts = benchmark._artifacts_from_mcmc_chain(chain, n_burnin=n_burnin)
    expected = theta_w[n_burnin:].reshape(-1, n_wallets).mean(axis=0)
    np.testing.assert_allclose(artifacts.theta_w, expected)


# ---------------- eval_c4.py ----------------


def test_eval_c4_smoke(tmp_path):
    """eval_c4.py runs on tiny synthetic scale and writes gate_pass to JSON."""
    json_path = tmp_path / "c4.json"
    rc = eval_c4.main(
        [
            "--synthetic-K",
            "2",
            "--synthetic-T",
            "60",
            "--synthetic-n-wallets",
            "10",
            "--config",
            "dev",
            "--vem-iters",
            "10",
            "--json-out",
            str(json_path),
            "--log-level",
            "WARNING",
        ],
    )
    assert rc == 0
    payload = json.loads(json_path.read_text())
    assert "gate_pass" in payload
    assert payload["inputs"]["synthetic"] is True


# ---------------- validate_vem.py ----------------

# Deliberately tiny: 2 markets x 100 trades, 2 restarts, 5 EM iterations and the
# PSIS tail-fit minimum of draws. Every knob that costs an ADF forward pass is at
# its floor so the fast suite pays seconds, not minutes.
_VALIDATE_VEM_TINY = [
    "--synthetic-K",
    "2",
    "--synthetic-T",
    "100",
    "--synthetic-n-wallets",
    "8",
    "--config",
    "dev",
    "--n-restarts",
    "2",
    "--psis-draws",
    "50",
    "--vem-iters",
    "5",
    "--log-level",
    "WARNING",
]


@pytest.fixture(scope="module")
def validate_vem_run(tmp_path_factory):
    """Run validate_vem.py once at tiny scale; yield ``(rc, payload, out_dir)``.

    Module-scoped because the run costs a handful of ADF passes and several
    tests assert on different parts of the same artifact.
    """
    out_dir = tmp_path_factory.mktemp("validation")
    json_path = out_dir / "vem_validation.json"
    rc = validate_vem.main(
        [
            *_VALIDATE_VEM_TINY,
            "--out-dir",
            str(out_dir),
            "--json-out",
            str(json_path),
        ],
    )
    payload = json.loads(json_path.read_text()) if json_path.exists() else {}
    return rc, payload, out_dir


def test_validate_vem_smoke(validate_vem_run):
    """validate_vem.py exits 0 and writes a JSON artifact with every section."""
    rc, payload, _ = validate_vem_run
    assert rc == 0
    assert set(payload) >= {
        "config",
        "inputs",
        "convergence_status",
        "prior",
        "restarts",
        "stability",
        "best_restart",
        "heldout",
        "psis",
        "laplace",
        "figures",
    }
    assert payload["inputs"]["synthetic"] is True
    assert len(payload["restarts"]) == 2
    assert payload["psis"]["psis_n_draws"] == 50
    assert np.isfinite(payload["psis"][PSIS_KHAT_KEY])
    assert "psis_scope_note" in payload["psis"]
    assert payload["heldout"]["pooled_n"] > 0
    assert len(payload["heldout"]["per_market"]) == 2


def test_validate_vem_records_convergence_status(validate_vem_run):
    """The artifact says whether the restarts converged, and warns if not.

    The tiny run caps at 5 EM iterations, so it is guaranteed pre-convergence:
    a reader must be able to see that from the JSON alone rather than inferring
    it by comparing ``n_iter_run`` against ``--vem-iters`` by hand.
    """
    _, payload, _ = validate_vem_run
    status = payload["convergence_status"]
    assert status["vem_iters"] == 5
    assert status["n_restarts_at_iter_cap"] == 2
    assert status["converged"] is False
    assert any("PRE-CONVERGENCE" in w for w in status["warnings"])
    assert isinstance(status["best_restart_selection_meaningful"], bool)
    assert np.isfinite(status["median_final_elbo_gain"])
    for r in payload["restarts"]:
        assert r["hit_iter_cap"] is True
        assert r["converged"] is False
        assert np.isfinite(r["final_rel_elbo_change"])


def test_validate_vem_stability_carries_its_escalation_flag(validate_vem_run):
    """The stability block states the AUC-spread threshold and its verdict."""
    _, payload, _ = validate_vem_run
    stability = payload["stability"]
    assert stability["pooled_auc_spread_threshold"] > 0.0
    assert isinstance(stability["pooled_auc_unstable"], bool)
    assert isinstance(stability["warnings"], list)
    # The flag and the warning list agree: one implies the other.
    assert stability["pooled_auc_unstable"] == bool(stability["warnings"])


def test_validate_vem_artifact_is_self_describing(validate_vem_run):
    """The payload records what was fit, not only how well it scored (H6)."""
    _, payload, _ = validate_vem_run
    cfg = payload["config"]
    assert (cfg["synthetic_K"], cfg["synthetic_T"], cfg["synthetic_n_wallets"]) == (
        2,
        100,
        8,
    )
    assert cfg["real"] is False
    # Fitted phi and the Laplace layer it induced, so the run is reproducible
    # and auditable from the artifact alone.
    best = payload["best_restart"]
    assert best["params"]["sigma2_0"] > 0.0
    # theta_w + the centering constants make best_restart a *complete* warm
    # start: score_stream.py rejects a params-only block, because beta_S/beta_Z
    # are on the standardized covariate scale (m_S, s_S, m_Z) defines.
    assert set(best) >= {"params", "theta_w", "m_S", "s_S", "m_Z"}
    assert len(best["theta_w"]) == cfg["synthetic_n_wallets"]
    assert best["s_S"] > 0.0
    laplace = payload["laplace"]
    assert len(laplace["dims"]) == 8
    assert len(laplace["mean_u"]) == 8
    assert np.asarray(laplace["cov_u"]).shape == (8, 8)


def test_validate_vem_reports_proposal_centring(validate_vem_run):
    """khat ships with the centring gradient that qualifies how to read it."""
    _, payload, _ = validate_vem_run
    psis = payload["psis"]
    grad = psis["centring_grad_sd_units"]
    assert set(grad) == set(payload["laplace"]["dims"])
    assert all(np.isfinite(v) for v in grad.values())
    assert psis["centring_grad_max_abs_dim"] in grad
    assert psis["centring_grad_max_abs_sd"] == pytest.approx(
        max(abs(v) for v in grad.values()),
    )


def test_validate_vem_writes_figures(validate_vem_run):
    """All three validation figures land under --out-dir and are non-empty."""
    _, payload, out_dir = validate_vem_run
    assert len(payload["figures"]) >= 3
    for rel in payload["figures"]:
        path = Path(rel)
        assert path.exists() and path.stat().st_size > 0
        assert path.parent == out_dir


def test_validate_vem_restarts_differ_by_seed(validate_vem_run):
    """Restarts use distinct seeds, so their terminal ELBOs are not identical."""
    _, payload, _ = validate_vem_run
    seeds = [r["seed"] for r in payload["restarts"]]
    assert len(set(seeds)) == len(seeds)
    elbos = [r["terminal_elbo"] for r in payload["restarts"]]
    assert len(set(elbos)) > 1
    assert payload["stability"]["terminal_elbo"]["spread"] > 0.0


def test_validate_vem_rejects_too_few_psis_draws(tmp_path):
    """--psis-draws below the PSIS tail-fit minimum fails fast with a message."""
    with pytest.raises(SystemExit):
        validate_vem.main(
            [
                *_VALIDATE_VEM_TINY,
                "--out-dir",
                str(tmp_path),
                # Later occurrence wins in argparse, overriding the 50 above.
                "--psis-draws",
                "10",
            ],
        )


# ---------------- pareto.py ----------------


def _fake_bench_json(
    *,
    method: str = "pg",
    mean_sec: float = 228.0,
    ci_hw: float = 12.0,
    pooled_auc: float = 0.91,
    gate_pass: bool = True,
    with_gate: bool = True,
    kendall_tau: float | None = None,
) -> dict:
    """Minimal benchmark JSON payload for offline pareto.py tests."""
    payload = {
        "method": method,
        "config": {
            "N": 50,
            "n_iter": 200,
            "n_burnin": 50,
            "seed_base": 42,
            "threads": 1,
        },
        "inputs": {"K": 2, "synthetic": True, "seeds": [42, 43]},
        "timings": {
            "sec_per_run": [mean_sec - 1.0, mean_sec + 1.0],
            "mean_sec_per_run": mean_sec,
            "ci_half_width_sec_per_run": ci_hw,
            "mean_sec_per_iter": mean_sec / 200.0,
            "ci_half_width_sec_per_iter": ci_hw / 200.0,
        },
        "profile_tottime": None,
        "gate": (
            {
                "pooled_auc": pooled_auc,
                "gate_pass": gate_pass,
            }
            if with_gate
            else None
        ),
    }
    if kendall_tau is not None:
        payload["kendall_tau_vs_baseline"] = kendall_tau
    return payload


def test_pareto_from_fake_benchmark_json(tmp_path):
    """pareto.py plots gated benchmark JSON and writes PNG + CSV summary."""
    bench_pg = tmp_path / "bench_pg.json"
    bench_vem = tmp_path / "bench_vem.json"
    bench_no_gate = tmp_path / "bench_no_gate.json"
    bench_pg.write_text(json.dumps(_fake_bench_json(method="pg", mean_sec=228.0)))
    bench_vem.write_text(
        json.dumps(
            _fake_bench_json(
                method="vem",
                mean_sec=45.0,
                ci_hw=0.0,
                pooled_auc=0.88,
                kendall_tau=0.72,
            )
        )
    )
    bench_no_gate.write_text(json.dumps(_fake_bench_json(with_gate=False)))

    png_out = tmp_path / "pareto.png"
    csv_out = tmp_path / "pareto.csv"
    rc = pareto.main(
        [
            "--bench-json",
            str(bench_pg),
            str(bench_vem),
            str(bench_no_gate),
            "--output",
            str(png_out),
            "--csv-out",
            str(csv_out),
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    assert png_out.exists() and png_out.stat().st_size > 0

    table = pd.read_csv(csv_out)
    assert len(table) == 2
    assert set(table["method"]) == {"pg", "vem"}
    assert table.loc[table["method"] == "pg", "pooled_auc"].iloc[0] == pytest.approx(
        0.91
    )
    vem_row = table.loc[table["method"] == "vem"].iloc[0]
    assert vem_row["kendall_tau_vs_baseline"] == pytest.approx(0.72)
    assert vem_row["label"] == "vem N=50 it=200"


# ---------------- stream_trades.py ----------------

_OTHER_COND_ID = "0xccc000000000000000000000000000000000000000000000000000000000cc01"


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL capture, asserting every line is a complete JSON object."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        assert line.endswith("}"), f"truncated record: {line!r}"
        records.append(json.loads(line))
    return records


def _socket_factory(frames: list):
    """One-shot factory handing `stream_trades.main` a scripted fake socket."""
    return lambda: FakeSocket(list(frames))


def test_stream_trades_records_normalized_jsonl(tmp_path):
    """Live frames land as one JSON object per line with RawTrade fields."""
    out = tmp_path / "live" / "trades.jsonl"
    frames = [make_frame(price=0.11, size=5), make_frame(price=0.22, size=7)]

    rc = stream_trades.main(
        ["--output", str(out), "--max-trades", "2", "--log-level", "WARNING"],
        socket_factory=_socket_factory(frames),
    )

    assert rc == 0
    records = _read_jsonl(out)
    assert [r["price"] for r in records] == pytest.approx([0.11, 0.22])
    assert [r["size"] for r in records] == pytest.approx([5.0, 7.0])
    assert all(r["wallet"].startswith("0x") for r in records)
    assert all(r["timestamp"] == 1785027308 for r in records)


def test_stream_trades_market_filter_keeps_only_requested_ids(tmp_path):
    """--markets drops trades from every other condition id."""
    out = tmp_path / "trades.jsonl"
    frames = [
        make_frame(conditionId=_OTHER_COND_ID, price=0.1),
        make_frame(price=0.2),
        make_frame(conditionId=_OTHER_COND_ID, price=0.3),
        make_frame(price=0.4),
    ]

    rc = stream_trades.main(
        [
            "--output",
            str(out),
            "--markets",
            LIVE_CONDITION_ID,
            "--max-trades",
            "2",
            "--log-level",
            "WARNING",
        ],
        socket_factory=_socket_factory(frames),
    )

    assert rc == 0
    records = _read_jsonl(out)
    assert [r["condition_id"] for r in records] == [LIVE_CONDITION_ID] * 2
    assert [r["price"] for r in records] == pytest.approx([0.2, 0.4])


def test_stream_trades_interrupt_leaves_whole_records(tmp_path):
    """A Ctrl-C mid-stream exits 0 with every written line complete JSON."""
    out = tmp_path / "trades.jsonl"
    frames = [
        make_frame(price=0.5),
        make_frame(price=0.6),
        KeyboardInterrupt("ctrl-c"),
    ]

    rc = stream_trades.main(
        ["--output", str(out), "--log-level", "WARNING"],
        socket_factory=_socket_factory(frames),
    )

    assert rc == 0
    records = _read_jsonl(out)
    assert [r["price"] for r in records] == pytest.approx([0.5, 0.6])


def test_stream_trades_appends_across_runs_and_compacts_parquet(tmp_path):
    """--parquet-every compacts the whole append-only capture, not just a slice."""
    out = tmp_path / "trades.jsonl"
    parquet = tmp_path / "trades.parquet"
    common = ["--output", str(out), "--max-trades", "2", "--log-level", "WARNING"]

    rc = stream_trades.main(
        common, socket_factory=_socket_factory([make_frame(price=0.1)] * 2)
    )
    assert rc == 0

    rc = stream_trades.main(
        common + ["--parquet-every", "2", "--parquet-path", str(parquet)],
        socket_factory=_socket_factory([make_frame(price=0.9)] * 2),
    )
    assert rc == 0

    assert len(_read_jsonl(out)) == 4  # second run appended, did not clobber
    frame = pd.read_parquet(parquet)
    assert len(frame) == 4
    assert set(frame.columns) >= {"timestamp", "price", "size", "wallet", "side"}
    assert frame["price"].tolist() == pytest.approx([0.1, 0.1, 0.9, 0.9])


# ---------------- score_stream.py ----------------

# Wallet addresses of the streaming fixtures, in the order a fresh WalletIndex
# would assign ids to them.
_STREAM_WALLETS = ("0xw0", "0xw1", "0xw2")


def _raw_trade(i: int, *, ts: float, price: float, size: float, wallet: str) -> dict:
    """Build one `stream_trades.py`-shaped raw trade record."""
    return {
        "timestamp": ts,
        "price": price,
        "size": size,
        "wallet": wallet,
        "side": "BUY",
        "transaction_hash": f"0x{i:06d}",
        "condition_id": _COND_ID,
        "asset_id": "1",
    }


def _write_trades(path: Path, records: list[dict]) -> Path:
    """Write raw trade records as a JSONL capture, one object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _stream_fixture(n: int = 40, *, constant_size: bool = False) -> list[dict]:
    """Deterministic pseudo-market: strictly ordered trades, mixed wallets."""
    rng = np.random.default_rng(7)
    prices = np.clip(0.4 + np.cumsum(rng.normal(0.0, 0.02, n)), 0.05, 0.95)
    sizes = np.full(n, 12.0) if constant_size else 5.0 + 40.0 * rng.random(n)
    return [
        _raw_trade(
            i,
            ts=1_700_000_000 + 3 * i,
            price=float(prices[i]),
            size=float(sizes[i]),
            wallet=_STREAM_WALLETS[i % len(_STREAM_WALLETS)],
        )
        for i in range(n)
    ]


def _fake_vem(theta_w: np.ndarray) -> VEMOutput:
    """A minimal fitted VEMOutput carrying non-trivial centering constants."""
    params = ModelParams(
        sigma2_0=0.05,
        sigma2_1=0.4,
        tau2_0=0.3,
        tau2_1=0.01,
        beta_S=0.8,
        beta_Z=0.5,
    )
    n = theta_w.size
    return VEMOutput(
        params=params,
        theta_w=theta_w,
        Z_prob=[],
        V_prob=[],
        X_mean=[],
        elbo_trace=np.zeros(1),
        n_iter_run=1,
        # Deliberately away from (0, 1, 0): a scorer that dropped them would
        # still run, and only these values make the check bite.
        m_S=0.3,
        s_S=1.7,
        m_Z=0.2,
        theta_w_logit_mean=logit(theta_w),
        theta_w_logit_var=np.full(n, 0.1),
        beta_S_orig=0.8,
        beta_Z_orig=0.5,
        beta_fisher_info=np.eye(2),
    )


def test_score_stream_replay_is_byte_identical_across_runs(tmp_path):
    """Same capture + same warm start replays to the same bytes twice."""
    capture = _write_trades(tmp_path / "trades.jsonl", _stream_fixture())
    vem = _fake_vem(np.array([0.1, 0.05, 0.2]))
    warm = tmp_path / "warm.json"
    warm.write_text(
        json.dumps(stream_scoring.warm_start_payload(vem)), encoding="utf-8"
    )

    outputs = []
    for run in ("a", "b"):
        out = tmp_path / f"scores_{run}.jsonl"
        rc = score_stream.main(
            [
                "--replay",
                str(capture),
                "--warm-start",
                str(warm),
                "--output",
                str(out),
                "--log-level",
                "WARNING",
            ]
        )
        assert rc == 0
        outputs.append(out.read_bytes())

    assert outputs[0] == outputs[1]
    assert len(_read_jsonl(tmp_path / "scores_a.jsonl")) == 40


def test_score_stream_replay_has_no_lookahead(tmp_path):
    """Deleting the tail of a capture leaves every surviving score unchanged."""
    records = _stream_fixture(40)
    full = _write_trades(tmp_path / "full.jsonl", records)
    prefix = _write_trades(tmp_path / "prefix.jsonl", records[:25])

    scores = []
    for name, capture in (("full", full), ("prefix", prefix)):
        out = tmp_path / f"{name}.scores.jsonl"
        # Default forgetting (adaptation on) is the strong version of the
        # invariant: the parameters themselves must depend only on the past.
        assert (
            score_stream.main(
                [
                    "--replay",
                    str(capture),
                    "--output",
                    str(out),
                    "--log-level",
                    "WARNING",
                ]
            )
            == 0
        )
        scores.append(out.read_text(encoding="utf-8").splitlines())

    assert len(scores[0]) == 40 and len(scores[1]) == 25
    assert scores[0][:25] == scores[1]


def test_score_stream_warm_start_restores_centering_constants(tmp_path):
    """A warm-started replay reproduces an in-process OnlineScorer exactly."""
    # Constant sizes pin log(S / S_bar) == 0 for every trade, so the reference
    # run needs no feature bookkeeping and the only thing (m_S, s_S) can do is
    # shift the logistic predictor — which is precisely what is under test.
    records = _stream_fixture(30, constant_size=True)
    capture = _write_trades(tmp_path / "trades.jsonl", records)

    theta_w = np.array([0.03, 0.11, 0.27])
    vem = _fake_vem(theta_w)
    warm = tmp_path / "warm.json"
    warm.write_text(
        json.dumps(stream_scoring.warm_start_payload(vem)), encoding="utf-8"
    )
    index = tmp_path / "wallet_index.json"
    index.write_text(
        json.dumps({w: i for i, w in enumerate(_STREAM_WALLETS)}), encoding="utf-8"
    )

    out = tmp_path / "scores.jsonl"
    assert (
        score_stream.main(
            [
                "--replay",
                str(capture),
                "--warm-start",
                str(warm),
                "--wallet-index",
                str(index),
                "--output",
                str(out),
                "--log-level",
                "WARNING",
            ]
        )
        == 0
    )

    reference = OnlineScorer(
        vem.params,
        vem.theta_w,
        vem.m_S,
        vem.s_S,
        vem.m_Z,
        config=OnlineScorerConfig(),
    )
    expected = []
    prev_ts = None
    for i, rec in enumerate(records):
        delta = 0.0 if prev_ts is None else rec["timestamp"] - prev_ts
        prev_ts = rec["timestamp"]
        step = reference.step(
            float(logit(rec["price"])), delta, 0.0, i % len(_STREAM_WALLETS)
        )
        expected.append(step.Z_prob)

    # Exact, not approximate: JSON round-trips a float's repr losslessly, so
    # any difference at all would mean the CLI took a different code path.
    got = [r["p_z"] for r in _read_jsonl(out)]
    assert got == expected

    # And the constants are load-bearing: a cold start scores differently.
    cold = tmp_path / "cold.jsonl"
    assert (
        score_stream.main(
            ["--replay", str(capture), "--output", str(cold), "--log-level", "WARNING"]
        )
        == 0
    )
    assert [r["p_z"] for r in _read_jsonl(cold)] != pytest.approx(expected)


def _validate_vem_shaped_artifact(vem: VEMOutput, *, centering: bool) -> dict:
    """A `validate_vem.py`-layout artifact, with or without its centering keys.

    ``centering=False`` reproduces the older artifact that carried ``params``
    only — the shape whose standardized betas used to be applied to raw
    covariates without a word.
    """
    best = {
        "index": 0,
        "seed": 7,
        **stream_scoring.warm_start_payload(vem),
        "beta_S_orig": float(vem.beta_S_orig),
        "beta_Z_orig": float(vem.beta_Z_orig),
    }
    if not centering:
        dropped = ("theta_w", "m_S", "s_S", "m_Z")
        best = {k: v for k, v in best.items() if k not in dropped}
    return {"config": {"n_restarts": 1}, "best_restart": best}


def test_score_stream_rejects_params_only_warm_start_with_fitted_betas(tmp_path):
    """A centering-less artifact with non-zero betas is refused, not guessed at."""
    vem = _fake_vem(np.array([0.1, 0.05, 0.2]))
    path = tmp_path / "vem_validation.json"
    path.write_text(
        json.dumps(_validate_vem_shaped_artifact(vem, centering=False)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as err:
        stream_scoring.load_warm_start(path)
    # The message has to name both the missing keys and the damage, since the
    # alternative (running anyway) produces plausible-looking wrong scores.
    assert "m_S" in str(err.value) and "s_S" in str(err.value)
    assert "raw covariates" in str(err.value)


def test_score_stream_loads_validate_vem_artifact_as_a_full_warm_start(tmp_path):
    """The current validate_vem.py artifact restores the in-process warm start."""
    vem = _fake_vem(np.array([0.1, 0.05, 0.2]))
    path = tmp_path / "vem_validation.json"
    path.write_text(
        json.dumps(_validate_vem_shaped_artifact(vem, centering=True)),
        encoding="utf-8",
    )

    warm = stream_scoring.load_warm_start(path)
    assert warm.params == vem.params
    assert np.array_equal(warm.theta_w, vem.theta_w)
    assert (warm.m_S, warm.s_S, warm.m_Z) == (vem.m_S, vem.s_S, vem.m_Z)


def test_score_stream_warns_instead_of_raising_when_betas_are_zero(tmp_path, caplog):
    """With an inert logistic predictor, identity centering is exact — so warn."""
    vem = _fake_vem(np.array([0.1, 0.05, 0.2]))
    payload = stream_scoring.warm_start_payload(vem)
    payload["params"]["beta_S"] = 0.0
    payload["params"]["beta_Z"] = 0.0
    for key in ("m_S", "s_S", "m_Z"):
        del payload[key]
    path = tmp_path / "warm.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with caplog.at_level("WARNING"):
        warm = stream_scoring.load_warm_start(path)

    assert (warm.m_S, warm.s_S, warm.m_Z) == (0.0, 1.0, 0.0)
    assert "m_S" in caplog.text and "cold-start" in caplog.text


def test_score_stream_replay_sorts_out_of_order_input(tmp_path):
    """Replay imposes (timestamp, tx_hash) order regardless of file order."""
    records = _stream_fixture(12)
    # Same-second pair so the hash tie-break is exercised, then a shuffle.
    records[5]["timestamp"] = records[4]["timestamp"]
    shuffled = _write_trades(tmp_path / "shuffled.jsonl", list(reversed(records)))
    ordered = _write_trades(
        tmp_path / "ordered.jsonl",
        sorted(records, key=lambda r: (r["timestamp"], r["transaction_hash"])),
    )

    outs = []
    for name, capture in (("shuffled", shuffled), ("ordered", ordered)):
        out = tmp_path / f"{name}.scores.jsonl"
        assert (
            score_stream.main(
                [
                    "--replay",
                    str(capture),
                    "--output",
                    str(out),
                    "--log-level",
                    "WARNING",
                ]
            )
            == 0
        )
        outs.append(out.read_bytes())

    assert outs[0] == outs[1]
    ts = [r["ts"] for r in _read_jsonl(tmp_path / "shuffled.scores.jsonl")]
    assert ts == sorted(ts)


def test_score_stream_live_rejects_out_of_order_input(tmp_path, caplog):
    """Live mode refuses to reorder a backwards sink and says why."""
    records = _stream_fixture(6)
    records[3]["timestamp"] = records[0]["timestamp"] - 60
    capture = _write_trades(tmp_path / "trades.jsonl", records)
    out = tmp_path / "scores.jsonl"

    with caplog.at_level("ERROR"):
        rc = score_stream.main(
            ["--live", str(capture), "--output", str(out), "--log-level", "ERROR"]
        )

    assert rc == 1
    assert "went backwards" in caplog.text
    # The three trades that arrived in order were still scored and flushed.
    assert len(_read_jsonl(out)) == 3


def test_score_stream_live_scores_arrival_order_and_dedupes_on_restart(
    tmp_path, monkeypatch
):
    """A restarted live run appends to its scores without duplicating them.

    `tail_live` re-reads the sink from byte 0 on every start and the output is
    opened for append, so the only thing standing between a restart and a
    doubled scores file is the dedupe set being seeded from that file.
    """
    records = _stream_fixture(8)
    capture = _write_trades(tmp_path / "trades.jsonl", records)
    out = tmp_path / "scores.jsonl"

    def interrupt_instead_of_polling(_seconds):
        # The second run scores nothing new, so --max-trades never fires and it
        # reaches the poll loop — which in a real deployment ends at Ctrl-C.
        raise KeyboardInterrupt

    monkeypatch.setattr(trade_stream.time, "sleep", interrupt_instead_of_polling)

    for _ in range(2):
        # --max-trades bounds the first run so it stops before polling.
        assert (
            score_stream.main(
                [
                    "--live",
                    str(capture),
                    "--output",
                    str(out),
                    "--max-trades",
                    "8",
                    "--log-level",
                    "WARNING",
                ]
            )
            == 0
        )

    scored = _read_jsonl(out)
    # Not 16: the second run recognized all 8 hashes as already scored. And not
    # 0 either — the file was appended to, never truncated.
    assert len(scored) == 8
    assert [r["ts"] for r in scored] == [r["timestamp"] for r in records]


def test_score_stream_live_accepts_same_second_trades_in_any_hash_order(tmp_path):
    """Same-second arrivals are not a time regression, whatever their hashes.

    Polymarket timestamps are second-resolution, so a busy market appends
    several trades per second; enforcing replay's (timestamp, tx_hash) tie-break
    on a live sink would abort on roughly half of those pairs.
    """
    records = _stream_fixture(2)
    records[1]["timestamp"] = records[0]["timestamp"]
    records[0]["transaction_hash"] = "0xffffff"
    records[1]["transaction_hash"] = "0x000001"
    capture = _write_trades(tmp_path / "trades.jsonl", records)
    out = tmp_path / "scores.jsonl"

    assert (
        score_stream.main(
            [
                "--live",
                str(capture),
                "--output",
                str(out),
                "--max-trades",
                "2",
                "--log-level",
                "WARNING",
            ]
        )
        == 0
    )
    assert [r["tx_hash"] for r in _read_jsonl(out)] == ["0xffffff", "0x000001"]


def test_tail_live_holds_back_a_torn_line_until_it_is_whole(tmp_path, monkeypatch):
    """A newline-less tail is the writer mid-record, not a record to parse."""
    records = _stream_fixture(2)
    path = tmp_path / "trades.jsonl"
    # Second record deliberately unterminated: exactly what a reader sees when
    # it catches `stream_trades.py` between write and flush.
    path.write_text(
        json.dumps(records[0]) + "\n" + json.dumps(records[1]), encoding="utf-8"
    )

    class _PolledDry(RuntimeError):
        """Sentinel: the tail loop polled with nothing left to yield."""

    polls = {"n": 0}

    def fake_sleep(_seconds):
        polls["n"] += 1
        if polls["n"] == 1:
            # The writer finishes the torn record between two polls.
            with path.open("a", encoding="utf-8") as fh:
                fh.write("\n")
        elif polls["n"] >= 3:
            raise _PolledDry

    monkeypatch.setattr(trade_stream.time, "sleep", fake_sleep)
    tail = trade_stream.tail_live(path, poll_interval=0.0)

    assert next(tail)["transaction_hash"] == records[0]["transaction_hash"]
    # The fragment was never yielded truncated: the loop rewound and waited.
    assert next(tail)["transaction_hash"] == records[1]["transaction_hash"]
    assert polls["n"] == 1
    # And it arrived once, not once per poll — the sink is dry from here on.
    with pytest.raises(_PolledDry):
        next(tail)


def test_score_stream_writes_a_deterministic_run_sidecar(tmp_path):
    """`<output>.meta.json` describes the run and repeats byte-for-byte."""
    capture = _write_trades(tmp_path / "trades.jsonl", _stream_fixture(6))
    out = tmp_path / "scores.jsonl"
    sidecar = tmp_path / "scores.jsonl.meta.json"

    sidecars = []
    for _ in range(2):
        assert (
            score_stream.main(
                [
                    "--replay",
                    str(capture),
                    "--output",
                    str(out),
                    "--forgetting",
                    "0.97",
                    "--log-level",
                    "WARNING",
                ]
            )
            == 0
        )
        sidecars.append(sidecar.read_bytes())

    # Deterministic: no clock reading anywhere in the payload.
    assert sidecars[0] == sidecars[1]
    meta = json.loads(sidecars[0])
    assert meta == {
        "mode": "replay",
        "input": str(capture),
        "forgetting": 0.97,
        "n_refresh": OnlineScorerConfig.n_refresh,
        "warm_start": None,
        "wallet_index": None,
        "output": str(out),
    }


def test_score_stream_replays_parquet_and_skips_dirty_rows(tmp_path):
    """Parquet replay matches JSONL, and unusable rows are dropped not scored."""
    records = _stream_fixture(10)
    dirty = records + [
        _raw_trade(900, ts=1_700_000_100, price=1.0, size=5.0, wallet="0xw0"),
        _raw_trade(901, ts=1_700_000_101, price=0.5, size=0.0, wallet="0xw0"),
        dict(records[0]),  # duplicate transaction_hash
    ]
    # Uncoercible field: float("n/a") would abort the run and take every
    # scorer's state with it, instead of dropping one unusable row. JSONL only —
    # a mixed str/float column has no parquet dtype, and the schema-less sink is
    # where such a row can actually turn up.
    unparseable = dict(
        _raw_trade(902, ts=1_700_000_102, price=0.5, size=5.0, wallet="0xw0"),
        price="n/a",
    )
    jsonl_out = tmp_path / "jsonl.scores.jsonl"
    assert (
        score_stream.main(
            [
                "--replay",
                str(_write_trades(tmp_path / "trades.jsonl", dirty + [unparseable])),
                "--output",
                str(jsonl_out),
                "--log-level",
                "WARNING",
            ]
        )
        == 0
    )

    parquet = tmp_path / "trades.parquet"
    pd.DataFrame(dirty).to_parquet(parquet, index=False)
    parquet_out = tmp_path / "parquet.scores.jsonl"
    assert (
        score_stream.main(
            [
                "--replay",
                str(parquet),
                "--output",
                str(parquet_out),
                "--log-level",
                "WARNING",
            ]
        )
        == 0
    )

    assert len(_read_jsonl(jsonl_out)) == 10
    assert jsonl_out.read_bytes() == parquet_out.read_bytes()
