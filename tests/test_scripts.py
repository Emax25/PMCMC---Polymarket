"""End-to-end smoke tests for the four CLI scripts.

All paths exercise the `--synthetic` mode of the runners + the offline
fixtures for pull_data.py, so the suite never hits Polymarket.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from scripts import _runner, make_figures, pull_data, run_ipmcmc, run_pg
from src.data.polymarket_api import MarketMeta, RawTrade
from src.inference.ipmcmc import iPMCMCOutput
from src.inference.particle_gibbs import PGOutput

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
