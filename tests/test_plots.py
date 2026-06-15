"""Smoke tests for src/analysis/plots.py.

Plots are visual artifacts — we don't pixel-test them. Instead we verify:
  * Each function returns the right matplotlib object (Axes / Figure)
  * Calls don't crash on representative inputs (PG output, iPMCMC output,
    synthetic vs real-data markets)
  * `save_paper_figure` writes the expected files
Uses the non-interactive Agg backend so the tests are headless.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from config.default_params import InferenceConfig, ModelParams
from src.analysis.plots import (
    PAPER_RCPARAMS,
    figure_chain_diagnostics,
    figure_market_overview,
    figure_synthetic_validation,
    plot_parameter_density,
    plot_parameter_trace,
    plot_price_track,
    plot_regime_posterior,
    plot_roc,
    plot_wallet_ranking,
    plot_z_posterior,
    save_paper_figure,
    set_paper_style,
)
from src.analysis.results import wallet_ranking
from src.data.polymarket_api import RawTrade
from src.data.preprocess import WalletIndex, build_processed_market
from src.data.synthetic import generate_market
from src.inference.ipmcmc import ipmcmc
from src.inference.particle_gibbs import MarketData, particle_gibbs

# ---------------- Fixtures ----------------


@pytest.fixture(autouse=True)
def _close_figs_after_each_test():
    """Close all Matplotlib figures after each test to free memory."""
    yield
    plt.close("all")


@pytest.fixture
def synth_market():
    """Synthetic market: 60 trades, 10 wallets, 2 insiders."""
    rng = np.random.default_rng(0)
    Y_dummy = rng.standard_normal(200)
    p = ModelParams.warm_start(Y_dummy)
    return generate_market(
        p,
        n_trades=60,
        n_wallets=10,
        n_insider_wallets=2,
        mean_inter_trade_time=1.0,
        rng=np.random.default_rng(3),
    )


def _to_md(mkt):
    """Wrap SyntheticMarket into MarketData inference struct."""
    return MarketData(
        Y=mkt.Y,
        delta=mkt.delta,
        log_size_ratio=np.log(mkt.S / mkt.S_bar),
        wallet_ids=mkt.wallet_ids,
    )


@pytest.fixture
def pg_output(synth_market):
    """PG chain: 12 iters (4 burn-in), N=20."""
    cfg = InferenceConfig(N=20, n_iter=12, n_burnin=4, seed=0)
    return particle_gibbs([_to_md(synth_market)], cfg, rng=np.random.default_rng(0))


@pytest.fixture
def ipmcmc_output(synth_market):
    """iPMCMC chain: 10 iters (3 burn-in), N=20, M=4, P=2."""
    cfg = InferenceConfig(N=20, M=4, P=2, n_iter=10, n_burnin=3, seed=0)
    return ipmcmc([_to_md(synth_market)], cfg, rng=np.random.default_rng(0))


@pytest.fixture
def wallet_index(synth_market):
    """WalletIndex with one entry per distinct wallet ID."""
    idx = WalletIndex()
    for w in range(int(synth_market.wallet_ids.max()) + 1):
        idx.add(f"0xW{w:04d}{'a'*36}")
    return idx


# ---------------- Style + save ----------------


def test_set_paper_style_applies_defaults():
    """PAPER_RCPARAMS values applied to global rcParams."""
    set_paper_style()
    for key, expected in PAPER_RCPARAMS.items():
        assert plt.rcParams[key] == expected, key


def test_save_paper_figure_writes_pdf_and_png(tmp_path):
    """Both .pdf and .png outputs exist and are non-empty."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = save_paper_figure(fig, "smoke", directory=tmp_path)
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


# ---------------- Single-panel plots ----------------


def test_plot_price_track_returns_axes_on_synthetic(synth_market, pg_output):
    """Returns Axes with at least 2 lines and a legend."""
    ax = plot_price_track(synth_market, pg_output, n_burnin=4)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) >= 2  # observed + smoothed
    assert ax.get_legend() is not None


def test_plot_price_track_on_real_market(pg_output):
    """plot_price_track must also accept a ProcessedMarket (no truth latents)."""
    trades = [
        RawTrade.from_dict(
            {
                "proxyWallet": "0xA",
                "side": "BUY",
                "asset": "1",
                "conditionId": "0xab",
                "size": "1.0",
                "price": str(0.4 + 0.001 * i),
                "timestamp": str(1700000000 + i),
                "transactionHash": f"0xT{i:02d}",
            }
        )
        for i in range(60)
    ]
    market = build_processed_market(trades, wallet_index=WalletIndex())
    # Synthesize a matching PG output by re-running on the real market's data
    cfg = InferenceConfig(N=20, n_iter=10, n_burnin=2, seed=0)
    md = market.to_market_data()
    out = particle_gibbs([md], cfg, rng=np.random.default_rng(0))
    ax = plot_price_track(market, out, n_burnin=2)
    assert isinstance(ax, plt.Axes)


def test_plot_z_posterior_overlays_ground_truth(pg_output, synth_market):
    """Legend references ground truth when Z provided."""
    ax = plot_z_posterior(
        pg_output,
        n_burnin=4,
        ground_truth_Z=synth_market.Z,
    )
    assert isinstance(ax, plt.Axes)
    # Legend should mention ground truth
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("true insider" in t for t in legend_texts)


def test_plot_z_posterior_without_truth(ipmcmc_output):
    """Runs without ground_truth_Z (iPMCMC output)."""
    ax = plot_z_posterior(ipmcmc_output, n_burnin=3)
    assert isinstance(ax, plt.Axes)


def test_plot_regime_posterior_works(pg_output, synth_market):
    """Returns Axes without error."""
    ax = plot_regime_posterior(
        pg_output,
        n_burnin=4,
        ground_truth_V=synth_market.V,
    )
    assert isinstance(ax, plt.Axes)


def test_plot_wallet_ranking_highlights_insiders(
    pg_output,
    synth_market,
    wallet_index,
):
    """y-tick count equals min(top_k, n_wallets)."""
    df = wallet_ranking(pg_output, wallet_index, n_burnin=4)
    insider_addrs = {df["wallet_address"].iloc[i] for i in range(min(2, len(df)))}
    ax = plot_wallet_ranking(df, top_k=5, insider_addresses=insider_addrs)
    assert isinstance(ax, plt.Axes)
    # y-tick labels equal number of rows shown
    assert len(ax.get_yticklabels()) == min(5, len(df))


def test_plot_roc_basic(pg_output, synth_market):
    """Returns Axes for random scores."""
    rng = np.random.default_rng(0)
    z_true = (rng.random(200) < 0.2).astype(int)
    z_score = rng.random(200) + 0.5 * z_true
    ax = plot_roc(z_true, z_score, label="PG")
    assert isinstance(ax, plt.Axes)


def test_plot_parameter_trace_pg(pg_output):
    """Trace plot returns Axes with at least one line."""
    ax = plot_parameter_trace(pg_output, "sigma2_0", n_burnin=4)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) >= 1


def test_plot_parameter_trace_ipmcmc_one_line_per_chain(ipmcmc_output):
    """One solid trace per chain (P=2) plus a dashed burn-in marker line."""
    ax = plot_parameter_trace(ipmcmc_output, "sigma2_0", n_burnin=3)
    assert isinstance(ax, plt.Axes)
    # One trace line per chain (P=2), plus a dashed burn-in marker line.
    solid_lines = [ln for ln in ax.lines if ln.get_linestyle() == "-"]
    assert len(solid_lines) == 2


def test_plot_parameter_density_with_truth(pg_output):
    """Legend shows truth when true_value provided."""
    ax = plot_parameter_density(
        pg_output,
        "tau2_0",
        n_burnin=4,
        true_value=0.5,
    )
    assert isinstance(ax, plt.Axes)
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("truth" in t for t in legend_texts)


# ---------------- Multi-panel composites ----------------


def test_figure_market_overview_three_panels(synth_market, pg_output):
    """Composite figure has exactly 3 axes."""
    fig = figure_market_overview(synth_market, pg_output, n_burnin=4)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3


def test_figure_market_overview_handles_ipmcmc(synth_market, ipmcmc_output):
    """Composite figure works with iPMCMC output."""
    fig = figure_market_overview(synth_market, ipmcmc_output, n_burnin=3)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3


def test_figure_chain_diagnostics_one_row_per_param(pg_output):
    """2 params × (trace + density) = 4 axes."""
    fig = figure_chain_diagnostics(
        pg_output,
        n_burnin=4,
        param_names=("sigma2_0", "tau2_0"),
    )
    assert isinstance(fig, plt.Figure)
    # 2 params × (trace + density) = 4 axes
    assert len(fig.axes) == 4


def test_figure_chain_diagnostics_default_covers_phi(pg_output):
    """Default params: 8 phi × 2 columns = 16 axes."""
    fig = figure_chain_diagnostics(pg_output, n_burnin=4)
    # 8 phi params × 2 columns
    assert len(fig.axes) == 16


def test_figure_synthetic_validation_one_curve_per_run():
    """Legend contains one entry per labelled run."""
    rng = np.random.default_rng(0)
    runs = []
    for label in ("PG", "iPMCMC"):
        z_true = (rng.random(300) < 0.2).astype(int)
        z_score = rng.random(300) + 0.4 * z_true
        runs.append((label, z_true, z_score))
    fig = figure_synthetic_validation(runs)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # ROC curve + diagonal × 2 runs
    labelled = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("PG" in t for t in labelled)
    assert any("iPMCMC" in t for t in labelled)
