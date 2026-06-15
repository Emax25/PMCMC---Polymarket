"""Paper-figure helpers for the §5 Polymarket application and §9 validation.

Every plot function returns the `Axes` it drew on so the caller can compose
multi-panel figures. Each function takes an optional `ax=None` argument that
creates a single-panel figure if omitted — convenient for notebook use.

The §5 figures are produced by `figure_market_overview` (3-panel: price track,
P(Z=1|D), P(V=1|D)) and `figure_wallet_ranking` (single-panel forest plot).
Synthetic §9 figures use `figure_synthetic_validation` (ROC + posterior-mean
recovery scatter).

Per §10 of README: matplotlib only, no seaborn-specific plots, all figures
keep their LaTeX-friendly defaults (vector PDF + serif font sizes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.results import (
    PGorIP,
    flagged_trade_indices,
    posterior_pi_mean,
    posterior_regime_probability,
    posterior_Z_probability,
    roc_auc,
    roc_curve,
)
from src.data.preprocess import ProcessedMarket
from src.data.synthetic import SyntheticMarket
from src.inference.diagnostics import PHI_PARAM_NAMES

# ---------------- Style ----------------

PAPER_RCPARAMS = {
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def set_paper_style() -> None:
    """Apply LaTeX-friendly matplotlib defaults to the current process."""
    plt.rcParams.update(PAPER_RCPARAMS)


def save_paper_figure(
    fig: plt.Figure,
    name: str,
    *,
    directory: str | Path = "results/figures",
    formats: Iterable[str] = ("pdf", "png"),
) -> list[Path]:
    """Save a figure under ``directory`` in each requested format.

    Args:
        fig: Matplotlib figure to save.
        name: Base filename without extension.
        directory: Destination directory; created (including parents) if
            it does not exist.
        formats: Iterable of file extensions, e.g. ``("pdf", "png")``.

    Returns:
        List of Paths of the files written, in the order of ``formats``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in formats:
        p = directory / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    return paths


# ---------------- Single-panel plots ----------------


def plot_price_track(
    market: ProcessedMarket | SyntheticMarket,
    out: PGorIP,
    *,
    market_idx: int = 0,
    n_burnin: int = 0,
    flag_threshold: float = 0.5,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot observed trade prices against the smoothed E[π|D] estimate.

    Trades where P(Z_i = 1 | D) >= ``flag_threshold`` are highlighted
    with open circle markers.

    Args:
        market: Processed or synthetic market providing the raw p array.
        out: PG or iPMCMC chain output.
        market_idx: Index of the market within the multi-market run.
        n_burnin: Number of leading iterations to discard as burn-in.
        flag_threshold: P(Z_i = 1 | D) cutoff for highlighting trades.
        ax: Axes to draw on; a new single-panel figure is created if
            None.

    Returns:
        Axes containing the price track and insider-flag annotations.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 2.4))

    pi_mean = posterior_pi_mean(out, market_idx, n_burnin)
    z_prob = posterior_Z_probability(out, market_idx, n_burnin)
    flagged = flagged_trade_indices(z_prob, threshold=flag_threshold)
    t_idx = np.arange(len(market.p))

    ax.plot(t_idx, market.p, ".", color="0.65", alpha=0.5, label="observed $p_i$")
    ax.plot(
        t_idx,
        pi_mean,
        "-",
        color="C0",
        label=r"$\mathbb{E}[\pi_{t_i} \mid \mathcal{D}]$",
    )
    if len(flagged) > 0:
        _lbl = rf"flagged $(P(Z_i{{=}}1{{\mid}}\mathcal{{D}})\geq {flag_threshold:g})$"
        ax.plot(
            flagged,
            market.p[flagged],
            "o",
            color="C3",
            ms=4,
            mfc="none",
            mew=1.0,
            label=_lbl,
        )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("trade index $i$")
    ax.set_ylabel("probability")
    ax.legend(loc="best")
    return ax


def plot_z_posterior(
    out: PGorIP,
    *,
    market_idx: int = 0,
    n_burnin: int = 0,
    ground_truth_Z: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot P(Z_i = 1 | D) as a filled area per trade.

    Args:
        out: PG or iPMCMC chain output.
        market_idx: Index of the market within the multi-market run.
        n_burnin: Number of leading iterations to discard as burn-in.
        ground_truth_Z: Optional binary array of shape ``(T,)``; insider
            trades (Z_i = 1) are marked with downward ticks above the
            plot.
        ax: Axes to draw on; a new single-panel figure is created if
            None.

    Returns:
        Axes containing the Z-posterior fill and optional truth markers.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 2.0))
    z_prob = posterior_Z_probability(out, market_idx, n_burnin)
    t_idx = np.arange(len(z_prob))
    ax.fill_between(
        t_idx, 0.0, z_prob, color="C3", alpha=0.35, label=r"$P(Z_i{=}1\mid\mathcal{D})$"
    )
    ax.plot(t_idx, z_prob, color="C3", lw=0.8)
    if ground_truth_Z is not None:
        truth_idx = np.flatnonzero(np.asarray(ground_truth_Z) == 1)
        if len(truth_idx) > 0:
            ax.plot(
                truth_idx,
                np.full_like(truth_idx, 1.02, dtype=float),
                "v",
                color="black",
                ms=4,
                label="true insider trade",
            )
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("trade index $i$")
    ax.set_ylabel(r"$P(Z_i{=}1\mid\mathcal{D})$")
    ax.legend(loc="best")
    return ax


def plot_regime_posterior(
    out: PGorIP,
    *,
    market_idx: int = 0,
    n_burnin: int = 0,
    ground_truth_V: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot P(V_{t_i} = 1 | D) as a filled area per trade.

    Args:
        out: PG or iPMCMC chain output.
        market_idx: Index of the market within the multi-market run.
        n_burnin: Number of leading iterations to discard as burn-in.
        ground_truth_V: Optional binary array of shape ``(T,)``; news-
            regime trades (V_i = 1) are marked with downward ticks above
            the plot.
        ax: Axes to draw on; a new single-panel figure is created if
            None.

    Returns:
        Axes containing the V-posterior fill and optional truth markers.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 2.0))
    v_prob = posterior_regime_probability(out, market_idx, n_burnin)
    t_idx = np.arange(len(v_prob))
    ax.fill_between(
        t_idx,
        0.0,
        v_prob,
        color="C2",
        alpha=0.35,
        label=r"$P(V_{t_i}{=}1\mid\mathcal{D})$",
    )
    ax.plot(t_idx, v_prob, color="C2", lw=0.8)
    if ground_truth_V is not None:
        truth_idx = np.flatnonzero(np.asarray(ground_truth_V) == 1)
        if len(truth_idx) > 0:
            ax.plot(
                truth_idx,
                np.full_like(truth_idx, 1.02, dtype=float),
                "v",
                color="black",
                ms=3,
                label="true news regime",
            )
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("trade index $i$")
    ax.set_ylabel(r"$P(V_{t_i}{=}1\mid\mathcal{D})$")
    ax.legend(loc="best")
    return ax


def plot_wallet_ranking(
    ranking: pd.DataFrame,
    *,
    top_k: int = 20,
    insider_addresses: set[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Forest plot of the top-K wallets by E[θ_w | D] with credible bars.

    If ``insider_addresses`` is provided (synthetic experiments only),
    those wallets are highlighted in red to surface recovery quality
    directly on the figure.

    Args:
        ranking: DataFrame as returned by ``wallet_ranking``; must have
            columns posterior_mean, ci_lo, ci_hi, wallet_address,
            wallet_id.
        top_k: Number of highest-ranked wallets to display.
        insider_addresses: Set of known-insider wallet address strings;
            highlighted in red (C3). Supply only for synthetic
            experiments.
        ax: Axes to draw on; a new figure is created if None.

    Returns:
        Axes containing the forest plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, max(2.5, 0.22 * top_k + 1.0)))
    df = ranking.head(top_k).reset_index(drop=True)
    y = np.arange(len(df))[::-1]
    means = df["posterior_mean"].to_numpy()
    lo = means - df["ci_lo"].to_numpy()
    hi = df["ci_hi"].to_numpy() - means

    colors = []
    insider_addresses = insider_addresses or set()
    for addr in df["wallet_address"]:
        colors.append("C3" if addr in insider_addresses else "C0")
    ax.errorbar(
        means,
        y,
        xerr=[lo, hi],
        fmt="o",
        color="black",
        ecolor="0.6",
        elinewidth=0.8,
        capsize=0,
    )
    for yi, c in zip(y, colors):
        ax.plot([], [], "o", color=c)  # legend dummies handled below
    for xi, yi, c in zip(means, y, colors):
        ax.plot(xi, yi, "o", color=c, ms=4)

    labels = [
        a[:6] + "…" + a[-4:] if len(a) > 12 else (a or f"#{wid}")
        for a, wid in zip(df["wallet_address"], df["wallet_id"])
    ]
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$\mathbb{E}[\theta_w \mid \mathcal{D}]$")
    ax.set_title(f"Top-{top_k} wallets by posterior insider propensity")
    return ax


def plot_roc(
    z_true: np.ndarray,
    z_score: np.ndarray,
    *,
    label: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot an ROC curve with AUC annotation and chance diagonal.

    Args:
        z_true: Binary ground-truth labels, shape ``(T,)``; 1 = insider.
        z_score: Continuous anomaly scores, shape ``(T,)``; higher values
            indicate a more likely insider trade.
        label: Prefix for the legend entry; AUC is appended automatically.
        ax: Axes to draw on; a new single-panel figure is created if
            None.

    Returns:
        Axes containing the ROC curve, chance diagonal, and legend.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.5, 3.0))
    fpr, tpr, _ = roc_curve(z_true, z_score)
    auc = roc_auc(z_true, z_score)
    lab = f"AUC = {auc:.3f}" if label is None else f"{label} (AUC = {auc:.3f})"
    ax.plot(fpr, tpr, label=lab, lw=1.2)
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_parameter_trace(
    out: PGorIP,
    param_name: str,
    *,
    n_burnin: int = 0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Trace plot of one φ parameter over MCMC iterations.

    Draws one line per chain for iPMCMC outputs; a single line for PG.
    A vertical dashed line marks the burn-in boundary when
    ``n_burnin > 0``.

    Args:
        out: PG or iPMCMC chain output.
        param_name: Name of the scalar φ attribute to trace (must be an
            attribute of ``out``).
        n_burnin: Number of leading iterations treated as burn-in; a
            vertical marker is drawn at this iteration.
        ax: Axes to draw on; a new single-panel figure is created if
            None.

    Returns:
        Axes containing the trace lines.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 2.0))
    raw = np.asarray(getattr(out, param_name))
    iters = np.arange(raw.shape[0])
    if raw.ndim == 1:
        ax.plot(iters, raw, color="C0", lw=0.6)
    else:
        for p in range(raw.shape[1]):
            ax.plot(iters, raw[:, p], lw=0.6, alpha=0.8, label=f"chain {p}")
        ax.legend(loc="best", fontsize=7)
    if n_burnin > 0:
        ax.axvline(n_burnin, color="0.6", lw=0.6, ls="--")
    ax.set_xlabel("iteration")
    ax.set_ylabel(param_name)
    return ax


def plot_parameter_density(
    out: PGorIP,
    param_name: str,
    *,
    n_burnin: int = 0,
    true_value: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histogram of post-burn-in samples for one φ parameter.

    Args:
        out: PG or iPMCMC chain output.
        param_name: Name of the scalar φ attribute to plot (must be an
            attribute of ``out``).
        n_burnin: Number of leading iterations to discard as burn-in.
        true_value: Optional ground-truth scalar; drawn as a vertical
            red (C3) line when provided.
        ax: Axes to draw on; a new single-panel figure is created if
            None.

    Returns:
        Axes containing the density histogram.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.5, 2.4))
    raw = np.asarray(getattr(out, param_name))[n_burnin:]
    flat = raw.reshape(-1)
    ax.hist(flat, bins=40, color="C0", alpha=0.7, density=True)
    if true_value is not None:
        ax.axvline(true_value, color="C3", lw=1.0, label=f"truth = {true_value:.3g}")
        ax.legend(loc="best")
    ax.set_xlabel(param_name)
    ax.set_ylabel("density")
    return ax


# ---------------- Multi-panel composites ----------------


def figure_market_overview(
    market: ProcessedMarket | SyntheticMarket,
    out: PGorIP,
    *,
    market_idx: int = 0,
    n_burnin: int = 0,
    flag_threshold: float = 0.5,
    figsize: tuple[float, float] = (7.0, 6.0),
) -> plt.Figure:
    """Produce the §5 flagship 3-panel market overview figure.

    Panels (top to bottom): price track + E[π|D], P(Z=1|D), P(V=1|D).
    Ground-truth Z and V markers are overlaid automatically when
    ``market`` is a ``SyntheticMarket`` (which carries truth latents);
    silently skipped for ``ProcessedMarket``.

    Args:
        market: Processed or synthetic market providing prices and
            optional ground-truth latents.
        out: PG or iPMCMC chain output.
        market_idx: Index of the market within the multi-market run.
        n_burnin: Number of leading iterations to discard as burn-in.
        flag_threshold: P(Z_i = 1 | D) cutoff for highlighting trades
            in the price-track panel.
        figsize: Figure dimensions in inches ``(width, height)``.

    Returns:
        Figure with three vertically stacked, x-axis-shared panels.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    plot_price_track(
        market,
        out,
        market_idx=market_idx,
        n_burnin=n_burnin,
        flag_threshold=flag_threshold,
        ax=axes[0],
    )
    truth_Z = getattr(market, "Z", None)
    truth_V = getattr(market, "V", None)
    plot_z_posterior(
        out,
        market_idx=market_idx,
        n_burnin=n_burnin,
        ground_truth_Z=truth_Z,
        ax=axes[1],
    )
    plot_regime_posterior(
        out,
        market_idx=market_idx,
        n_burnin=n_burnin,
        ground_truth_V=truth_V,
        ax=axes[2],
    )
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    title = getattr(market, "slug", "") or getattr(market, "condition_id", "")
    if title:
        fig.suptitle(title, fontsize=10)
        fig.subplots_adjust(top=0.94)
    fig.tight_layout()
    return fig


def figure_chain_diagnostics(
    out: PGorIP,
    *,
    n_burnin: int = 0,
    param_names: tuple[str, ...] | None = None,
    true_params: dict[str, float] | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Produce a per-φ diagnostics figure with trace and density columns.

    Each row corresponds to one φ parameter in ``param_names``; the left
    column is a trace plot and the right column is a marginal density.

    Args:
        out: PG or iPMCMC chain output.
        n_burnin: Number of leading iterations to discard as burn-in.
        param_names: Parameters to display; defaults to
            ``PHI_PARAM_NAMES``.
        true_params: Optional dict mapping parameter name to its ground-
            truth scalar value; overlaid on density plots.
        figsize: Figure dimensions in inches; defaults to
            ``(8.0, 1.6 * len(param_names))``.

    Returns:
        Figure with ``len(param_names)`` rows and 2 columns (trace,
        density).
    """
    names = param_names or PHI_PARAM_NAMES
    if figsize is None:
        figsize = (8.0, 1.6 * len(names))
    fig, axes = plt.subplots(
        len(names), 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]}
    )
    if len(names) == 1:
        axes = axes[None, :]
    for i, name in enumerate(names):
        plot_parameter_trace(out, name, n_burnin=n_burnin, ax=axes[i, 0])
        true_v = (true_params or {}).get(name)
        plot_parameter_density(
            out,
            name,
            n_burnin=n_burnin,
            true_value=true_v,
            ax=axes[i, 1],
        )
        axes[i, 0].set_xlabel("")
    axes[-1, 0].set_xlabel("iteration")
    fig.tight_layout()
    return fig


def figure_synthetic_validation(
    runs: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    figsize: tuple[float, float] = (3.5, 3.0),
) -> plt.Figure:
    """Stacked ROC curves for §9 — one entry per labelled (sampler) run.

    Args:
        runs: List of ``(label, z_true, z_score)`` tuples; one ROC curve
            is drawn per entry.
        figsize: Figure dimensions in inches ``(width, height)``.

    Returns:
        Figure containing all ROC curves on a single panel.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for label, z_true, z_score in runs:
        plot_roc(z_true, z_score, label=label, ax=ax)
    fig.tight_layout()
    return fig
