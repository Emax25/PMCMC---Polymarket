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

import math
import textwrap
from pathlib import Path
from typing import Iterable, Mapping

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
from src.analysis.sbc import RANK_INTERPRETATION_KEY, UniformityResult
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


# ---------------- VEM validation figures ----------------


def plot_elbo_traces(
    traces: Iterable[np.ndarray],
    *,
    labels: Iterable[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay the per-restart ELBO (log-marginal) trace of a multi-start VEM fit.

    One line per restart. Restarts that climb to visibly different plateaus are
    the multimodality signal the stability check exists to expose; traces that
    land on top of each other are the reassuring case.

    Args:
        traces: One 1-D array per restart, each holding the ELBO proxy per EM
            iteration. Traces may differ in length (early convergence stops
            the loop at different iterations).
        labels: Legend entries aligned with ``traces``; defaults to
            ``restart 0, restart 1, ...``.
        ax: Axes to draw on; a new single-panel figure is created if None.

    Returns:
        Axes containing one ELBO trace per restart.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.0))
    traces = list(traces)
    names = (
        list(labels)
        if labels is not None
        else [f"restart {i}" for i in range(len(traces))]
    )
    for i, (trace, name) in enumerate(zip(traces, names)):
        values = np.asarray(trace, dtype=float)
        ax.plot(
            np.arange(1, values.size + 1),
            values,
            color=f"C{i % 10}",
            lw=1.0,
            marker="o",
            label=name,
        )
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("ELBO proxy (ADF log-marginal)")
    if traces:
        ax.legend(loc="lower right", fontsize=7)
    return ax


def _log_self_normalize(log_weights: np.ndarray) -> np.ndarray:
    """Normalize log weights so their natural-scale exponentials sum to one."""
    x = np.asarray(log_weights, dtype=float)
    shift = x.max()
    return x - (shift + np.log(np.sum(np.exp(x - shift))))


def plot_psis_diagnostic(
    log_weights: np.ndarray,
    log_weights_smoothed: np.ndarray,
    khat: float,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot sorted raw vs Pareto-smoothed log importance weights with khat.

    The two curves are the diagnostic's own content: PSIS replaces only the
    largest ~n/5 weights, so the curves coincide over the body and separate in
    the upper tail — the heavy tail that ``khat`` quantifies. Both series are
    *self-normalized* (their natural-scale weights each sum to one), which is
    the form importance sampling actually uses, so the visible gap is a real
    difference in the weight distribution rather than the arbitrary additive
    constant an unnormalized log ratio carries.

    Args:
        log_weights: Raw log importance ratios, shape ``(n_draws,)``.
        log_weights_smoothed: PSIS-smoothed log weights, same shape.
        khat: Fitted generalized-Pareto tail shape; annotated in the title
            together with the Yao et al. (2018) band the value falls in.
        ax: Axes to draw on; a new single-panel figure is created if None.

    Returns:
        Axes containing both weight curves.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.0))
    raw = np.sort(_log_self_normalize(log_weights))
    smoothed = np.sort(_log_self_normalize(log_weights_smoothed))
    ranks = np.arange(1, raw.size + 1)
    ax.plot(ranks, raw, color="C0", lw=1.0, label="raw")
    ax.plot(ranks, smoothed, color="C1", lw=1.0, ls="--", label="PSIS-smoothed")
    band = "good" if khat < 0.5 else ("ok" if khat <= 0.7 else "bad")
    ax.set_title(f"khat = {khat:.3f} ({band})")
    ax.set_xlabel("draw rank (ascending weight)")
    ax.set_ylabel("log self-normalized weight")
    ax.legend(loc="upper left", fontsize=7)
    return ax


def plot_heldout_ll(
    per_market_mean: np.ndarray,
    *,
    labels: Iterable[str] | None = None,
    pooled_mean: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart of per-market held-out one-step predictive log-likelihood.

    Values are per-held-out-trade means so markets with different tail lengths
    are comparable; the pooled mean (trade-weighted) is drawn as a reference
    line. Bars are log densities, hence negative — less negative is better.

    The bars stand on a data-derived baseline rather than zero: held-out log
    densities sit several nats below zero while differing between markets by
    fractions of a nat, so a zero baseline would render every market as the same
    full-height bar and hide the entire signal.

    Args:
        per_market_mean: Mean held-out log predictive density per market,
            shape ``(K,)``.
        labels: Bar labels aligned with ``per_market_mean``; defaults to
            ``market 0, market 1, ...``.
        pooled_mean: Optional pooled (trade-weighted) mean drawn as a dashed
            horizontal reference.
        ax: Axes to draw on; a new single-panel figure is created if None.

    Returns:
        Axes containing the per-market bars.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.0))
    values = np.asarray(per_market_mean, dtype=float)
    names = (
        list(labels)
        if labels is not None
        else [f"market {k}" for k in range(values.size)]
    )
    positions = np.arange(values.size)
    marks = np.append(values, pooled_mean) if pooled_mean is not None else values
    lo, hi = float(np.nanmin(marks)), float(np.nanmax(marks))
    # Pad by the spread, falling back to a fixed margin when every market scores
    # identically (spread 0 would collapse the axis onto the bars).
    pad = 0.1 * (hi - lo) if hi > lo else max(0.05 * abs(hi), 0.1)
    floor = lo - pad
    ax.bar(positions, values - floor, bottom=floor, color="C0", alpha=0.8)
    ax.set_ylim(floor, hi + pad)
    if pooled_mean is not None:
        ax.axhline(
            pooled_mean,
            color="C3",
            lw=1.0,
            ls="--",
            label=f"pooled = {pooled_mean:.3f}",
        )
        ax.legend(loc="best", fontsize=7)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("held-out mean log predictive density")
    return ax


def _format_wall_time(seconds: float) -> str:
    """Format elapsed seconds as a compact human-readable duration."""
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds / 3600.0:.1f} hr"


def pareto_plot(
    rows: pd.DataFrame,
    *,
    title: str | None = None,
    gate_threshold: float = 0.85,
    figsize: tuple[float, float] = (6.0, 4.0),
) -> plt.Figure:
    """Pareto scatter: synthetic pooled ROC AUC vs mean wall-clock per run.

    Each row supplies ``mean_sec_per_run``, ``ci_half_width`` (optional
    horizontal error bar when positive), ``pooled_auc``, and ``label``.
    The x-axis is log-scaled in seconds; a horizontal reference marks
    the synthetic validation gate threshold.

    Args:
        rows: DataFrame with columns ``label``, ``mean_sec_per_run``,
            ``ci_half_width``, and ``pooled_auc``; one point per method
            or configuration.
        title: Optional figure title; omitted when None.
        gate_threshold: Horizontal AUC reference line (default 0.85).
        figsize: Figure dimensions in inches ``(width, height)``.

    Returns:
        Figure containing the Pareto scatter with annotations.

    Raises:
        ValueError: If ``rows`` is empty or required columns are missing.
    """
    required = ("label", "mean_sec_per_run", "ci_half_width", "pooled_auc")
    missing = [c for c in required if c not in rows.columns]
    if missing:
        raise ValueError(f"rows missing columns: {missing}")
    if len(rows) == 0:
        raise ValueError("rows is empty; nothing to plot")

    fig, ax = plt.subplots(figsize=figsize)
    x = rows["mean_sec_per_run"].to_numpy(dtype=float)
    y = rows["pooled_auc"].to_numpy(dtype=float)
    ci = rows["ci_half_width"].to_numpy(dtype=float)
    labels = rows["label"].tolist()

    xerr = np.where(ci > 0.0, ci, np.nan)
    has_xerr = np.any(np.isfinite(xerr))
    if has_xerr:
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="none",
            ecolor="0.45",
            elinewidth=0.9,
            capsize=2.5,
            zorder=1,
        )

    for i, (xi, yi, lab) in enumerate(zip(x, y, labels)):
        color = f"C{i % 10}"
        ax.plot(xi, yi, "o", color=color, ms=7, zorder=2)
        time_txt = _format_wall_time(float(xi))
        ax.annotate(
            lab,
            (xi, yi),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            color=color,
        )
        ax.annotate(
            time_txt,
            (xi, yi),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=6,
            color="0.35",
        )

    ax.axhline(
        gate_threshold,
        color="0.55",
        lw=0.9,
        ls="--",
        label="gate threshold",
        zorder=0,
    )
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock per run (seconds, log scale)")
    ax.set_ylabel("pooled synthetic ROC AUC")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------- SBC rank figures ----------------


def _uniformity_panel_title(result: UniformityResult) -> str:
    """Title carrying the numeric verdict so a panel is readable without the table."""
    verdict = "FLAG" if result.flagged else "ok"
    return f"{result.component}  (p={result.p_value:.3f}, {verdict})"


def plot_rank_ecdf(
    result: UniformityResult,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the rank-PIT ECDF *difference* with its simultaneous uniformity band.

    Differencing against the uniform CDF is what makes the diagnostic legible:
    the raw ECDF of a calibrated run is a diagonal on which no deviation is
    visible, whereas ``ECDF(u) - u`` is flat at zero and the three failure modes
    separate — a dip-then-rise for the U-shape (overconfident), a bulge for the
    inverted-U (underconfident), and a one-sided excursion for a location bias.

    The shaded band is the Dvoretzky-Kiefer-Wolfowitz band: distribution-free
    and valid for the whole curve at once, and Bonferroni-widened across the
    components plotted beside it, so ``1 - alpha`` is the level at which *no
    panel* leaves its band under calibration — the claim a reader makes when
    scanning the grid.

    Args:
        result: Per-component verdict from ``rank_uniformity``.
        ax: Axes to draw on; a new single-panel figure is created if None.

    Returns:
        Axes containing the ECDF-difference curve and its band.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 2.4))
    ordered = np.sort(result.pit)
    steps = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    # Anchor at (0, 0) and (1, 0): the ECDF difference is pinned to zero at both
    # ends of the unit interval, and drawing it keeps the band visually closed.
    x = np.concatenate(([0.0], ordered, [1.0]))
    y = np.concatenate(([0.0], steps, [1.0])) - x

    half = result.band_half_width
    ax.axhspan(
        -half,
        half,
        color="0.90",
        zorder=0,
        label=f"{1.0 - result.alpha:.0%} band (joint over panels)",
    )
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.plot(x, y, color="C0", lw=1.0)
    ax.set_xlim(0.0, 1.0)
    if result.ks_stat <= half:
        # Calibrated panels would otherwise autoscale to the curve and hide the
        # band that makes them readable as "inside the band".
        ax.set_ylim(-1.6 * half, 1.6 * half)
    ax.set_xlabel("rank PIT $u$")
    ax.set_ylabel(r"$\hat{F}(u) - u$")
    ax.set_title(_uniformity_panel_title(result))
    return ax


def plot_rank_histogram(
    result: UniformityResult,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the binned rank histogram against its expected count and binomial band.

    Bins are drawn on the PIT scale (rank / (L + 1)) so panels for different
    components are directly comparable. Bin widths can differ by one rank when
    ``L + 1`` is not a multiple of the bin count, which is why the expected
    count is drawn as a step rather than a single horizontal line.

    Args:
        result: Per-component verdict from ``rank_uniformity``.
        ax: Axes to draw on; a new single-panel figure is created if None.

    Returns:
        Axes containing the histogram, expected step, and band.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 2.4))
    edges = np.asarray(result.bin_edges, dtype=float) / float(result.L + 1)
    widths = np.diff(edges)
    centres = edges[:-1] + 0.5 * widths

    # `fill_between`/`step` need one extra sample to close the final bin.
    ax.fill_between(
        edges,
        np.append(result.band_lo, result.band_lo[-1]),
        np.append(result.band_hi, result.band_hi[-1]),
        step="post",
        color="0.90",
        zorder=0,
        label=f"{1.0 - result.alpha:.0%} band (joint over panels)",
    )
    # The lower band edge is usually 0, so the fill alone covers the whole panel
    # and reads as background; the explicit upper edge is what makes it a band.
    ax.step(
        edges,
        np.append(result.band_hi, result.band_hi[-1]),
        where="post",
        color="0.55",
        lw=0.7,
        zorder=1,
    )
    ax.bar(centres, result.counts, width=widths * 0.9, color="C0", alpha=0.85, zorder=2)
    ax.step(
        edges,
        np.append(result.expected, result.expected[-1]),
        where="post",
        color="C3",
        lw=0.9,
        ls="--",
        zorder=3,
        label="expected",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("rank PIT $u$")
    ax.set_ylabel("replicates")
    ax.set_title(_uniformity_panel_title(result))
    return ax


def figure_sbc_ranks(
    results: Mapping[str, UniformityResult],
    *,
    kind: str = "ecdf",
    ncols: int = 4,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Produce the SBC rank figure — one panel per phi component.

    A single legend and the interpretation key are attached at figure level so
    the panel grid stays uncluttered and the figure is self-explanatory in the
    paper without its caption.

    Args:
        results: Mapping from component name to ``UniformityResult``, as
            returned by ``rank_uniformity``; panel order follows the mapping.
        kind: ``"ecdf"`` for ECDF-difference panels, ``"hist"`` for binned
            rank histograms.
        ncols: Panels per row.
        figsize: Figure dimensions in inches; defaults to a size derived from
            the grid shape.

    Returns:
        Figure containing the panel grid.

    Raises:
        ValueError: If ``results`` is empty or ``kind`` is unrecognized.
    """
    if not results:
        raise ValueError("results is empty; nothing to plot")
    if kind not in ("ecdf", "hist"):
        raise ValueError(f"kind must be 'ecdf' or 'hist'; got {kind!r}")

    items = list(results.items())
    ncols = max(1, min(ncols, len(items)))
    nrows = math.ceil(len(items) / ncols)
    if figsize is None:
        figsize = (2.6 * ncols, 2.3 * nrows + 0.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.reshape(-1)

    draw = plot_rank_ecdf if kind == "ecdf" else plot_rank_histogram
    for ax, (_, result) in zip(flat, items):
        draw(result, ax=ax)
    for ax in flat[len(items) :]:
        ax.set_visible(False)

    handles, labels = flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=7)
    fig.suptitle(
        f"SBC rank {'ECDF difference' if kind == 'ecdf' else 'histograms'} "
        f"(n={items[0][1].n} replicates, L={items[0][1].L} draws)",
        fontsize=10,
    )
    # Wrapped by hand rather than via `wrap=True`: matplotlib's wrapping is
    # measured against the figure width at draw time and silently truncates in
    # the tight-bbox PDF export this repo saves with. The reserved strip is then
    # sized from the actual line count, so the key can never land on the bottom
    # row's axis labels.
    caption = textwrap.wrap(RANK_INTERPRETATION_KEY, width=max(60, 26 * ncols))
    fig.text(
        0.5,
        0.012,
        "\n".join(caption),
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="0.3",
    )
    fig.tight_layout(rect=(0.0, 0.02 + 0.03 * len(caption), 1.0, 0.95))
    return fig
