"""Phase-0 figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import EdgeResult


def plot_edge_vs_null(results: list[EdgeResult], null_peaks: np.ndarray, out: Path) -> None:
    edge_peaks = [r.peak_corr for r in results]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-0.1, 0.9, 40)
    ax.hist(null_peaks, bins=bins, alpha=0.6, density=True, label="random non-edge pairs", color="#999")
    ax.hist(edge_peaks, bins=bins, alpha=0.7, density=True, label="economic-graph edges", color="#d95f02")
    ax.axvline(np.mean(null_peaks), color="#555", ls="--", lw=1)
    ax.axvline(np.mean(edge_peaks), color="#a33e00", ls="--", lw=1)
    ax.set_xlabel("peak lagged correlation of attention shocks (lags 0–30d)")
    ax.set_ylabel("density")
    ax.set_title("Attention co-movement: economic edges vs random pairs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_distance_decay(by_distance: dict[int, np.ndarray], out: Path) -> None:
    """The dose-response curve: attention co-movement vs graph distance."""
    dists = sorted(by_distance)
    means = [by_distance[d].mean() for d in dists]
    sems = [by_distance[d].std() / np.sqrt(len(by_distance[d])) for d in dists]
    ns = [len(by_distance[d]) for d in dists]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(dists, means, yerr=sems, marker="o", capsize=4, color="#d95f02", lw=2)
    for d, m, n in zip(dists, means, ns):
        ax.annotate(f"n={n}", (d, m), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xlabel("graph distance between entities (hops)")
    ax.set_ylabel("mean peak lagged attention correlation")
    ax.set_title("Attention co-movement decays with economic-graph distance")
    ax.set_xticks(dists)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_theme_decay(
    theme_profiles: dict[str, dict[int, np.ndarray]],
    pooled: dict[int, np.ndarray],
    out: Path,
) -> None:
    """Distance-decay curves for every theme plus the pooled curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"ai-buildout": "#d95f02", "glp1": "#7570b3", "ev-battery": "#1b9e77", "covid": "#e7298a"}
    for theme, by_d in theme_profiles.items():
        dists = sorted(by_d)
        ax.plot(
            dists,
            [by_d[d].mean() for d in dists],
            marker="o",
            lw=1.2,
            alpha=0.55,
            label=theme,
            color=colors.get(theme, "#666"),
        )
    dists = sorted(pooled)
    means = [pooled[d].mean() for d in dists]
    sems = [pooled[d].std() / np.sqrt(len(pooled[d])) for d in dists]
    ax.errorbar(dists, means, yerr=sems, marker="o", lw=3, capsize=4, color="#222", label="pooled (all themes)")
    ax.set_xlabel("graph distance between entities (hops)")
    ax.set_ylabel("mean peak lagged attention correlation")
    ax.set_title("Attention co-movement vs economic-graph distance, four independent themes")
    ax.set_xticks(dists)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_proxy_replication(
    profiles: dict[str, dict[int, np.ndarray]],
    out: Path,
    title: str = "Attention co-movement vs graph distance, by proxy (AI theme)",
) -> None:
    """Decay curves for the same graph under different attention proxies."""
    colors = {"Wikipedia pageviews": "#d95f02", "GDELT news volume": "#1b9e77"}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, by_d in profiles.items():
        dists = sorted(by_d)
        means = [by_d[d].mean() for d in dists]
        sems = [by_d[d].std() / np.sqrt(len(by_d[d])) for d in dists]
        ax.errorbar(
            dists, means, yerr=sems, marker="o", lw=2, capsize=4,
            label=label, color=colors.get(label, "#666"),
        )
    ax.set_xlabel("graph distance between entities (hops)")
    ax.set_ylabel("mean peak lagged attention correlation")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_top_edges(results: list[EdgeResult], out: Path, top_n: int = 20) -> None:
    top = sorted(results, key=lambda r: r.peak_corr, reverse=True)[:top_n]
    labels = [f"{r.src} → {r.dst}" for r in top][::-1]
    corrs = [r.peak_corr for r in top][::-1]
    lags = [r.peak_lag for r in top][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(top) + 1.5))
    bars = ax.barh(labels, corrs, color="#1b9e77")
    for bar, lag in zip(bars, lags):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"lag {lag}d",
            va="center",
            fontsize=8,
            color="#333",
        )
    ax.set_xlabel("peak lagged correlation (attention shocks)")
    ax.set_title("Strongest attention-propagation edges")
    ax.set_xlim(0, max(corrs) * 1.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
