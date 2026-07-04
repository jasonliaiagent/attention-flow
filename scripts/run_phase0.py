"""Phase-0 falsification test, end to end.

Fetches ~4 years of daily Wikipedia pageviews for every node in the AI-buildout
graph, computes attention shocks, and tests whether shocks propagate along the
hypothesized economic edges better than along random pairs.

Usage:  python scripts/run_phase0.py
Output: results/edge_lags.csv, results/summary.md, results/*.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention_flow.analysis import (
    attention_shocks,
    distance_profile,
    mann_whitney_p,
    null_distribution,
    score_edges,
    summarize,
)
from attention_flow.graph import EDGES, NODES, edge_set
from attention_flow.plots import plot_distance_decay, plot_edge_vs_null, plot_top_edges
from attention_flow.wiki import load_panel

START, END = "20220701", "20260630"
RESULTS = Path(__file__).resolve().parents[1] / "results"
DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print(f"Fetching pageviews for {len(NODES)} entities, {START}–{END} ...")
    panel = load_panel(NODES, START, END, cache_dir=DATA)
    print(f"Panel: {panel.shape[0]} days × {panel.shape[1]} entities")

    shocks = attention_shocks(panel)
    results = score_edges(shocks, EDGES)
    null_peaks = null_distribution(shocks, edge_set())
    stats = summarize(results, null_peaks)

    pair_df, by_distance = distance_profile(shocks, edge_set())
    near = pair_df[pair_df.distance == 1].peak_corr.to_numpy()
    far = pair_df[pair_df.distance >= 3].peak_corr.to_numpy()
    p_decay = mann_whitney_p(near, far)

    df = pd.DataFrame([vars(r) for r in results]).sort_values("peak_corr", ascending=False)
    df.to_csv(RESULTS / "edge_lags.csv", index=False)
    pair_df.to_csv(RESULTS / "pair_distances.csv", index=False)

    plot_edge_vs_null(results, null_peaks, RESULTS / "edges_vs_null.png")
    plot_top_edges(results, RESULTS / "top_edges.png")
    plot_distance_decay(by_distance, RESULTS / "distance_decay.png")

    lines = [
        "# Phase 0 — does attention propagate along economic edges?",
        "",
        f"- Window: {START}–{END}, {panel.shape[0]} days, {panel.shape[1]} entities, {stats['n_edges']} edges",
        f"- **Edge mean peak correlation: {stats['edge_mean_peak_corr']:.3f}** "
        f"vs random-pair null {stats['null_mean_peak_corr']:.3f} "
        f"(empirical p = {stats['empirical_p']:.4f})",
        f"- Median peak lag: **{stats['median_peak_lag_days']:.0f} days**; "
        f"{stats['share_lag_ge_1d']:.0%} of edges peak at ≥1 day, "
        f"{stats['share_lag_ge_3d']:.0%} at ≥3 days",
        f"- Forward direction beats reverse on {stats['share_forward_beats_reverse']:.0%} of edges",
        "",
        "## Dose-response: co-movement vs graph distance",
        "",
        "| distance (hops) | n pairs | mean peak corr |",
        "|---|---|---|",
        *[
            f"| {d} | {len(v)} | {v.mean():.3f} |"
            for d, v in by_distance.items()
        ],
        "",
        f"Adjacent pairs vs distance>=3 pairs: one-sided Mann-Whitney **p = {p_decay:.4f}**",
        "",
        "## Top edges",
        "",
        df.head(15).to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Read",
        "",
        "- Lag 0 peaks = attention already synchronized (news co-coverage) — not exploitable.",
        "- Lags of days+ with edge >> null = the propagation signal Phase 1 will model.",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines))

    print("\n=== PHASE 0 RESULT ===")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("  --- distance decay ---")
    for d, v in by_distance.items():
        print(f"  distance {d}: n={len(v)}, mean peak corr {v.mean():.3f}")
    print(f"  adjacent vs distance>=3: Mann-Whitney p = {p_decay:.4f}")
    print(f"\nWrote {RESULTS}/edge_lags.csv, summary.md, edges_vs_null.png, top_edges.png")


if __name__ == "__main__":
    main()
