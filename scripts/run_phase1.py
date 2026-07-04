"""Phase 1 — scale the evidence across four independent themes.

Phase 0 found distance-decay in ONE theme (AI buildout) at p ~ 0.08 with 39
edges. Phase 1 adds three unrelated narratives — GLP-1 drugs, EV/battery, and
COVID (a different era entirely) — and upgrades the statistics:

- degree-preserving graph-permutation null (per theme), Fisher-combined
- pooled Mann-Whitney across all themes' pairs
- direction test restricted to edges with peak lag >= 1 day

Usage:  python scripts/run_phase1.py
Output: results/phase1_summary.md, phase1_decay.png, phase1_pairs.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention_flow.analysis import (
    attention_shocks,
    direction_test,
    distance_profile,
    fisher_combine,
    mann_whitney_p,
    pair_peaks,
    permutation_test,
    score_edges,
)
from attention_flow.graph import THEMES
from attention_flow.plots import plot_theme_decay
from attention_flow.wiki import load_panel

RESULTS = Path(__file__).resolve().parents[1] / "results"
DATA = Path(__file__).resolve().parents[1] / "data" / "raw"

N_PERM = 1000


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    theme_rows = []
    theme_profiles: dict[str, dict[int, np.ndarray]] = {}
    all_pairs: list[pd.DataFrame] = []
    perm_ps: list[float] = []
    dir_n_total = dir_wins_total = 0

    for name, theme in THEMES.items():
        start, end = theme.window
        print(f"\n--- {name} ({start}-{end}, {len(theme.nodes)} nodes, {len(theme.edges)} edges)")
        panel = load_panel(theme.nodes, start, end, cache_dir=DATA)
        shocks = attention_shocks(panel)
        peaks = pair_peaks(shocks)
        edges = theme.edge_set()

        pair_df, by_d = distance_profile(shocks, edges, peaks)
        pair_df["theme"] = name
        all_pairs.append(pair_df)
        theme_profiles[name] = by_d

        obs, p_perm = permutation_test(peaks, list(shocks.columns), edges, n_perm=N_PERM)
        perm_ps.append(p_perm)

        results = score_edges(shocks, theme.edges)
        n_dir, wins_dir, p_dir = direction_test(results)
        dir_n_total += n_dir
        dir_wins_total += wins_dir

        theme_rows.append(
            {
                "theme": name,
                "days": panel.shape[0],
                "nodes": panel.shape[1],
                "edges": len(results),
                "decay_stat": obs,
                "perm_p": p_perm,
                "dir_n": n_dir,
                "dir_wins": wins_dir,
                "dir_p": p_dir,
            }
        )
        print(f"    decay stat (1hop - >=3hop) = {obs:.4f}, permutation p = {p_perm:.4f}")
        print(f"    direction (lag>=1d edges): {wins_dir}/{n_dir} forward, p = {p_dir:.3f}")

    pooled_df = pd.concat(all_pairs, ignore_index=True)
    pooled_df.to_csv(RESULTS / "phase1_pairs.csv", index=False)
    pooled_profile = {
        int(d): g.peak_corr.to_numpy() for d, g in pooled_df.groupby("distance")
    }
    near = pooled_df[pooled_df.distance == 1].peak_corr.to_numpy()
    far = pooled_df[pooled_df.distance >= 3].peak_corr.to_numpy()
    p_pooled_mw = mann_whitney_p(near, far)
    p_fisher = fisher_combine(perm_ps)

    from math import comb

    p_dir_pooled = (
        sum(comb(dir_n_total, k) for k in range(dir_wins_total, dir_n_total + 1))
        / 2**dir_n_total
    )

    plot_theme_decay(theme_profiles, pooled_profile, RESULTS / "phase1_decay.png")

    tdf = pd.DataFrame(theme_rows)
    lines = [
        "# Phase 1 — the same test, four independent narratives",
        "",
        "| theme | days | nodes | edges | decay stat (1hop − ≥3hop) | permutation p | direction (fwd/lagged) | dir p |",
        "|---|---|---|---|---|---|---|---|",
        *[
            f"| {r.theme} | {r.days} | {r.nodes} | {r.edges} | {r.decay_stat:.3f} | "
            f"{r.perm_p:.3f} | {r.dir_wins}/{r.dir_n} | {r.dir_p:.3f} |"
            for r in tdf.itertuples()
        ],
        "",
        "## Pooled",
        "",
        "| distance (hops) | n pairs | mean peak corr |",
        "|---|---|---|",
        *[
            f"| {d} | {len(v)} | {v.mean():.3f} |"
            for d, v in sorted(pooled_profile.items())
        ],
        "",
        f"- Pooled adjacent vs distance>=3 (all themes): Mann-Whitney **p = {p_pooled_mw:.5f}** "
        f"(n = {len(near)} vs {len(far)})",
        f"- Fisher-combined per-theme graph-permutation p ({N_PERM} degree-preserving rewires each): "
        f"**p = {p_fisher:.5f}**",
        f"- Pooled direction test (lag>=1d edges): {dir_wins_total}/{dir_n_total} forward, "
        f"binomial p = {p_dir_pooled:.3f}",
    ]
    (RESULTS / "phase1_summary.md").write_text("\n".join(lines))

    print("\n=== PHASE 1 POOLED ===")
    print(f"  pooled Mann-Whitney (1 hop vs >=3): p = {p_pooled_mw:.5f}")
    print(f"  Fisher-combined permutation p:      p = {p_fisher:.5f}")
    print(f"  pooled direction: {dir_wins_total}/{dir_n_total} forward, p = {p_dir_pooled:.3f}")
    print(f"\nWrote {RESULTS}/phase1_summary.md, phase1_decay.png, phase1_pairs.csv")


if __name__ == "__main__":
    main()
