"""Phase 1b — direction (transfer entropy) + proxy replication (GDELT).

Phase 1 left two open questions:

1. DIRECTION. The forward/reverse cross-correlation test was flat — but
   cross-correlation is nearly symmetric by construction. Transfer entropy
   asks the right question: does SRC's past reduce uncertainty about DST's
   future beyond DST's own past? Net TE > 0 across edges = downstream flow.

2. PROXY. Distance decay was measured only in Wikipedia pageviews. Here the
   AI-theme graph is re-tested with GDELT global news volume — attention as
   produced by journalists rather than encyclopedia readers.

Usage:  python scripts/run_phase1b.py
Output: results/phase1b_summary.md, phase1b_replication.png, phase1b_te.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention_flow.analysis import attention_shocks, distance_profile, permutation_test, pair_peaks
from attention_flow.gdelt import load_gdelt_panel
from attention_flow.graph import THEMES
from attention_flow.plots import plot_proxy_replication
from attention_flow.te import edge_direction_scores, sign_test
from attention_flow.wiki import load_panel

RESULTS = Path(__file__).resolve().parents[1] / "results"
DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
GDELT_DATA = Path(__file__).resolve().parents[1] / "data" / "gdelt"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    # ---- Part A: transfer-entropy direction test, all four themes ----------
    te_frames = []
    theme_dir_rows = []
    for name, theme in THEMES.items():
        start, end = theme.window
        shocks = attention_shocks(load_panel(theme.nodes, start, end, cache_dir=DATA))
        for freq, weekly in (("daily", False), ("weekly", True)):
            df = edge_direction_scores(shocks, theme.edges, weekly=weekly)
            df["theme"], df["freq"] = name, freq
            te_frames.append(df)
            n, wins, p = sign_test(df.net_te.to_numpy())
            theme_dir_rows.append({"theme": name, "freq": freq, "n": n, "positive": wins, "p": p})
            print(f"{name:12s} {freq:6s}: net-TE > 0 on {wins}/{n} edges (sign p = {p:.3f})")

    te_df = pd.concat(te_frames, ignore_index=True)
    te_df.to_csv(RESULTS / "phase1b_te.csv", index=False)
    pooled_dir = {}
    for freq in ("daily", "weekly"):
        vals = te_df[te_df.freq == freq].net_te.to_numpy()
        pooled_dir[freq] = sign_test(vals)
        n, wins, p = pooled_dir[freq]
        print(f"POOLED {freq}: {wins}/{n} positive, sign p = {p:.4f}")

    # ---- Part B: GDELT replication of the AI-theme distance decay ----------
    theme = THEMES["ai-buildout"]
    start, end = theme.window
    print("\nFetching GDELT news volume for the AI theme ...")
    gdelt_panel = load_gdelt_panel(theme.nodes, start, end, cache_dir=GDELT_DATA)
    print(f"GDELT panel: {gdelt_panel.shape[0]} days x {gdelt_panel.shape[1]} entities")

    g_shocks = attention_shocks(gdelt_panel)
    g_peaks = pair_peaks(g_shocks)
    edges = theme.edge_set()
    _, g_by_d = distance_profile(g_shocks, edges, g_peaks)
    g_obs, g_p = permutation_test(g_peaks, list(g_shocks.columns), edges)

    w_shocks = attention_shocks(load_panel(theme.nodes, start, end, cache_dir=DATA))
    w_peaks = pair_peaks(w_shocks)
    _, w_by_d = distance_profile(w_shocks, edges, w_peaks)

    plot_proxy_replication(
        {"Wikipedia pageviews": w_by_d, "GDELT news volume": g_by_d},
        RESULTS / "phase1b_replication.png",
    )
    print(f"GDELT decay stat = {g_obs:.4f}, permutation p = {g_p:.4f}")

    # ---- summary ------------------------------------------------------------
    lines = [
        "# Phase 1b — direction (transfer entropy) + proxy replication (GDELT)",
        "",
        "## A. Direction: net transfer entropy along hypothesized edges",
        "",
        "| theme | freq | edges | net-TE > 0 | sign p |",
        "|---|---|---|---|---|",
        *[
            f"| {r['theme']} | {r['freq']} | {r['n']} | {r['positive']} | {r['p']:.3f} |"
            for r in theme_dir_rows
        ],
        "",
        f"- POOLED daily: {pooled_dir['daily'][1]}/{pooled_dir['daily'][0]} positive, "
        f"sign p = {pooled_dir['daily'][2]:.4f}",
        f"- POOLED weekly: {pooled_dir['weekly'][1]}/{pooled_dir['weekly'][0]} positive, "
        f"sign p = {pooled_dir['weekly'][2]:.4f}",
        "",
        "## B. GDELT replication (AI theme)",
        "",
        "| distance (hops) | n pairs | GDELT mean peak corr | Wikipedia mean peak corr |",
        "|---|---|---|---|",
        *[
            f"| {d} | {len(g_by_d[d])} | {g_by_d[d].mean():.3f} | {w_by_d[d].mean():.3f} |"
            for d in sorted(set(g_by_d) & set(w_by_d))
        ],
        "",
        f"GDELT decay stat (1hop − ≥3hop) = **{g_obs:.3f}**, "
        f"degree-preserving permutation **p = {g_p:.4f}**",
    ]
    (RESULTS / "phase1b_summary.md").write_text("\n".join(lines))
    print(f"\nWrote {RESULTS}/phase1b_summary.md, phase1b_replication.png, phase1b_te.csv")


if __name__ == "__main__":
    main()
