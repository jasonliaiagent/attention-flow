"""Render a static preview image of the attention map (for README / social).

Reads docs/data.json (the same data the live page uses), runs a small force
layout, and renders the AI-buildout graph in the page's visual language.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BG, COLD, SLATE, EMBER, HOT = "#06080b", "#3f6d9e", "#5c6b76", "#e8842c", "#ff5d3a"


def color(v: float):
    stops = [(-2.5, COLD), (0.0, SLATE), (1.2, EMBER), (2.5, HOT)]
    v = max(-2.5, min(2.5, v))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            ca, cb = matplotlib.colors.to_rgb(ca), matplotlib.colors.to_rgb(cb)
            return tuple(x + (y - x) * t for x, y in zip(ca, cb))
    return matplotlib.colors.to_rgb(HOT)


def layout(n_nodes: int, edges: list[tuple[int, int]], seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    i = np.arange(n_nodes)
    pos = np.stack([16 * np.sqrt(i + 1) * np.cos(i * 2.39996),
                    16 * np.sqrt(i + 1) * np.sin(i * 2.39996)], axis=1)
    pos += rng.normal(0, 2, pos.shape)
    vel = np.zeros_like(pos)
    for _ in range(600):
        d = pos[:, None, :] - pos[None, :, :]
        dist2 = (d**2).sum(-1) + 200
        f = 1300 / dist2
        np.fill_diagonal(f, 0)
        vel += (d * f[:, :, None]).sum(1)
        for a, b in edges:
            delta = pos[b] - pos[a]
            dist = np.hypot(*delta) or 1
            pull = (dist - 105) * 0.008 * delta / dist
            vel[a] += pull
            vel[b] -= pull
        vel += -pos * 2.0e-3
        vel = np.clip(vel * 0.84, -3.5, 3.5)
        pos += vel
    return pos


def main() -> None:
    data = json.loads((ROOT / "docs" / "data.json").read_text())
    theme = data["themes"][0]  # AI buildout
    nodes = theme["nodes"]
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    edges = [(idx[e["source"]], idx[e["target"]]) for e in theme["edges"]]
    pos = layout(len(nodes), edges)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=125)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_position([0, 0, 1, 1])
    for a, b in edges:
        ax.plot(*zip(pos[a], pos[b]), color="#82aacd", alpha=0.13, lw=0.8, zorder=1)
    for n, (x, y) in zip(nodes, pos):
        v = n["now"]
        c = color(v)
        r = 60 + 240 * min(abs(v), 3) / 3
        ax.scatter(x, y, s=r * 4, color=c, alpha=0.15, lw=0, zorder=2)  # glow
        ax.scatter(x, y, s=r, color=c, alpha=0.95, lw=0, zorder=3)
        if abs(v) > 0.5 or n["id"] in ("Nvidia", "Data center", "OpenAI"):
            ax.annotate(n["id"], (x, y), xytext=(8, 3), textcoords="offset points",
                        color="#c8d7e4", fontsize=8, family="monospace", alpha=0.8, zorder=4)
    ax.text(0.03, 0.95, "ATTENTION-FLOW", transform=ax.transAxes, color="#eef4f8",
            fontsize=17, family="monospace", weight="bold")
    ax.text(0.03, 0.905, f"a live map of where the market is looking · data through {data['data_through']}",
            transform=ax.transAxes, color="#5f7181", fontsize=9, family="monospace")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(ROOT / "docs" / "preview.png", facecolor=BG)
    print("Wrote docs/preview.png")


if __name__ == "__main__":
    main()
