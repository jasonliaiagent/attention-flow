"""Phase-2 dataset: sliding windows over the attention-shock panels.

Each sample is one theme at one point in time:
    input   X[n, w]   — the last INPUT_DAYS days of attention shocks per node
    target  y[n]      — each node's mean shock over the NEXT HORIZON_DAYS days

Splits are strictly temporal (train | val | test by target-end date), so the
model never sees the future. Training windows stride 1 day (augmentation);
val/test stride HORIZON_DAYS so reported targets never overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import attention_shocks, graph_distances
from .graph import Theme
from .wiki import load_panel

INPUT_DAYS = 90
HORIZON_DAYS = 7
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10  # remainder = test


@dataclass
class ThemeWindows:
    """All windows for one theme, plus its (dense, undirected) adjacency."""

    name: str
    nodes: list[str]
    adj: np.ndarray  # [n, n] float, symmetric, self-loops included
    X: dict[str, np.ndarray]  # split -> [batch, n, INPUT_DAYS]
    y: dict[str, np.ndarray]  # split -> [batch, n]
    mask: dict[str, np.ndarray]  # split -> [batch, n] bool, valid targets


def build_adjacency(nodes: list[str], edge_pairs: set[tuple[str, str]]) -> np.ndarray:
    """Symmetric adjacency with self-loops. Undirected by design: Phase 1b
    found no directional asymmetry, so the model gets no arrow to lean on."""
    idx = {n: i for i, n in enumerate(nodes)}
    adj = np.eye(len(nodes), dtype=np.float32)
    for a, b in edge_pairs:
        if a in idx and b in idx:
            adj[idx[a], idx[b]] = adj[idx[b], idx[a]] = 1.0
    return adj


def build_theme_windows(theme: Theme, cache_dir: Path) -> ThemeWindows:
    panel = load_panel(theme.nodes, *theme.window, cache_dir=cache_dir)
    shocks = attention_shocks(panel)
    nodes = list(shocks.columns)
    values = shocks.to_numpy(dtype=np.float32)  # [days, n]
    n_days = len(values)

    # temporal split boundaries on the TARGET-END index
    train_end = int(n_days * TRAIN_FRAC)
    val_end = int(n_days * (TRAIN_FRAC + VAL_FRAC))
    bounds = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n_days)}

    X: dict[str, list] = {s: [] for s in bounds}
    y: dict[str, list] = {s: [] for s in bounds}
    mask: dict[str, list] = {s: [] for s in bounds}

    for split, (lo, hi) in bounds.items():
        stride = 1 if split == "train" else HORIZON_DAYS
        t = max(INPUT_DAYS, lo)
        while t + HORIZON_DAYS <= hi:
            window = values[t - INPUT_DAYS : t]  # inputs strictly before t
            future = values[t : t + HORIZON_DAYS]
            n_valid = (~np.isnan(future)).sum(axis=0)
            valid = n_valid >= 4  # need >=4 of the 7 target days
            tgt = np.nansum(future, axis=0) / np.maximum(n_valid, 1)
            X[split].append(np.nan_to_num(window.T, nan=0.0))  # [n, INPUT_DAYS]
            y[split].append(np.where(valid, tgt, 0.0).astype(np.float32))
            mask[split].append(valid)
            t += stride

    return ThemeWindows(
        name=theme.name,
        nodes=nodes,
        adj=build_adjacency(nodes, theme.edge_set()),
        X={s: np.stack(v) if v else np.empty((0, len(nodes), INPUT_DAYS), np.float32) for s, v in X.items()},
        y={s: np.stack(v).astype(np.float32) if v else np.empty((0, len(nodes)), np.float32) for s, v in y.items()},
        mask={s: np.stack(v) if v else np.empty((0, len(nodes)), bool) for s, v in mask.items()},
    )
