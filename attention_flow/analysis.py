"""Phase-0 statistics: does attention propagate along economic edges?

Method
------
1. Attention *shocks*: following Da, Engelberg & Gao (2011), abnormal attention
   is log(views) minus the median log(views) over the trailing 28 days. This
   strips level differences, slow trends, and most weekly seasonality, leaving
   "how unusual is today's attention for this entity".
2. Lead-lag: for each hypothesized edge SRC -> DST, the cross-correlation
   corr(shock_SRC[t], shock_DST[t+k]) for k = 0..MAX_LAG days. The peak
   correlation and its lag are the edge's propagation estimate.
3. Null model: the same statistic on random ordered node pairs that are NOT
   edges. If economic edges carry attention, edge peak-correlations should
   beat this null distribution.
4. Direction: the same statistic on each edge reversed (DST -> SRC). Genuine
   downstream propagation should be asymmetric.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_LAG = 30
TRAILING_WINDOW = 28


def attention_shocks(panel: pd.DataFrame) -> pd.DataFrame:
    """Abnormal-attention series per node (z-scored for comparability).

    Two confound removals, without which ALL pairs correlate ~0.35:
    - per-series day-of-week demeaning (Wikipedia traffic drops every weekend,
      synchronously for every article);
    - subtraction of the daily cross-sectional mean shock (the market-wide
      news-cycle factor: elections, holidays, outages move everything at once).
    What remains is idiosyncratic abnormal attention — the thing that can
    plausibly *propagate* along edges rather than hit all nodes simultaneously.
    """
    log_views = np.log1p(panel)
    baseline = log_views.rolling(TRAILING_WINDOW, min_periods=14).median().shift(1)
    shocks = log_views - baseline
    shocks = shocks - shocks.groupby(shocks.index.dayofweek).transform("mean")
    shocks = shocks.sub(shocks.mean(axis=1), axis=0)
    return (shocks - shocks.mean()) / shocks.std()


def lagged_corr(a: pd.Series, b: pd.Series, max_lag: int = MAX_LAG) -> tuple[np.ndarray, int, float]:
    """corr(a[t], b[t+k]) for k=0..max_lag. Returns (all corrs, peak lag, peak corr)."""
    corrs = np.array([a.corr(b.shift(-k)) for k in range(max_lag + 1)])
    peak = int(np.nanargmax(corrs))
    return corrs, peak, float(corrs[peak])


@dataclass
class EdgeResult:
    src: str
    dst: str
    relation: str
    peak_lag: int
    peak_corr: float
    corr_at_lag0: float
    reverse_peak_corr: float  # DST -> SRC, the direction check


def score_edges(shocks: pd.DataFrame, edges: list[tuple[str, str, str]]) -> list[EdgeResult]:
    results = []
    for src, dst, rel in edges:
        if src not in shocks or dst not in shocks:
            continue
        corrs, lag, peak = lagged_corr(shocks[src], shocks[dst])
        _, _, rev_peak = lagged_corr(shocks[dst], shocks[src])
        results.append(EdgeResult(src, dst, rel, lag, peak, float(corrs[0]), rev_peak))
    return results


def null_distribution(
    shocks: pd.DataFrame,
    edge_pairs: set[tuple[str, str]],
    n_samples: int = 400,
    seed: int = 42,
) -> np.ndarray:
    """Peak lagged correlation for random ordered NON-edge pairs."""
    rng = random.Random(seed)
    nodes = list(shocks.columns)
    peaks = []
    seen: set[tuple[str, str]] = set()
    while len(peaks) < n_samples:
        a, b = rng.sample(nodes, 2)
        if (a, b) in edge_pairs or (a, b) in seen:
            continue
        seen.add((a, b))
        _, _, peak = lagged_corr(shocks[a], shocks[b])
        peaks.append(peak)
    return np.array(peaks)


def graph_distances(nodes: list[str], edges: set[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """Undirected shortest-path distance between all node pairs (BFS)."""
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    dist: dict[tuple[str, str], int] = {}
    for src in nodes:
        seen = {src: 0}
        frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in seen:
                        seen[v] = seen[u] + 1
                        nxt.append(v)
            frontier = nxt
        for dst, d in seen.items():
            if dst != src:
                dist[(src, dst)] = d
    return dist


def distance_profile(
    shocks: pd.DataFrame, edges: set[tuple[str, str]]
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Peak lagged correlation of every unordered pair, grouped by graph distance.

    If attention propagates along the graph, peak correlation should DECAY
    monotonically with distance — a dose-response curve that a single
    edge-vs-random comparison can't show (in a small supply-chain graph most
    'random' pairs are 2-hop neighbours, i.e. genuinely connected).
    """
    nodes = list(shocks.columns)
    dist = graph_distances(nodes, edges)
    rows = []
    by_distance: dict[int, list[float]] = {}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            d = dist.get((a, b))
            if d is None:
                continue
            _, lag_ab, p_ab = lagged_corr(shocks[a], shocks[b])
            _, lag_ba, p_ba = lagged_corr(shocks[b], shocks[a])
            peak = max(p_ab, p_ba)
            rows.append({"a": a, "b": b, "distance": d, "peak_corr": peak})
            by_distance.setdefault(d, []).append(peak)
    return pd.DataFrame(rows), {d: np.array(v) for d, v in sorted(by_distance.items())}


def mann_whitney_p(x: np.ndarray, y: np.ndarray) -> float:
    """One-sided Mann-Whitney U (x > y), normal approximation."""
    nx, ny = len(x), len(y)
    ranks = pd.Series(np.concatenate([x, y])).rank().to_numpy()
    u = ranks[:nx].sum() - nx * (nx + 1) / 2
    mu, sigma = nx * ny / 2, np.sqrt(nx * ny * (nx + ny + 1) / 12)
    from math import erf

    z = (u - mu) / sigma
    return 1 - (0.5 * (1 + erf(z / np.sqrt(2))))


def summarize(results: list[EdgeResult], null_peaks: np.ndarray) -> dict:
    edge_peaks = np.array([r.peak_corr for r in results])
    # empirical p-value: how often does a random non-edge pair beat the mean edge?
    p_value = float((null_peaks >= edge_peaks.mean()).mean())
    lags = np.array([r.peak_lag for r in results])
    return {
        "n_edges": len(results),
        "edge_mean_peak_corr": float(edge_peaks.mean()),
        "null_mean_peak_corr": float(null_peaks.mean()),
        "empirical_p": p_value,
        "median_peak_lag_days": float(np.median(lags)),
        "share_lag_ge_1d": float((lags >= 1).mean()),
        "share_lag_ge_3d": float((lags >= 3).mean()),
        "share_forward_beats_reverse": float(
            np.mean([r.peak_corr > r.reverse_peak_corr for r in results])
        ),
    }
