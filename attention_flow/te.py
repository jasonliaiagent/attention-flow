"""Transfer entropy for direction of attention flow.

Cross-correlation is symmetric in information terms: corr(x_t, y_{t+k}) can be
high because x drives y OR because both follow a common driver with different
inertia. Transfer entropy asks the sharper question: does knowing SRC's past
reduce uncertainty about DST's next step, beyond DST's own past?

    TE(X -> Y) = sum p(y_t+1, y_t, x_t) * log2[ p(y_t+1 | y_t, x_t) / p(y_t+1 | y_t) ]

We use the standard k = l = 1 history, on tercile-discretized attention shocks
(~27 joint states, comfortably estimable from 1,000+ daily samples), and report
the NET direction TE(fwd) - TE(rev) per edge. Because propagation lags run to
weeks, each edge is also tested on 7-day-mean resampled series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def discretize(s: pd.Series, bins: int = 3) -> pd.Series:
    """Quantile-bin a series into integer states (NaNs preserved)."""
    return pd.qcut(s, bins, labels=False, duplicates="drop")


def transfer_entropy(x: pd.Series, y: pd.Series, bins: int = 3) -> float:
    """TE(X -> Y) in bits, k = l = 1, quantile-discretized."""
    df = pd.DataFrame({"x": discretize(x, bins), "y": discretize(y, bins)}).dropna()
    y1 = df["y"].shift(-1)
    joint = pd.DataFrame({"y1": y1, "y0": df["y"], "x0": df["x"]}).dropna().astype(int)
    n = len(joint)
    if n < 100:
        return float("nan")
    p_y1y0x0 = joint.value_counts(normalize=True)
    p_y0x0 = joint[["y0", "x0"]].value_counts(normalize=True)
    p_y1y0 = joint[["y1", "y0"]].value_counts(normalize=True)
    p_y0 = joint["y0"].value_counts(normalize=True)
    te = 0.0
    for (v_y1, v_y0, v_x0), p in p_y1y0x0.items():
        cond_full = p / p_y0x0[(v_y0, v_x0)]
        cond_self = p_y1y0[(v_y1, v_y0)] / p_y0[v_y0]
        te += p * np.log2(cond_full / cond_self)
    return float(te)


def net_te(src: pd.Series, dst: pd.Series, bins: int = 3) -> float:
    """TE(src->dst) - TE(dst->src). Positive = information flows src -> dst."""
    return transfer_entropy(src, dst, bins) - transfer_entropy(dst, src, bins)


def edge_direction_scores(
    shocks: pd.DataFrame,
    edges: list[tuple[str, str, str]],
    weekly: bool = False,
) -> pd.DataFrame:
    """Net TE for every edge; weekly=True tests slow (multi-day) propagation."""
    data = shocks.resample("W").mean() if weekly else shocks
    rows = []
    for src, dst, rel in edges:
        if src not in data or dst not in data:
            continue
        nte = net_te(data[src], data[dst])
        if not np.isnan(nte):
            rows.append({"src": src, "dst": dst, "relation": rel, "net_te": nte})
    return pd.DataFrame(rows)


def sign_test(values: np.ndarray) -> tuple[int, int, float]:
    """One-sided sign test: are positives over-represented? -> (n, positives, p)."""
    from math import comb

    vals = values[~np.isnan(values)]
    n = len(vals)
    wins = int((vals > 0).sum())
    p = sum(comb(n, k) for k in range(wins, n + 1)) / 2**n if n else float("nan")
    return n, wins, p
