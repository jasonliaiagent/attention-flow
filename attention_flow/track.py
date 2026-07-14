"""Public track record: the deployed model grades itself weekly.

Every Saturday-anchored week, the trained GNN's next-7-day attention forecast
is logged per theme. Once the week completes, the record is graded: Spearman
IC between forecast and realized mean shock, cross-sectionally over nodes.
The full log lives in docs/track.json (auditable by anyone); the demo page
shows the trailing summary.

Honesty notes:
- Records created for already-completed weeks are labeled mode="backfill".
  They are still out-of-sample (the trailing live window sits inside Phase 2's
  held-out test region), but they were not logged before the fact. Records
  created before their week completes are mode="live" — those were.
- The COVID theme is archived (static 2022 data) and excluded.
- Realized values are recomputed from the current rolling window, so a
  node's normalization can drift slightly between forecast and grading;
  rank-based IC makes this second-order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import INPUT_DAYS, build_adjacency
from .graph import THEMES

HORIZON = 7
MIN_NODES = 8


def _spearman(a: list[float], b: list[float]) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def _forecast(model, shocks: pd.DataFrame, theme, t: int) -> np.ndarray:
    """Model forecast using inputs strictly before day-index t."""
    values = np.nan_to_num(shocks.to_numpy(dtype=np.float32), nan=0.0)
    x = torch.from_numpy(values[t - INPUT_DAYS : t].T[None])
    adj = torch.from_numpy(build_adjacency(list(shocks.columns), theme.edge_set()))
    with torch.no_grad():
        return model(x, adj)[0].numpy()


def update(track_path: Path, model, theme_shocks: dict[str, pd.DataFrame]) -> dict:
    """Create this week's forecast records and grade any completed ones."""
    track = json.loads(track_path.read_text()) if track_path.exists() else {"records": []}
    recs = {r["id"]: r for r in track["records"]}

    for name, shocks in theme_shocks.items():
        theme = THEMES[name]
        dates, n = shocks.index, len(shocks)
        saturdays = [t for t in range(INPUT_DAYS, n) if dates[t].dayofweek == 5]
        for t in saturdays:
            rid = f"{name}|{dates[t].date()}"
            complete = t + HORIZON <= n
            if rid not in recs:
                pred = _forecast(model, shocks, theme, t)
                recs[rid] = {
                    "id": rid,
                    "theme": name,
                    "made": str(dates[t - 1].date()),
                    "target_start": str(dates[t].date()),
                    "target_end": str((dates[t] + pd.Timedelta(days=HORIZON - 1)).date()),
                    "mode": "backfill" if complete else "live",
                    "forecast": {c: round(float(v), 3) for c, v in zip(shocks.columns, pred)},
                    "ic": None,
                    "n": None,
                }
            r = recs[rid]
            if r["ic"] is None and complete:
                realized = shocks.iloc[t : t + HORIZON].mean(axis=0, skipna=True)
                cols = [c for c in r["forecast"] if c in realized.index and pd.notna(realized[c])]
                if len(cols) >= MIN_NODES:
                    r["ic"] = round(_spearman([r["forecast"][c] for c in cols],
                                              [float(realized[c]) for c in cols]), 3)
                    r["n"] = len(cols)

    track["records"] = sorted(recs.values(), key=lambda r: (r["target_start"], r["theme"]))
    track_path.write_text(json.dumps(track))
    return track


def summary(track: dict, trailing: int = 12) -> dict:
    """Trailing weekly ICs (averaged across themes per week) for the page."""
    by_week: dict[str, list[float]] = {}
    n_live = 0
    for r in track["records"]:
        if r["ic"] is not None:
            by_week.setdefault(r["target_end"], []).append(r["ic"])
            n_live += r["mode"] == "live"
    weeks = sorted(by_week)[-trailing:]
    ics = [round(float(np.mean(by_week[w])), 3) for w in weeks]
    return {
        "weeks": weeks,
        "ics": ics,
        "mean_ic": round(float(np.mean(ics)), 3) if ics else None,
        "n_weeks": len(ics),
        "share_positive": round(float(np.mean([i > 0 for i in ics])), 2) if ics else None,
        "n_live_graded": n_live,
    }
