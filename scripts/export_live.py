"""Export live attention data + GNN forecasts for the demo page.

Fetches the trailing ~240 days of Wikipedia pageviews for every theme,
recomputes attention shocks with the exact Phase-0 pipeline, runs the trained
Phase-2 model for next-week forecasts, and writes docs/data.json.

Run daily by .github/workflows/live.yml; safe to run locally any time.
Fails loudly on fetch errors — CI keeps the previous data.json on failure.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention_flow.analysis import attention_shocks
from attention_flow.dataset import INPUT_DAYS, build_adjacency
from attention_flow.graph import THEMES
from attention_flow.model import AttentionDiffusionNet
from attention_flow.track import summary as track_summary
from attention_flow.track import update as track_update
from attention_flow.wiki import load_panel

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LIVE_CACHE = ROOT / "data" / "live"
WINDOW_DAYS = 240
SPARK_DAYS = 30

THEME_LABELS = {
    "ai-buildout": "AI buildout",
    "glp1": "GLP-1",
    "ev-battery": "EV / battery",
    "covid": "COVID (archive)",
}


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=WINDOW_DAYS - 1)
    start_s, end_s = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    model = AttentionDiffusionNet()
    model.load_state_dict(torch.load(ROOT / "models" / "attention_gnn.pt", weights_only=True))
    model.eval()
    # the no-graph twin: its forecast is the "own-history" component, so
    # (full - base) isolates what the graph itself adds per node
    base_model = AttentionDiffusionNet(use_graph=False)
    base_model.load_state_dict(torch.load(ROOT / "models" / "attention_nograph.pt", weights_only=True))
    base_model.eval()

    themes_out = []
    live_shocks: dict = {}
    data_through = None
    for name, theme in THEMES.items():
        if name == "covid":  # archived era: show its historical final window instead
            t_start, t_end = theme.window
            t_start = (datetime.strptime(t_end, "%Y%m%d") - timedelta(days=WINDOW_DAYS - 1)).strftime("%Y%m%d")
            panel = load_panel(theme.nodes, t_start, t_end, cache_dir=ROOT / "data" / "raw")
        else:
            panel = load_panel(theme.nodes, start_s, end_s, cache_dir=LIVE_CACHE)
        panel = panel.dropna(how="all")
        shocks = attention_shocks(panel)
        if name != "covid":
            live_shocks[name] = shocks
            last = shocks.index[-1]
            data_through = max(data_through, last) if data_through is not None else last

        values = np.nan_to_num(shocks.to_numpy(dtype=np.float32), nan=0.0)
        if len(values) < INPUT_DAYS:
            raise RuntimeError(f"{name}: only {len(values)} days fetched, need {INPUT_DAYS}")
        with torch.no_grad():
            x = torch.from_numpy(values[-INPUT_DAYS:].T[None])
            adj = torch.from_numpy(build_adjacency(list(shocks.columns), theme.edge_set()))
            pred = model(x, adj)[0].numpy()
            pred_base = base_model(x, adj)[0].numpy()

        now = np.nan_to_num(shocks.tail(7).mean().to_numpy(dtype=float), nan=0.0)
        nodes = [
            {
                "id": node,
                "article": theme.nodes[node],
                "now": round(float(now[i]), 3),
                "forecast": round(float(pred[i]), 3),
                "base": round(float(pred_base[i]), 3),
                "spark": [round(float(v), 3) for v in values[-SPARK_DAYS:, i]],
            }
            for i, node in enumerate(shocks.columns)
        ]
        edges = [
            {"source": s, "target": d, "rel": rel}
            for s, d, rel in theme.edges
            if s in shocks.columns and d in shocks.columns
        ]
        themes_out.append(
            {"key": name, "label": THEME_LABELS[name], "live": name != "covid",
             "nodes": nodes, "edges": edges}
        )
        print(f"  {name}: {len(nodes)} nodes, {len(edges)} edges, "
              f"{len(panel)} days through {panel.index[-1].date()}")

    # detected events: attention bursts over the last 5 days, all live themes
    events = []
    for name, shocks in live_shocks.items():
        recent = shocks.tail(5)
        for node in recent.columns:
            s = recent[node].dropna()
            if s.empty:
                continue
            peak_day = s.abs().idxmax()
            z = float(s.loc[peak_day])
            if abs(z) >= 2.0:
                events.append({"theme": name, "node": node,
                               "date": str(peak_day.date()), "z": round(z, 2)})
    events.sort(key=lambda e: -abs(e["z"]))
    events = events[:10]

    track = track_update(DOCS / "track.json", model, live_shocks)
    ts = track_summary(track)
    print(f"  track record: {ts['n_weeks']} graded weeks, mean IC {ts['mean_ic']}, "
          f"{ts['n_live_graded']} graded live")

    out = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_through": str(data_through.date()),
        "track": ts,
        "events": events,
        "themes": themes_out,
    }
    (DOCS / "data.json").write_text(json.dumps(out))
    print(f"Wrote {DOCS / 'data.json'} (data through {out['data_through']})")


if __name__ == "__main__":
    main()
