"""Phase 2 — can a graph neural network forecast next-week attention?

Trains AttentionDiffusionNet (GRU temporal encoder + undirected graph
attention) on all four theme graphs jointly, against three graph-blind
baselines AND its own no-graph ablation (identical architecture, identity
adjacency). Strict walk-forward splits: train 70% | val 10% | test 20% by date.

The question being answered: does knowing the economic graph improve
next-week attention forecasts beyond each entity's own history?

Usage:  python scripts/run_phase2.py
Output: results/phase2_summary.md, phase2_comparison.png, models/*.pt
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention_flow.baselines import RidgeBaseline, predict_persistence, predict_zero
from attention_flow.dataset import build_theme_windows
from attention_flow.graph import THEMES
from attention_flow.model import AttentionDiffusionNet

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODELS = Path(__file__).resolve().parents[1] / "models"
DATA = Path(__file__).resolve().parents[1] / "data" / "raw"

EPOCHS, PATIENCE, BATCH, LR, SEED = 80, 10, 64, 1e-3, 0


def masked_mse(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    return float(((pred - y) ** 2)[mask].mean())


def pooled_ic(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    p, t = pred[mask], y[mask]
    if p.std() < 1e-12 or t.std() < 1e-12:
        return float("nan")  # constant predictor: correlation undefined
    return float(np.corrcoef(p, t)[0, 1])


def fmt(v: float) -> str:
    return "—" if np.isnan(v) else f"{v:+.3f}"


def per_window_ic(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cross-sectional IC per window (the metric a forecaster is paid on)."""
    ics = []
    for b in range(pred.shape[0]):
        m = mask[b]
        if m.sum() < 5 or y[b][m].std() < 1e-8 or pred[b][m].std() < 1e-8:
            continue
        ics.append(np.corrcoef(pred[b][m], y[b][m])[0, 1])
    return np.array(ics)


def train_model(themes: list, use_graph: bool) -> AttentionDiffusionNet:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = AttentionDiffusionNet(use_graph=use_graph)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    adjs = {t.name: torch.from_numpy(t.adj) for t in themes}

    best_val, best_state, stale = np.inf, None, 0
    for epoch in range(EPOCHS):
        model.train()
        for t in themes:
            order = rng.permutation(len(t.X["train"]))
            for i in range(0, len(order), BATCH):
                idx = order[i : i + BATCH]
                x = torch.from_numpy(t.X["train"][idx])
                y = torch.from_numpy(t.y["train"][idx])
                m = torch.from_numpy(t.mask["train"][idx])
                pred = model(x, adjs[t.name])
                loss = ((pred - y) ** 2)[m].mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            val_losses = []
            for t in themes:
                pred = model(torch.from_numpy(t.X["val"]), adjs[t.name]).numpy()
                val_losses.append(masked_mse(pred, t.y["val"], t.mask["val"]))
            val = float(np.mean(val_losses))
        if val < best_val - 1e-5:
            best_val, best_state, stale = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    print(f"  trained use_graph={use_graph}: best val MSE {best_val:.4f} ({epoch + 1} epochs)")
    return model


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    print("Building windows ...")
    themes = [build_theme_windows(t, DATA) for t in THEMES.values()]
    for t in themes:
        print(f"  {t.name}: train/val/test windows = "
              f"{len(t.X['train'])}/{len(t.X['val'])}/{len(t.X['test'])}, nodes = {len(t.nodes)}")

    print("Training ...")
    gnn = train_model(themes, use_graph=True)
    nognn = train_model(themes, use_graph=False)
    torch.save(gnn.state_dict(), MODELS / "attention_gnn.pt")
    torch.save(nognn.state_dict(), MODELS / "attention_nograph.pt")

    w = themes[0].X["train"].shape[2]
    ridge = RidgeBaseline()
    ridge.fit(
        np.concatenate([t.X["train"].reshape(-1, 1, w) for t in themes], axis=0),
        np.concatenate([t.y["train"].reshape(-1, 1) for t in themes], axis=0),
        np.concatenate([t.mask["train"].reshape(-1, 1) for t in themes], axis=0),
    )

    # ---- evaluate on test ----------------------------------------------------
    model_preds: dict[str, list[np.ndarray]] = {k: [] for k in
        ("zero", "persistence", "ridge", "GRU (no graph)", "GRU + graph attention")}
    ys, masks = [], []
    for t in themes:
        x = torch.from_numpy(t.X["test"])
        adj = torch.from_numpy(t.adj)
        with torch.no_grad():
            model_preds["GRU + graph attention"].append(gnn(x, adj).numpy())
            model_preds["GRU (no graph)"].append(nognn(x, adj).numpy())
        model_preds["zero"].append(predict_zero(t.X["test"]))
        model_preds["persistence"].append(predict_persistence(t.X["test"]))
        model_preds["ridge"].append(ridge.predict(t.X["test"]))
        ys.append(t.y["test"])
        masks.append(t.mask["test"])

    rows = []
    window_ics: dict[str, np.ndarray] = {}
    for name, preds in model_preds.items():
        mse = np.mean([masked_mse(p, y, m) for p, y, m in zip(preds, ys, masks)])
        ic = np.nanmean([pooled_ic(p, y, m) for p, y, m in zip(preds, ys, masks)])
        wic = np.concatenate([per_window_ic(p, y, m) for p, y, m in zip(preds, ys, masks)])
        window_ics[name] = wic
        wic_mean = float(wic.mean()) if len(wic) else float("nan")
        rows.append({"model": name, "test_mse": float(mse), "pooled_ic": float(ic),
                     "mean_window_ic": wic_mean, "n_windows": len(wic)})
        print(f"  {name:24s} MSE {mse:.4f}  pooled IC {fmt(ic)}  window IC {fmt(wic_mean)}")

    # paired comparison: does the graph help the SAME architecture?
    a, b = window_ics["GRU + graph attention"], window_ics["GRU (no graph)"]
    from math import comb
    n_pairs = min(len(a), len(b))
    wins = int((a[:n_pairs] > b[:n_pairs]).sum())
    p_graph = sum(comb(n_pairs, k) for k in range(wins, n_pairs + 1)) / 2**n_pairs
    print(f"\n  graph vs no-graph, per-window IC: {wins}/{n_pairs} wins, sign p = {p_graph:.4f}")

    # burst-conditioned: does the graph help when there is something to propagate?
    burst_g, burst_ng, calm_g, calm_ng = [], [], [], []
    for t, pg, png, y, m in zip(
        themes, model_preds["GRU + graph attention"], model_preds["GRU (no graph)"], ys, masks
    ):
        recent = np.abs(t.X["test"][:, :, -7:]).max(axis=(1, 2))  # biggest shock, last 7d
        burst = recent > 2.5
        for cond, acc_g, acc_ng in ((burst, burst_g, burst_ng), (~burst, calm_g, calm_ng)):
            if cond.sum():
                acc_g.append(per_window_ic(pg[cond], y[cond], m[cond]))
                acc_ng.append(per_window_ic(png[cond], y[cond], m[cond]))
    burst_g, burst_ng = np.concatenate(burst_g), np.concatenate(burst_ng)
    calm_g, calm_ng = np.concatenate(calm_g), np.concatenate(calm_ng)
    print(f"  burst windows (|shock|>2.5 in last 7d): graph {burst_g.mean():+.3f} "
          f"vs no-graph {burst_ng.mean():+.3f} (n={len(burst_g)})")
    print(f"  calm windows:                           graph {calm_g.mean():+.3f} "
          f"vs no-graph {calm_ng.mean():+.3f} (n={len(calm_g)})")

    # per-theme IC for the full model
    theme_ic_lines = []
    for t, p, y, m in zip(themes, model_preds["GRU + graph attention"], ys, masks):
        wic = per_window_ic(p, y, m)
        theme_ic_lines.append(f"| {t.name} | {len(wic)} | {wic.mean():+.3f} |")

    # ---- figure --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["model"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = [r["mean_window_ic"] for r in rows]
    bars = ax.bar(names, vals, color=["#999", "#999", "#999", "#7570b3", "#d95f02"])
    ax.set_ylabel("mean cross-sectional IC (test, next-week attention)")
    ax.set_title("Does the economic graph improve attention forecasts?")
    ax.axhline(0, color="#333", lw=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:+.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=12, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "phase2_comparison.png", dpi=150)

    # ---- summary -------------------------------------------------------------
    lines = [
        "# Phase 2 — forecasting next-week attention with a temporal GNN",
        "",
        "Walk-forward: train 70% / val 10% / test 20% by date, per theme. "
        "Target = each node's mean attention shock over the next 7 days.",
        "",
        "| model | test MSE | pooled IC | mean window IC | n windows |",
        "|---|---|---|---|---|",
        *[
            f"| {r['model']} | {r['test_mse']:.4f} | {fmt(r['pooled_ic'])} | "
            f"{fmt(r['mean_window_ic'])} | {r['n_windows']} |"
            for r in rows
        ],
        "",
        f"Graph vs no-graph (identical architecture), per-window IC: "
        f"**{wins}/{n_pairs} wins, sign p = {p_graph:.4f}**",
        "",
        "## Burst-conditioned (is the graph useful when something is propagating?)",
        "",
        "| condition | n windows | graph IC | no-graph IC |",
        "|---|---|---|---|",
        f"| burst (recent shock > 2.5σ) | {len(burst_g)} | {burst_g.mean():+.3f} | {burst_ng.mean():+.3f} |",
        f"| calm | {len(calm_g)} | {calm_g.mean():+.3f} | {calm_ng.mean():+.3f} |",
        "",
        "## Full model, per theme (test window IC)",
        "",
        "| theme | windows | mean IC |",
        "|---|---|---|",
        *theme_ic_lines,
    ]
    (RESULTS / "phase2_summary.md").write_text("\n".join(lines))
    print(f"\nWrote {RESULTS}/phase2_summary.md, phase2_comparison.png, models/*.pt")


if __name__ == "__main__":
    main()
