"""Phase 3 — the capital-flow link: does entity attention lead its security?

Weekly Fama-MacBeth-style cross-sectional test on ~38 ticker-mapped entities:

    signal (known end of week w)          outcome (week w+1)
    ------------------------------        -------------------------
    realized attention shock, week w  ->  next-week return
                                      ->  next-week abnormal volume
    + reversal control (attention residualized on week-w return)
    + momentum comparison (week-w return as its own signal)
    + the Phase-2 GNN's out-of-sample attention forecast as signal
      (test period only — the model never saw those weeks)

Metrics: weekly cross-sectional Spearman IC; t-stat of the weekly IC series;
tercile long-short next-week return (gross, equal-weight).

Usage:  python scripts/run_phase3.py
Output: results/phase3_summary.md, phase3_capitalflow.png, phase3_weekly_ics.csv
"""

import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention_flow.analysis import attention_shocks
from attention_flow.graph import THEMES
from attention_flow.model import AttentionDiffusionNet
from attention_flow.dataset import build_adjacency, INPUT_DAYS
from attention_flow.prices import load_price_panel
from attention_flow.tickers import TICKER_MAP
from attention_flow.wiki import load_panel

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODELS = Path(__file__).resolve().parents[1] / "models"
DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
PRICES = Path(__file__).resolve().parents[1] / "data" / "prices"

MIN_NAMES = 8  # minimum cross-section size for a weekly IC


def weekly_attention() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """W-FRI mean attention shock per ticker (themes averaged where entities
    repeat), plus each theme's daily shocks for the GNN pass."""
    per_theme_daily: dict[str, pd.DataFrame] = {}
    weekly_frames = []
    for name, theme in THEMES.items():
        shocks = attention_shocks(load_panel(theme.nodes, *theme.window, cache_dir=DATA))
        per_theme_daily[name] = shocks
        mapped = shocks[[n for n in shocks.columns if n in TICKER_MAP]]
        mapped = mapped.rename(columns=TICKER_MAP)
        weekly_frames.append(mapped.resample("W-FRI").mean())
    weekly = pd.concat(weekly_frames, axis=1)
    weekly = weekly.T.groupby(level=0).mean().T  # average duplicate tickers
    return weekly, per_theme_daily


def weekly_market(closes: pd.DataFrame, volumes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Next-week returns and next-week abnormal volume, W-FRI labels."""
    wclose = closes.resample("W-FRI").last()
    wret = wclose.pct_change(fill_method=None)
    wvol = volumes.resample("W-FRI").mean()
    abvol = np.log(wvol / wvol.shift(1).rolling(4).median())
    return wret, abvol


def weekly_ics(signal: pd.DataFrame, outcome: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman IC per week (signal_w vs outcome_w, same label)."""
    ics = {}
    common = signal.index.intersection(outcome.index)
    cols = signal.columns.intersection(outcome.columns)
    for w in common:
        s, o = signal.loc[w, cols], outcome.loc[w, cols]
        m = s.notna() & o.notna()
        if m.sum() >= MIN_NAMES and s[m].std() > 0 and o[m].std() > 0:
            ics[w] = s[m].rank().corr(o[m].rank())  # Spearman, scipy-free
    return pd.Series(ics).dropna()


def ic_stats(ics: pd.Series) -> dict:
    n = len(ics)
    t = ics.mean() / (ics.std() / np.sqrt(n)) if n > 1 else np.nan
    pos = int((ics > 0).sum())
    p_sign = sum(comb(n, k) for k in range(pos, n + 1)) / 2**n if n else np.nan
    return {"mean_ic": ics.mean(), "t": t, "n_weeks": n, "pos": pos, "p_sign": p_sign}


def residualize(signal: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    """Per-week cross-sectional OLS residual of signal on control."""
    out = pd.DataFrame(index=signal.index, columns=signal.columns, dtype=float)
    cols = signal.columns.intersection(control.columns)
    for w in signal.index.intersection(control.index):
        s, c = signal.loc[w, cols], control.loc[w, cols]
        m = s.notna() & c.notna()
        if m.sum() >= MIN_NAMES and c[m].std() > 0:
            b, a = np.polyfit(c[m], s[m], 1)
            out.loc[w, m[m].index] = s[m] - (a + b * c[m])
    return out


def long_short(signal: pd.DataFrame, next_ret: pd.DataFrame) -> pd.Series:
    """Weekly top-tercile minus bottom-tercile next-week return (equal weight)."""
    spread = {}
    cols = signal.columns.intersection(next_ret.columns)
    for w in signal.index.intersection(next_ret.index):
        s, r = signal.loc[w, cols], next_ret.loc[w, cols]
        m = s.notna() & r.notna()
        if m.sum() < 9:
            continue
        ranked = s[m].rank(pct=True)
        spread[w] = r[m][ranked >= 2 / 3].mean() - r[m][ranked <= 1 / 3].mean()
    return pd.Series(spread).dropna()


def gnn_forecasts(per_theme_daily: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Out-of-sample GNN attention forecasts, weekly, test period only.

    For each Saturday t in the final 20% of a theme's window, feed the prior 90
    days and predict the mean shock over [t, t+6] — exactly the W-FRI week
    ending the following Friday. The signal is 'known' the Friday before."""
    model = AttentionDiffusionNet()
    model.load_state_dict(torch.load(MODELS / "attention_gnn.pt", weights_only=True))
    model.eval()
    frames = []
    for name, shocks in per_theme_daily.items():
        theme = THEMES[name]
        nodes = list(shocks.columns)
        adj = torch.from_numpy(build_adjacency(nodes, theme.edge_set()))
        values = np.nan_to_num(shocks.to_numpy(dtype=np.float32), nan=0.0)
        start = max(INPUT_DAYS, int(len(values) * 0.8))
        preds = {}
        for t in range(start, len(values) - 6):
            if shocks.index[t].dayofweek != 5:  # Saturdays only -> clean W-FRI weeks
                continue
            x = torch.from_numpy(values[t - INPUT_DAYS : t].T[None])
            with torch.no_grad():
                p = model(x, adj)[0].numpy()
            preds[shocks.index[t + 6]] = p  # label = the Friday ending that week
        if not preds:
            continue
        df = pd.DataFrame.from_dict(preds, orient="index", columns=nodes)
        df = df[[n for n in nodes if n in TICKER_MAP]].rename(columns=TICKER_MAP)
        frames.append(df)
    pred = pd.concat(frames, axis=1)
    return pred.T.groupby(level=0).mean().T


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print("Building weekly attention signals ...")
    att, per_theme_daily = weekly_attention()
    print(f"  attention: {att.shape[0]} weeks x {att.shape[1]} tickers")

    print("Fetching prices ...")
    closes, volumes = load_price_panel(TICKER_MAP, cache_dir=PRICES)
    closes = closes.rename(columns=TICKER_MAP)  # entity names -> ticker columns
    volumes = volumes.rename(columns=TICKER_MAP)
    wret, abvol = weekly_market(closes, volumes)
    next_ret, next_abvol = wret.shift(-1), abvol.shift(-1)

    att_resid = residualize(att, wret)
    gnn_pred = gnn_forecasts(per_theme_daily)
    print(f"  GNN forecasts: {gnn_pred.shape[0]} test weeks x {gnn_pred.shape[1]} tickers")

    tests = {
        "attention -> next-week abnormal volume": (att, next_abvol),
        "attention -> next-week return": (att, next_ret),
        "attention (reversal-controlled) -> return": (att_resid, next_ret),
        "momentum (last-week return) -> return": (wret.loc[wret.index.isin(att.index)], next_ret),
        "GNN forecast -> same-week abnormal volume": (gnn_pred, abvol),
        "GNN forecast -> same-week return": (gnn_pred, wret),
    }
    rows, ic_series = [], {}
    for label, (sig, out) in tests.items():
        ics = weekly_ics(sig, out)
        ic_series[label] = ics
        s = ic_stats(ics)
        rows.append({"test": label, **s})
        print(f"  {label:44s} IC {s['mean_ic']:+.3f}  t = {s['t']:+.2f}  "
              f"({s['pos']}/{s['n_weeks']} weeks +, sign p = {s['p_sign']:.3f})")

    ls = long_short(att, next_ret)
    ann_ret = ls.mean() * 52
    ann_sharpe = ls.mean() / ls.std() * np.sqrt(52) if ls.std() > 0 else np.nan
    print(f"\n  long-short (attention terciles, gross): {ann_ret:+.1%}/yr, Sharpe {ann_sharpe:.2f}, "
          f"{len(ls)} weeks")

    pd.DataFrame(ic_series).to_csv(RESULTS / "phase3_weekly_ics.csv")

    # ---- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    labels = [r["test"] for r in rows]
    means = [r["mean_ic"] for r in rows]
    errs = [ic_series[l].std() / np.sqrt(len(ic_series[l])) for l in labels]
    colors = ["#1b9e77", "#d95f02", "#d95f02", "#999", "#7570b3", "#7570b3"]
    ax1.barh(labels[::-1], means[::-1], xerr=errs[::-1], color=colors[::-1], capsize=3)
    ax1.axvline(0, color="#333", lw=0.8)
    ax1.set_xlabel("mean weekly cross-sectional Spearman IC (±SE)")
    ax1.set_title("Does attention lead the market?")
    ax1.tick_params(axis="y", labelsize=8)

    ax2.plot(ls.index, ls.cumsum() * 100, color="#d95f02", lw=1.5)
    ax2.axhline(0, color="#333", lw=0.8)
    ax2.set_ylabel("cumulative long-short return (%, gross)")
    ax2.set_title(f"Attention terciles L/S: {ann_ret:+.1%}/yr gross, Sharpe {ann_sharpe:.2f}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS / "phase3_capitalflow.png", dpi=150)

    # ---- summary --------------------------------------------------------------
    lines = [
        "# Phase 3 — the capital-flow link",
        "",
        f"Universe: {att.shape[1]} ticker-mapped entities. Weekly cross-sectional "
        f"Spearman ICs (min {MIN_NAMES} names/week); signals known end of week w, "
        "outcomes measured week w+1 (GNN rows: forecast made before the outcome week).",
        "",
        "| test | mean IC | t | weeks | share + | sign p |",
        "|---|---|---|---|---|---|",
        *[
            f"| {r['test']} | {r['mean_ic']:+.3f} | {r['t']:+.2f} | {r['n_weeks']} | "
            f"{r['pos']}/{r['n_weeks']} | {r['p_sign']:.3f} |"
            for r in rows
        ],
        "",
        f"Attention-tercile long-short (gross, equal-weight): **{ann_ret:+.1%}/yr, "
        f"Sharpe {ann_sharpe:.2f}** over {len(ls)} weeks.",
        "",
        "Notes: gross of costs; small universe; shock normalization uses "
        "full-sample stats (rank ICs mitigate). See README for interpretation.",
    ]
    (RESULTS / "phase3_summary.md").write_text("\n".join(lines))
    print(f"\nWrote {RESULTS}/phase3_summary.md, phase3_capitalflow.png, phase3_weekly_ics.csv")


if __name__ == "__main__":
    main()
