# Phase 2 — forecasting next-week attention with a temporal GNN

Walk-forward: train 70% / val 10% / test 20% by date, per theme. Target = each node's mean attention shock over the next 7 days.

| model | test MSE | pooled IC | mean window IC | n windows |
|---|---|---|---|---|
| zero | 0.4418 | — | — | 0 |
| persistence | 0.5251 | +0.404 | +0.376 | 186 |
| ridge | 0.3250 | +0.513 | +0.476 | 186 |
| GRU (no graph) | 0.3242 | +0.530 | +0.490 | 186 |
| GRU + graph attention | 0.3282 | +0.527 | +0.489 | 186 |

Graph vs no-graph (identical architecture), per-window IC: **86/186 wins, sign p = 0.8643**

## Burst-conditioned (is the graph useful when something is propagating?)

| condition | n windows | graph IC | no-graph IC |
|---|---|---|---|
| burst (recent shock > 2.5σ) | 129 | +0.503 | +0.506 |
| calm | 57 | +0.458 | +0.455 |

## Full model, per theme (test window IC)

| theme | windows | mean IC |
|---|---|---|
| ai-buildout | 41 | +0.586 |
| glp1 | 47 | +0.415 |
| ev-battery | 67 | +0.445 |
| covid | 31 | +0.569 |