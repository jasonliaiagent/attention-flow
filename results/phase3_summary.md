# Phase 3 — the capital-flow link

Universe: 39 ticker-mapped entities. Weekly cross-sectional Spearman ICs (min 8 names/week); signals known end of week w, outcomes measured week w+1 (GNN rows: forecast made before the outcome week).

| test | mean IC | t | weeks | share + | sign p |
|---|---|---|---|---|---|
| attention -> next-week abnormal volume | +0.059 | +4.82 | 338 | 201/338 | 0.000 |
| attention -> next-week return | +0.018 | +1.36 | 338 | 180/338 | 0.127 |
| attention (reversal-controlled) -> return | +0.010 | +0.78 | 338 | 170/338 | 0.478 |
| momentum (last-week return) -> return | +0.007 | +0.55 | 340 | 174/340 | 0.352 |
| GNN forecast -> same-week abnormal volume | +0.104 | +3.46 | 77 | 50/77 | 0.006 |
| GNN forecast -> same-week return | +0.016 | +0.50 | 77 | 40/77 | 0.410 |

Attention-tercile long-short (gross, equal-weight): **+9.2%/yr, Sharpe 0.30** over 338 weeks.

Notes: gross of costs; small universe; shock normalization uses full-sample stats (rank ICs mitigate). See README for interpretation.