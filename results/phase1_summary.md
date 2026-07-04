# Phase 1 — the same test, four independent narratives

| theme | days | nodes | edges | decay stat (1hop − ≥3hop) | permutation p | direction (fwd/lagged) | dir p |
|---|---|---|---|---|---|---|---|
| ai-buildout | 1461 | 34 | 39 | 0.035 | 0.003 | 17/27 | 0.124 |
| glp1 | 1642 | 15 | 14 | 0.008 | 0.499 | 5/10 | 0.623 |
| ev-battery | 2373 | 16 | 17 | 0.072 | 0.131 | 2/9 | 0.980 |
| covid | 1096 | 18 | 17 | 0.115 | 0.001 | 4/11 | 0.887 |

## Pooled

| distance (hops) | n pairs | mean peak corr |
|---|---|---|
| 1 | 87 | 0.149 |
| 2 | 280 | 0.114 |
| 3 | 327 | 0.101 |
| 4 | 192 | 0.097 |
| 5 | 53 | 0.078 |

- Pooled adjacent vs distance>=3 (all themes): Mann-Whitney **p = 0.00080** (n = 87 vs 572)
- Fisher-combined per-theme graph-permutation p (1000 degree-preserving rewires each): **p = 0.00015**
- Pooled direction test (lag>=1d edges): 28/57 forward, binomial p = 0.604