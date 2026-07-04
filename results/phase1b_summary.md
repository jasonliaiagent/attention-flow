# Phase 1b — direction (transfer entropy) + proxy replication (GDELT)

## A. Direction: net transfer entropy along hypothesized edges

| theme | freq | edges | net-TE > 0 | sign p |
|---|---|---|---|---|
| ai-buildout | daily | 39 | 19 | 0.625 |
| ai-buildout | weekly | 39 | 15 | 0.946 |
| glp1 | daily | 14 | 10 | 0.090 |
| glp1 | weekly | 14 | 8 | 0.395 |
| ev-battery | daily | 17 | 10 | 0.315 |
| ev-battery | weekly | 17 | 8 | 0.685 |
| covid | daily | 17 | 8 | 0.685 |
| covid | weekly | 17 | 4 | 0.994 |

- POOLED daily: 47/87 positive, sign p = 0.2601
- POOLED weekly: 35/87 positive, sign p = 0.9735

## B. GDELT replication (AI theme)

| distance (hops) | n pairs | GDELT mean peak corr | Wikipedia mean peak corr |
|---|---|---|---|
| 1 | 39 | 0.149 | 0.126 |
| 2 | 144 | 0.214 | 0.101 |
| 3 | 207 | 0.219 | 0.091 |
| 4 | 135 | 0.237 | 0.094 |
| 5 | 36 | 0.301 | 0.081 |

GDELT decay stat (1hop − ≥3hop) = **-0.084**, degree-preserving permutation **p = 0.3227**