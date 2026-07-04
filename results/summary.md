# Phase 0 — does attention propagate along economic edges?

- Window: 20220701–20260630, 1461 days, 34 entities, 39 edges
- **Edge mean peak correlation: 0.116** vs random-pair null 0.073 (empirical p = 0.1775)
- Median peak lag: **7 days**; 69% of edges peak at ≥1 day, 59% at ≥3 days
- Forward direction beats reverse on 44% of edges

## Dose-response: co-movement vs graph distance

| distance (hops) | n pairs | mean peak corr |
|---|---|---|
| 1 | 39 | 0.126 |
| 2 | 144 | 0.101 |
| 3 | 207 | 0.091 |
| 4 | 135 | 0.094 |
| 5 | 36 | 0.081 |

Adjacent pairs vs distance>=3 pairs: one-sided Mann-Whitney **p = 0.0842**

## Top edges

| src             | dst                  | relation                |   peak_lag |   peak_corr |   corr_at_lag0 |   reverse_peak_corr |
|:----------------|:---------------------|:------------------------|-----------:|------------:|---------------:|--------------------:|
| Nuclear power   | Uranium              | fuel                    |          0 |       0.506 |          0.506 |               0.506 |
| ChatGPT         | OpenAI               | product-of              |          0 |       0.329 |          0.329 |               0.329 |
| Electrical grid | Copper               | material                |          0 |       0.288 |          0.288 |               0.288 |
| TSMC            | ASML                 | equipment               |          0 |       0.287 |          0.287 |               0.287 |
| Electrical grid | Transformer (device) | equipment               |          0 |       0.271 |          0.271 |               0.271 |
| HBM             | Micron               | supplier                |          0 |       0.248 |          0.248 |               0.248 |
| AMD             | TSMC                 | foundry                 |          0 |       0.236 |          0.236 |               0.236 |
| Nvidia          | TSMC                 | foundry                 |          0 |       0.231 |          0.231 |               0.231 |
| Nuclear power   | SMR                  | technology              |          0 |       0.185 |          0.185 |               0.185 |
| Data center     | NextEra              | power-purchase          |          1 |       0.167 |          0.151 |               0.159 |
| AI              | Nvidia               | compute-demand          |         19 |       0.143 |          0.033 |               0.063 |
| Data center     | Vistra               | power-purchase          |         14 |       0.130 |          0.004 |               0.070 |
| Data center     | Vertiv               | cooling-power-equipment |          1 |       0.130 |          0.111 |               0.135 |
| Data center     | Electrical grid      | load-growth             |          0 |       0.118 |          0.118 |               0.118 |
| HBM             | SK Hynix             | supplier                |         23 |       0.114 |         -0.046 |               0.098 |

## Read

- Lag 0 peaks = attention already synchronized (news co-coverage) — not exploitable.
- Lags of days+ with edge >> null = the propagation signal Phase 1 will model.