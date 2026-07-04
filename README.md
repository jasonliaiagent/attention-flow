# attention-flow

**Where does market attention go next?**

Most financial ML tries to predict prices. This project tries to learn something different: **how human attention propagates through the economy's graph** — from a viral AI product, to the GPUs it runs on, to the fabs that make them, to the data centers that house them, to the power plants that feed them, to the copper in the wires — and how that propagation eventually becomes capital flow.

```
ChatGPT ──► Nvidia ──► TSMC ──► ASML
              │
              ▼
         Data centers ──► Power (nuclear / gas) ──► Grid ──► Copper
```

The hypothesis: attention is not instantaneous. It cascades along real economic links with lags of **days to weeks**, and those lags are measurable — and eventually, forecastable.

## Phase 0 — falsify it first (this repo, today)

Before building any neural network, Phase 0 asks the question the whole project depends on:

> Does attention actually propagate along economic edges, with lags long enough to model?

**Setup.** 34 entities in the 2022–2026 AI-buildout chain (models → chips → fabs → memory → data centers → power → grid → commodities), 39 directed supply-chain edges, and 4 years of daily Wikipedia pageviews (a free, entity-resolved attention proxy — the same family of measure as Da, Engelberg & Gao's *In Search of Attention*, 2011).

**Method.** Raw pageviews are turned into *idiosyncratic attention shocks*: log views minus a trailing 28-day median baseline, then day-of-week demeaned, then the daily cross-sectional mean (the market-wide news-cycle factor) is removed. Without those two corrections, *every* pair of articles correlates ~0.35 and the graph tells you nothing — the first real lesson of the project. For each pair we compute the peak lagged cross-correlation over lags 0–30 days.

**Result.** Attention co-movement **decays monotonically with graph distance** — the dose-response you'd demand before believing propagation is real:

| graph distance (hops) | pairs | mean peak correlation |
|---|---|---|
| 1 (direct edge) | 39 | **0.126** |
| 2 | 144 | 0.101 |
| 3 | 207 | 0.091 |
| 4 | 135 | 0.094 |
| 5 | 36 | 0.081 |

*(adjacent vs distance≥3: one-sided Mann-Whitney p ≈ 0.08 — suggestive at n=39 edges, not yet decisive; growing the graph is Phase 1's job)*

![distance decay](results/distance_decay.png)

The lag structure is where it gets interesting. The *strongest* edges peak at lag 0 — that's just co-coverage (ChatGPT and OpenAI appear in the same articles). But the economically meaningful hops show real, multi-day propagation:

| edge | peak lag |
|---|---|
| Generative AI → Nvidia | 19 days |
| Data center → Vistra | 14 days |
| Data center → Constellation Energy | 21 days |
| HBM → SK hynix | 23 days |
| Nvidia → HBM | 29 days |

![top edges](results/top_edges.png)

Median peak lag across all edges: **7 days**. 69% of edges peak at a lag of ≥1 day. That is the window a model — or a person — could act in.

## Run it yourself

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_phase0.py
```

One script, no API keys, ~3 minutes (Wikimedia pageviews are free; responses are cached in `data/raw/`). Outputs land in `results/`.

## Honest limitations (Phase 0)

- **One theme, one proxy.** A single 34-node graph and Wikipedia-only attention. The gradient is real but underpowered (p ≈ 0.08).
- **Correlation, not causation.** A 19-day lag from "Generative AI" to "Nvidia" is consistent with propagation, and also with slow common drivers. Phase 1 adds directionality tests and event-based identification.
- **Hand-drawn edges.** The graph encodes my priors. The eventual system learns its edges from data (supply-chain filings, co-mentions, patents).
- **Attention ≠ alpha.** Whether any of this survives transaction costs against efficient prices is a Phase 3 question, deliberately deferred.

## Roadmap

- [x] **Phase 0 — falsification test**: does attention decay with graph distance? *(yes, weakly — worth continuing)*
- [ ] **Phase 1 — scale the evidence**: more themes (GLP-1 chain, EV chain, COVID chain), GDELT news mentions + Google Trends as second/third attention proxies, degree-preserving graph permutation tests, transfer-entropy for direction.
- [ ] **Phase 2 — model it**: temporal GNN over the entity graph predicting next-week attention shocks; Hawkes processes for burst timing; contrastive embeddings of "attention episodes" (does robots→GPU→power rhyme with ChatGPT→GPU→power?).
- [ ] **Phase 3 — the capital-flow link**: do graph-predicted attention shocks lead returns/volume beyond momentum baselines (Cohen & Frazzini's *Economic Links and Predictable Returns* is the benchmark to beat)?

## Why this design

The research bet is a different **learning objective**. Instead of `loss = (predicted_return − return)²` — the most efficiently arbitraged target on earth — the system learns `P(attention shock at node j, t+k | graph, shocks ≤ t)`: how information spreads through the economy before it becomes price. Attention trajectories are noisier but far less crowded, and they come with free, abundant, self-supervised training data: the internet's own logs.

## References

- Da, Engelberg & Gao (2011), *In Search of Attention* — search volume as investor attention.
- Cohen & Frazzini (2008), *Economic Links and Predictable Returns* — attention migrates slowly along customer-supplier links.
- Shiller (2017), *Narrative Economics* — narratives as epidemics.

## License

MIT
