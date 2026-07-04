"""The economic attention graph for the AI-buildout theme (Phase 0).

Nodes are entities with a Wikipedia article (our attention proxy).
Edges are directed hypotheses: "attention on SRC should later show up on DST",
following supply-chain / dependency links downstream.

Phase 0 deliberately uses ONE theme (the 2023-2026 AI capex chain) because it
is the cleanest recent example of multi-hop attention propagation:
AI models -> GPUs -> fabs/HBM -> data centers -> power -> grid -> commodities.
"""

# node name -> exact English Wikipedia article title
NODES: dict[str, str] = {
    # Layer 0: the spark
    "ChatGPT": "ChatGPT",
    "OpenAI": "OpenAI",
    "Anthropic": "Anthropic",
    "AI": "Generative artificial intelligence",
    "Humanoid robot": "Humanoid robot",
    # Layer 1: compute
    "Nvidia": "Nvidia",
    "AMD": "Advanced Micro Devices",
    # Layer 2: fabs & memory
    "TSMC": "TSMC",
    "ASML": "ASML Holding",
    "SK Hynix": "SK hynix",
    "Micron": "Micron Technology",
    "Samsung": "Samsung Electronics",
    "HBM": "High Bandwidth Memory",
    # Layer 3: cloud & data centers
    "Data center": "Data center",
    "Azure": "Microsoft Azure",
    "AWS": "Amazon Web Services",
    "Google Cloud": "Google Cloud Platform",
    "CoreWeave": "CoreWeave",
    "Oracle": "Oracle Corporation",
    "Vertiv": "Vertiv",
    # Layer 4: power
    "Nuclear power": "Nuclear power",
    "SMR": "Small modular reactor",
    "Natural gas": "Natural gas",
    "Constellation": "Constellation Energy",
    "NextEra": "NextEra Energy",
    "Vistra": "Vistra Corp",
    "GE Vernova": "GE Vernova",
    # Layer 5: grid & commodities
    "Electrical grid": "Electrical grid",
    "Transformer (device)": "Transformer",
    "Quanta Services": "Quanta Services",
    "Eaton": "Eaton Corporation",
    "Copper": "Copper",
    "Uranium": "Uranium",
    # Robotics tail (the "same motif, later" check)
    "Tesla": "Tesla, Inc.",
}

# (src, dst, relation) — attention hypothesized to flow src -> dst
EDGES: list[tuple[str, str, str]] = [
    # spark -> compute
    ("ChatGPT", "OpenAI", "product-of"),
    ("ChatGPT", "Nvidia", "compute-demand"),
    ("OpenAI", "Nvidia", "compute-demand"),
    ("Anthropic", "Nvidia", "compute-demand"),
    ("AI", "Nvidia", "compute-demand"),
    ("AI", "AMD", "compute-demand"),
    ("Humanoid robot", "Nvidia", "compute-demand"),
    ("Tesla", "Humanoid robot", "builds"),
    # compute -> fabs & memory
    ("Nvidia", "TSMC", "foundry"),
    ("AMD", "TSMC", "foundry"),
    ("TSMC", "ASML", "equipment"),
    ("Samsung", "ASML", "equipment"),
    ("Nvidia", "HBM", "component"),
    ("HBM", "SK Hynix", "supplier"),
    ("HBM", "Micron", "supplier"),
    ("HBM", "Samsung", "supplier"),
    # compute / spark -> cloud & data centers
    ("OpenAI", "Azure", "hosted-on"),
    ("ChatGPT", "Azure", "hosted-on"),
    ("Nvidia", "Data center", "deployed-in"),
    ("AI", "Data center", "deployed-in"),
    ("Data center", "CoreWeave", "operator"),
    ("Data center", "Oracle", "operator"),
    ("Data center", "AWS", "operator"),
    ("Data center", "Google Cloud", "operator"),
    ("Data center", "Vertiv", "cooling-power-equipment"),
    # data centers -> power
    ("Data center", "Nuclear power", "power-demand"),
    ("Data center", "Natural gas", "power-demand"),
    ("Data center", "Constellation", "power-purchase"),
    ("Data center", "NextEra", "power-purchase"),
    ("Data center", "Vistra", "power-purchase"),
    ("Data center", "GE Vernova", "turbines"),
    ("Nuclear power", "SMR", "technology"),
    ("Nuclear power", "Constellation", "operator"),
    ("Nuclear power", "Uranium", "fuel"),
    # power -> grid & commodities
    ("Data center", "Electrical grid", "load-growth"),
    ("Electrical grid", "Transformer (device)", "equipment"),
    ("Electrical grid", "Quanta Services", "construction"),
    ("Electrical grid", "Eaton", "equipment"),
    ("Electrical grid", "Copper", "material"),
]


def edge_set() -> set[tuple[str, str]]:
    return {(s, d) for s, d, _ in EDGES}
