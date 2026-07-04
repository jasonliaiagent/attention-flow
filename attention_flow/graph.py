"""Economic attention graphs.

Nodes are entities with a Wikipedia article (our attention proxy).
Edges are directed hypotheses: "attention on SRC should later show up on DST",
following supply-chain / dependency links downstream.

Phase 0 used ONE theme (the 2023-2026 AI capex chain). Phase 1 adds three
independent themes — GLP-1 drugs, EV/battery, and COVID (a different era
entirely) — so the distance-decay test is powered by ~120 edges across
unrelated narratives instead of 39 edges from a single one.
"""

from dataclasses import dataclass, field


@dataclass
class Theme:
    name: str
    window: tuple[str, str]  # (YYYYMMDD, YYYYMMDD)
    nodes: dict[str, str]  # node name -> Wikipedia article title
    edges: list[tuple[str, str, str]] = field(default_factory=list)

    def edge_set(self) -> set[tuple[str, str]]:
        return {(s, d) for s, d, _ in self.edges}

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


# ---------------------------------------------------------------------------
# Phase 1 themes
# ---------------------------------------------------------------------------

GLP1_NODES = {
    "Semaglutide": "Semaglutide",
    "Tirzepatide": "Tirzepatide",
    "GLP-1": "Glucagon-like peptide-1 receptor agonist",
    "Novo Nordisk": "Novo Nordisk",
    "Eli Lilly": "Eli Lilly and Company",
    "Obesity": "Obesity",
    "Weight loss": "Weight loss",
    "Bariatric surgery": "Bariatric surgery",
    "Compounding": "Compounding",
    "Catalent": "Catalent",
    "WeightWatchers": "WW International",
    "PepsiCo": "PepsiCo",
    "Mondelez": "Mondelez International",
    "DaVita": "DaVita",
    "Hims & Hers": "Hims & Hers Health",
}

GLP1_EDGES = [
    ("Semaglutide", "Novo Nordisk", "maker"),
    ("Tirzepatide", "Eli Lilly", "maker"),
    ("Semaglutide", "GLP-1", "drug-class"),
    ("Tirzepatide", "GLP-1", "drug-class"),
    ("Semaglutide", "Weight loss", "indication"),
    ("Semaglutide", "Obesity", "indication"),
    ("GLP-1", "Bariatric surgery", "substitute"),
    ("GLP-1", "WeightWatchers", "disruption"),
    ("GLP-1", "PepsiCo", "demand-risk"),
    ("GLP-1", "Mondelez", "demand-risk"),
    ("GLP-1", "DaVita", "demand-risk"),
    ("Novo Nordisk", "Catalent", "manufacturing"),
    ("Semaglutide", "Compounding", "shortage"),
    ("Compounding", "Hims & Hers", "seller"),
]

EV_NODES = {
    "EV": "Electric vehicle",
    "Tesla": "Tesla, Inc.",
    "BYD": "BYD Company",
    "Rivian": "Rivian",
    "Lucid": "Lucid Motors",
    "Li-ion battery": "Lithium-ion battery",
    "Solid-state battery": "Solid-state battery",
    "CATL": "Contemporary Amperex Technology",
    "Panasonic": "Panasonic",
    "LG Energy Solution": "LG Energy Solution",
    "Lithium": "Lithium",
    "Cobalt": "Cobalt",
    "Nickel": "Nickel",
    "Graphite": "Graphite",
    "Albemarle": "Albemarle Corporation",
    "Charging station": "Charging station",
}

EV_EDGES = [
    ("Tesla", "EV", "category"),
    ("BYD", "EV", "category"),
    ("EV", "Rivian", "maker"),
    ("EV", "Lucid", "maker"),
    ("Tesla", "Li-ion battery", "component"),
    ("BYD", "Li-ion battery", "component"),
    ("EV", "Li-ion battery", "component"),
    ("EV", "Charging station", "infrastructure"),
    ("Li-ion battery", "CATL", "supplier"),
    ("Li-ion battery", "Panasonic", "supplier"),
    ("Li-ion battery", "LG Energy Solution", "supplier"),
    ("Li-ion battery", "Lithium", "material"),
    ("Li-ion battery", "Cobalt", "material"),
    ("Li-ion battery", "Nickel", "material"),
    ("Li-ion battery", "Graphite", "material"),
    ("Li-ion battery", "Solid-state battery", "next-gen"),
    ("Lithium", "Albemarle", "producer"),
]

COVID_NODES = {
    "COVID-19 pandemic": "COVID-19 pandemic",
    "Coronavirus": "Coronavirus",
    "Vaccine": "Vaccine",
    "mRNA vaccine": "MRNA vaccine",
    "Pfizer": "Pfizer",
    "BioNTech": "BioNTech",
    "Moderna": "Moderna",
    "AstraZeneca": "AstraZeneca",
    "Remdesivir": "Remdesivir",
    "Gilead": "Gilead Sciences",
    "Lockdown": "Lockdown",
    "Zoom": "Zoom Video Communications",
    "Peloton": "Peloton Interactive",
    "Cruise ship": "Cruise ship",
    "Carnival": "Carnival Corporation & plc",
    "Airline": "Airline",
    "Boeing": "Boeing",
    "PPE": "Personal protective equipment",
}

COVID_EDGES = [
    ("Coronavirus", "COVID-19 pandemic", "causes"),
    ("COVID-19 pandemic", "Vaccine", "response"),
    ("Vaccine", "mRNA vaccine", "technology"),
    ("mRNA vaccine", "Moderna", "maker"),
    ("mRNA vaccine", "BioNTech", "maker"),
    ("mRNA vaccine", "Pfizer", "maker"),
    ("Vaccine", "AstraZeneca", "maker"),
    ("COVID-19 pandemic", "Remdesivir", "treatment"),
    ("Remdesivir", "Gilead", "maker"),
    ("COVID-19 pandemic", "Lockdown", "response"),
    ("Lockdown", "Zoom", "beneficiary"),
    ("Lockdown", "Peloton", "beneficiary"),
    ("COVID-19 pandemic", "Cruise ship", "victim"),
    ("Cruise ship", "Carnival", "operator"),
    ("COVID-19 pandemic", "Airline", "victim"),
    ("Airline", "Boeing", "supplier"),
    ("COVID-19 pandemic", "PPE", "demand"),
]

THEMES: dict[str, Theme] = {
    "ai-buildout": Theme("ai-buildout", ("20220701", "20260630"), NODES, EDGES),
    "glp1": Theme("glp1", ("20220101", "20260630"), GLP1_NODES, GLP1_EDGES),
    "ev-battery": Theme("ev-battery", ("20200101", "20260630"), EV_NODES, EV_EDGES),
    "covid": Theme("covid", ("20200101", "20221231"), COVID_NODES, COVID_EDGES),
}
