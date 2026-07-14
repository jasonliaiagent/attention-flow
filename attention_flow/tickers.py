"""Entity -> listed-security mapping for the capital-flow test.

Only nodes with a liquid, cleanly-attributable listing are mapped. Concepts
(Data center, Lithium, Nuclear power...) and private companies (OpenAI,
Anthropic, CATL's A-shares...) are excluded by design — the test is about
whether ENTITY attention leads that entity's OWN security.

Product/subsidiary nodes map to the parent (Azure -> MSFT): entity attention
on a product is attention on the parent's business.

Symbols are Yahoo Finance format (US listings / ADRs).
"""

# node name -> Yahoo symbol
TICKER_MAP: dict[str, str] = {
    # AI buildout
    "Nvidia": "NVDA",
    "AMD": "AMD",
    "TSMC": "TSM",
    "ASML": "ASML",
    "Micron": "MU",
    "Oracle": "ORCL",
    "CoreWeave": "CRWV",  # listed Mar 2025 — short history, handled downstream
    "Vertiv": "VRT",
    "Constellation": "CEG",
    "NextEra": "NEE",
    "Vistra": "VST",
    "GE Vernova": "GEV",  # spun off Apr 2024
    "Quanta Services": "PWR",
    "Eaton": "ETN",
    "Azure": "MSFT",
    "AWS": "AMZN",
    "Google Cloud": "GOOGL",
    # GLP-1
    "Novo Nordisk": "NVO",
    "Eli Lilly": "LLY",
    "WeightWatchers": "WW",  # post-2025 relisting only; pre-bankruptcy history unavailable
    "PepsiCo": "PEP",
    "Mondelez": "MDLZ",
    "DaVita": "DVA",
    "Hims & Hers": "HIMS",
    # EV / battery
    "Tesla": "TSLA",
    "BYD": "BYDDY",
    "Rivian": "RIVN",
    "Lucid": "LCID",
    "Panasonic": "PCRFF",
    "Albemarle": "ALB",
    # COVID
    "Pfizer": "PFE",
    "BioNTech": "BNTX",
    "Moderna": "MRNA",
    "AstraZeneca": "AZN",
    "Gilead": "GILD",
    "Zoom": "ZM",
    "Peloton": "PTON",
    "Carnival": "CCL",
    "Boeing": "BA",
}
