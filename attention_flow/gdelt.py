"""GDELT global news volume — an attention proxy independent of Wikipedia.

The GDELT 2.0 DOC API's timelinevol mode returns, per day, the share of all
global online news coverage matching a query. Free, no key. If the
distance-decay result only existed in Wikipedia pageviews it could be an
artifact of how people browse an encyclopedia; news volume is produced by a
different population (journalists) through a different mechanism.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

API = "https://api.gdeltproject.org/api/v2/doc/doc"
HEADERS = {"User-Agent": "attention-flow/0.1 (open-source research; contact via GitHub)"}

# News queries default to the node's SHORT name (how journalists write it),
# not the Wikipedia article title ("Tesla, Inc." never appears in news text).
# Overrides handle names that are ambiguous as bare news terms.
QUERY_OVERRIDES = {
    # GDELT rejects very short terms — acronym entities need long-form names
    "AMD": '"Advanced Micro Devices"',
    "TSMC": '"Taiwan Semiconductor"',
    "ASML": '"ASML Holding"',
    "AWS": '"Amazon Web Services"',
    "AI": '"generative AI"',
    "HBM": '"high bandwidth memory"',
    "SMR": '"small modular reactor"',
    "Transformer (device)": '"electrical transformer"',
    "Electrical grid": '"power grid"',
    "Azure": '"Microsoft Azure"',
    "Constellation": '"Constellation Energy"',
    "Micron": '"Micron Technology"',
    "Eaton": '"Eaton Corporation"',
    "Compounding": '"compounding pharmacy"',
    "Lockdown": '"covid lockdown"',
    "EV": '"electric vehicle"',
    "Copper": "copper",
    "Nickel": "nickel",
    "Uranium": "uranium",
    "Lithium": "lithium",
    "Graphite": "graphite",
}


def fetch_news_volume(
    query: str,
    start: str,
    end: str,
    cache_dir: Path = Path("data/gdelt"),
) -> dict[str, float] | None:
    """Return {YYYYMMDD: volume%} of global news coverage matching the query."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in query)[:80]
    cache = cache_dir / f"{safe}_{start}_{end}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    params = {
        "query": query,
        "mode": "timelinevol",
        "format": "json",
        "STARTDATETIME": f"{start}000000",
        "ENDDATETIME": f"{end}235959",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for attempt in range(8):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            if resp.status_code == 200 and resp.text.strip().startswith("{"):
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(min(2**attempt, 30))
    else:
        return None

    try:
        timeline = resp.json()["timeline"][0]["data"]
    except (KeyError, IndexError, json.JSONDecodeError):
        return None
    data = {item["date"][:8]: float(item["value"]) for item in timeline}
    cache.write_text(json.dumps(data))
    time.sleep(3.0)  # GDELT rate limit is strict; be polite
    return data


def load_gdelt_panel(
    nodes: dict[str, str],
    start: str,
    end: str,
    cache_dir: Path = Path("data/gdelt"),
) -> pd.DataFrame:
    """Daily news-volume panel for a theme (columns = node names)."""
    series: dict[str, pd.Series] = {}
    missing: list[str] = []
    for name, article in nodes.items():
        query = QUERY_OVERRIDES.get(name, f'"{name}"')
        data = fetch_news_volume(query, start, end, cache_dir=cache_dir)
        if not data:
            missing.append(f"{name} ({query})")
            continue
        s = pd.Series(data, dtype="float64")
        s.index = pd.to_datetime(s.index, format="%Y%m%d")
        series[name] = s
    if missing:
        print(f"WARNING: no GDELT volume for: {', '.join(missing)}")
    return pd.DataFrame(series).sort_index().asfreq("D")
