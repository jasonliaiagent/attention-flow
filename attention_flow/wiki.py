"""Wikipedia pageviews as a free, entity-resolved attention proxy.

Daily per-article view counts from the Wikimedia REST API (data since 2015-07).
Responses are cached to disk so the full pipeline is re-runnable offline.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "{project}/all-access/user/{article}/daily/{start}/{end}"
)
# Wikimedia requires a descriptive User-Agent; the default python UA is blocked.
HEADERS = {"User-Agent": "attention-flow/0.1 (open-source research; contact via GitHub)"}


def fetch_pageviews(
    article: str,
    start: str,
    end: str,
    project: str = "en.wikipedia",
    cache_dir: Path = Path("data/raw"),
) -> dict[str, int] | None:
    """Return {YYYYMMDD: views} for one article, or None if the article 404s."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{article.replace('/', '_').replace(' ', '_')}_{start}_{end}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    quoted = urllib.parse.quote(article.replace(" ", "_"), safe="")
    url = API.format(project=project, article=quoted, start=start, end=end)
    resp = None
    for attempt in range(6):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            if resp.status_code != 429:
                break
        except requests.exceptions.RequestException:
            if attempt == 5:
                raise
        time.sleep(2**attempt)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = {item["timestamp"][:8]: item["views"] for item in resp.json()["items"]}
    cache.write_text(json.dumps(data))
    time.sleep(1.0)  # stay well under the API's rate limit
    return data


def load_panel(
    nodes: dict[str, str],
    start: str,
    end: str,
    cache_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
    """Fetch all nodes into one daily DataFrame (columns = node names)."""
    series: dict[str, pd.Series] = {}
    missing: list[str] = []
    for name, article in nodes.items():
        data = fetch_pageviews(article, start, end, cache_dir=cache_dir)
        if data is None:
            missing.append(f"{name} ({article})")
            continue
        s = pd.Series(data, dtype="float64")
        s.index = pd.to_datetime(s.index, format="%Y%m%d")
        series[name] = s
    if missing:
        print(f"WARNING: no Wikipedia article found for: {', '.join(missing)}")
    panel = pd.DataFrame(series).sort_index()
    return panel.asfreq("D")
