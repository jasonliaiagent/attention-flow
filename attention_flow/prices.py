"""Free daily prices via the Yahoo Finance chart API, cached to disk.

Uses adjusted close (splits + dividends) for returns and raw volume for the
abnormal-volume measure. No API key; one request per symbol, cached as JSON.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?range=10y&interval=1d&events=div%2Csplit"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (attention-flow research; contact via GitHub)"}


def fetch_prices(symbol: str, cache_dir: Path = Path("data/prices")) -> pd.DataFrame | None:
    """Daily [Close, Volume] (Close = adjusted) indexed by date, else None."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{symbol}.json"
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        resp = None
        for attempt in range(6):
            try:
                resp = requests.get(URL.format(symbol=symbol), headers=HEADERS, timeout=30)
                if resp.status_code in (200, 404):
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(2**attempt)
        if resp is None or resp.status_code != 200:
            return None
        payload = resp.json()
        cache.write_text(json.dumps(payload))
        time.sleep(0.5)

    try:
        result = payload["chart"]["result"][0]
        ts = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        close = result["indicators"].get("adjclose", [{}])[0].get("adjclose") or quote["close"]
        df = pd.DataFrame(
            {"Close": close, "Volume": quote["volume"]},
            index=pd.to_datetime(ts, unit="s").normalize(),
        )
    except (KeyError, IndexError, TypeError):
        return None
    df = df[~df.index.duplicated(keep="last")].dropna(how="all")
    return df if len(df) else None


def load_price_panel(
    symbols: dict[str, str], cache_dir: Path = Path("data/prices")
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(close, volume) daily panels, columns = node names."""
    closes, volumes, missing = {}, {}, []
    for name, symbol in symbols.items():
        df = fetch_prices(symbol, cache_dir=cache_dir)
        if df is None:
            missing.append(f"{name} ({symbol})")
            continue
        closes[name] = df["Close"]
        volumes[name] = df["Volume"]
    if missing:
        print(f"WARNING: no price data for: {', '.join(missing)}")
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(volumes).sort_index()
