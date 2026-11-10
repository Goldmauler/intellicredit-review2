"""
IntelliCredit - Live Macroeconomic Data Collector
=================================================
Collects India macroeconomic indicators from two public APIs and builds one
monthly table used by the portfolio simulation, the time-series forecaster and
the underwriter dashboard.

Sources (no API key needed):
    1. FRED (Federal Reserve Bank of St. Louis) - monthly series
       https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>
    2. World Bank Indicators API v2 - annual series
       https://api.worldbank.org/v2/country/IND/indicator/<ID>?format=json

Fallback order:  live API  ->  cached raw response in data/external/raw/  ->  synthetic series.
Every column records where it came from in data/external/macro_sources.json.

Usage:
    python -m src.macro_api            # live fetch (falls back automatically)
    python -m src.macro_api --offline  # rebuild from cached raw files only
"""

import argparse
import io
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from src.config import DATA_DIR

EXTERNAL_DIR = os.path.join(DATA_DIR, "external")
RAW_DIR = os.path.join(EXTERNAL_DIR, "raw")
MACRO_MONTHLY_PATH = os.path.join(EXTERNAL_DIR, "macro_india_monthly.csv")
SOURCES_PATH = os.path.join(EXTERNAL_DIR, "macro_sources.json")

START, END = "2017-01-01", "2025-12-01"

FRED_SERIES = {
    "cpi_inflation_yoy": ("CPALTT01INM659N", "CPI all items, % change vs same month previous year (OECD MEI)"),
    "call_money_rate": ("IRSTCI01INM156N", "Immediate interbank (call money) rate, % p.a."),
    "gsec_10y_yield": ("INDIRLTLT01STM", "10-year government bond yield, % p.a."),
    "inr_per_usd": ("EXINUS", "Indian rupees per US dollar, monthly average"),
}

WORLD_BANK_INDICATORS = {
    "unemployment_rate": ("SL.UEM.TOTL.ZS", "Unemployment, % of labour force (ILO modelled)"),
    "gdp_growth": ("NY.GDP.MKTP.KD.ZG", "Real GDP growth, annual %"),
    "lending_rate": ("FR.INR.LEND", "Commercial bank lending rate, %"),
}

# Public-record event windows (not API data): national lockdown and the second-wave restrictions.
COVID_LOCKDOWN_MONTHS = ["2020-03", "2020-04", "2020-05", "2020-06"]
COVID_SECOND_WAVE_MONTHS = ["2021-04", "2021-05", "2021-06"]

HEADERS = {"User-Agent": "IntelliCredit-capstone/1.0 (academic project)"}


def _get(url, params=None, retries=3, timeout=30):
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as err:
            last_err = err
            time.sleep(1.5 * (attempt + 1))
    raise last_err


def fetch_fred(series_id, offline=False):
    """Return a monthly pd.Series indexed by month start. Raw CSV is cached to RAW_DIR."""
    cache = os.path.join(RAW_DIR, f"fred_{series_id}.csv")
    origin = "cache"
    if not offline:
        try:
            resp = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id})
            if not resp.text.lstrip().lower().startswith("observation_date"):
                raise ValueError(f"FRED returned non-CSV content for {series_id}")
            with open(cache, "w", encoding="utf-8") as f:
                f.write(resp.text)
            origin = "live"
        except Exception as err:  # noqa: BLE001 - any failure means fall back to cache
            print(f"    ! FRED {series_id} live fetch failed ({err}); trying cache")
    if not os.path.exists(cache):
        raise FileNotFoundError(cache)
    df = pd.read_csv(cache)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().set_index("date")["value"]
    return s.resample("MS").mean(), origin


def fetch_world_bank(indicator_id, offline=False):
    """Return an annual pd.Series indexed by year (int). Raw JSON is cached to RAW_DIR."""
    cache = os.path.join(RAW_DIR, f"worldbank_{indicator_id}.json")
    origin = "cache"
    if not offline:
        try:
            resp = _get(
                f"https://api.worldbank.org/v2/country/IND/indicator/{indicator_id}",
                params={"format": "json", "date": "2010:2025", "per_page": 100},
            )
            payload = resp.json()
            if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                raise ValueError(f"World Bank returned no data for {indicator_id}")
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1)
            origin = "live"
        except Exception as err:  # noqa: BLE001
            print(f"    ! World Bank {indicator_id} live fetch failed ({err}); trying cache")
    if not os.path.exists(cache):
        raise FileNotFoundError(cache)
    with open(cache, encoding="utf-8") as f:
        payload = json.load(f)
    rows = [(int(r["date"]), r["value"]) for r in payload[1] if r["value"] is not None]
    return pd.Series(dict(rows)).sort_index(), origin


def annual_to_monthly(annual, months):
    """Place each annual value at mid-year (July) and linearly interpolate between years.
    Months after the last published year are carried forward and flagged by the caller."""
    anchors = pd.Series(annual.values, index=pd.to_datetime([f"{y}-07-01" for y in annual.index]))
    full = anchors.reindex(anchors.index.union(months)).interpolate(method="time")
    return full.reindex(months).ffill().bfill()


def synthetic_macro(months, seed=42):
    """Last-resort fallback with the same columns, so downstream code never breaks offline."""
    rng = np.random.default_rng(seed)
    n = len(months)
    t = np.arange(n)
    shock = np.isin(months.strftime("%Y-%m"), COVID_LOCKDOWN_MONTHS).astype(float)
    return pd.DataFrame({
        "cpi_inflation_yoy": 5 + 1.2 * np.sin(t / 9) + rng.normal(0, 0.3, n),
        "call_money_rate": 5.5 + 0.8 * np.sin(t / 20) - 1.5 * shock + rng.normal(0, 0.1, n),
        "gsec_10y_yield": 7 + 0.5 * np.sin(t / 18) + rng.normal(0, 0.05, n),
        "inr_per_usd": 70 + 0.2 * t + rng.normal(0, 0.4, n),
        "unemployment_rate": 6 + 2.5 * shock + rng.normal(0, 0.2, n),
        "gdp_growth": 6.5 - 12 * shock + rng.normal(0, 0.3, n),
        "lending_rate": 9 + rng.normal(0, 0.1, n),
    }, index=months)


def build_macro_table(offline=False):
    os.makedirs(RAW_DIR, exist_ok=True)
    months = pd.date_range(START, END, freq="MS")
    table = pd.DataFrame(index=months)
    sources = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "columns": {}}
    synthetic = synthetic_macro(months)

    print("[1/2] FRED monthly series")
    for col, (series_id, desc) in FRED_SERIES.items():
        try:
            s, origin = fetch_fred(series_id, offline)
            last_obs = s.index.max()
            table[col] = s.reindex(months).ffill()
            table[f"{col}_is_carried_forward"] = (months > last_obs).astype(int)
            sources["columns"][col] = {
                "api": "FRED", "series_id": series_id, "description": desc, "origin": origin,
                "url": f"https://fred.stlouisfed.org/series/{series_id}",
                "first_obs": str(s.index.min().date()), "last_obs": str(last_obs.date()),
            }
            print(f"    {col:<20} {series_id:<18} {origin:<5} last obs {last_obs:%Y-%m}")
        except Exception as err:  # noqa: BLE001
            table[col] = synthetic[col]
            table[f"{col}_is_carried_forward"] = 0
            sources["columns"][col] = {"api": "synthetic-fallback", "reason": str(err)}
            print(f"    {col:<20} -> synthetic fallback ({err})")

    print("[2/2] World Bank annual indicators")
    for col, (indicator_id, desc) in WORLD_BANK_INDICATORS.items():
        try:
            annual, origin = fetch_world_bank(indicator_id, offline)
            table[col] = annual_to_monthly(annual, months)
            table[f"{col}_is_carried_forward"] = (months > pd.Timestamp(f"{annual.index.max()}-07-01")).astype(int)
            sources["columns"][col] = {
                "api": "World Bank", "indicator_id": indicator_id, "description": desc, "origin": origin,
                "url": f"https://data.worldbank.org/indicator/{indicator_id}?locations=IN",
                "frequency_note": "annual values anchored at July and linearly interpolated to months",
                "years": f"{annual.index.min()}-{annual.index.max()}",
            }
            print(f"    {col:<20} {indicator_id:<18} {origin:<5} years {annual.index.min()}-{annual.index.max()}")
        except Exception as err:  # noqa: BLE001
            table[col] = synthetic[col]
            table[f"{col}_is_carried_forward"] = 0
            sources["columns"][col] = {"api": "synthetic-fallback", "reason": str(err)}
            print(f"    {col:<20} -> synthetic fallback ({err})")

    # Derived indicators
    table["inr_depreciation_yoy"] = table["inr_per_usd"].pct_change(12) * 100
    table["covid_lockdown"] = np.isin(months.strftime("%Y-%m"), COVID_LOCKDOWN_MONTHS).astype(int)
    table["covid_second_wave"] = np.isin(months.strftime("%Y-%m"), COVID_SECOND_WAVE_MONTHS).astype(int)
    sources["columns"]["covid_lockdown"] = {"api": "event-calendar", "months": COVID_LOCKDOWN_MONTHS}
    sources["columns"]["covid_second_wave"] = {"api": "event-calendar", "months": COVID_SECOND_WAVE_MONTHS}

    table.index.name = "date"
    table = table.reset_index()
    table.to_csv(MACRO_MONTHLY_PATH, index=False)
    with open(SOURCES_PATH, "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2)
    return table, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="use cached raw API files only")
    args = parser.parse_args()

    print("=" * 70)
    print("  IntelliCredit - Macroeconomic API collection (FRED + World Bank)")
    print("=" * 70)
    table, sources = build_macro_table(offline=args.offline)
    live = sum(1 for c in sources["columns"].values() if c.get("origin") == "live")
    print(f"\nSaved {MACRO_MONTHLY_PATH} ({len(table)} months x {table.shape[1]} cols); {live} series fetched live")
    print(f"Provenance written to {SOURCES_PATH}")


if __name__ == "__main__":
    main()
