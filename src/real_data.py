"""
IntelliCredit - Real-World Data Collector (Review 2 / Review 3 "real data" track)
=================================================================================
Review 1 feedback: "why synthetic data? collect real data."  Borrower-level credit
records in India are confidential (Credit Information Companies (Regulation) Act 2005,
DPDP Act 2023), so no lender or bureau can hand students raw borrower files.  This
module therefore collects every REAL credit-risk dataset that is legally open, from
official statistical APIs, and stores the raw responses + provenance so a reviewer can
re-run the collection.

    Source                          What it gives us                                   Used for
    ------------------------------  -------------------------------------------------  -------------------------------
    1. IMF Financial Soundness      India banks: NPL ratio (% of gross loans) and      Method 1 on REAL India defaults
       Indicators (SDMX API)        provisioning, QUARTERLY 2010-Q1 .. latest          (SARIMAX + conformal intervals)
    2. World Bank WDI API           NPL ratio + macro drivers for ~150 countries,       Panel check that the macro ->
                                    annual 2000 .. latest                               default link we simulate is real
    3. FRED (St. Louis Fed)         US consumer-loan / credit-card delinquency and     Long real default series (1985+)
                                    charge-off rates, quarterly + US macro              for a conformal-coverage backtest
    4. CFPB HMDA Data Browser API   Loan-level mortgage APPLICATIONS with the lender's  Method 2 on REAL loan records
                                    decision (originated / denied) + denial reasons     (association rules -> DENIED),
                                                                                         fairness check
    5. CFPB Consumer Complaint API  Borrower complaints about payday / personal loans  Borrower pain points (issue mix,
                                    (issue, sub-issue, company response)                company responses) for the
                                                                                         recommendations

All five are public, need no API key, and allow reuse (IMF/World Bank: CC BY 4.0;
FRED: public-domain source series; CFPB data: US federal government open data).

Raw responses are cached in data/real/raw/ and every output records its endpoint,
query, row count and fetch time in data/real/real_sources.json.

Usage:
    python -m src.real_data             # fetch everything (uses cache when present)
    python -m src.real_data --refresh   # ignore the cache and re-download
"""

import argparse
import io
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from src.config import DATA_DIR

REAL_DIR = os.path.join(DATA_DIR, "real")
RAW_DIR = os.path.join(REAL_DIR, "raw")
SOURCES_PATH = os.path.join(REAL_DIR, "real_sources.json")
HEADERS = {"User-Agent": "IntelliCredit-capstone/1.0 (academic project)"}

IMF_URL = "https://api.imf.org/external/sdmx/2.1/data/IMF.STA,FSIC/{country}..?startPeriod={start}"
IMF_INDICATORS = {
    "AQ12_CFSI_PT": "npl_ratio_pct",          # Nonperforming loans to total gross loans, deposit takers, %
    "AQ14_CFSI_PT": "provisions_to_npl_pct",  # Provisions to nonperforming loans, deposit takers, %
}

WB_URL = "https://api.worldbank.org/v2/country/all/indicator/{ind}"
WB_INDICATORS = {
    "FB.AST.NPER.ZS": "npl_ratio_pct",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "SL.UEM.TOTL.ZS": "unemployment_rate",
    "FP.CPI.TOTL.ZG": "inflation",
    "FR.INR.LEND": "lending_rate",
    "FS.AST.PRVT.GD.ZS": "private_credit_gdp",
}

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FRED_SERIES = {
    "DRCLACBS": "consumer_loan_delinquency_pct",   # Delinquency rate on consumer loans, all commercial banks (Q, SA)
    "DRCCLACBS": "credit_card_delinquency_pct",    # Delinquency rate on credit card loans (Q, SA)
    "CORCCACBS": "credit_card_chargeoff_pct",      # Charge-off rate on credit card loans (Q, SA)
    "UNRATE": "unemployment_rate",                 # US unemployment rate (M)
    "FEDFUNDS": "fed_funds_rate",                  # Effective federal funds rate (M)
    "CPIAUCSL": "cpi_index",                       # CPI, all urban consumers (M)
}

HMDA_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
HMDA_QUERY = {"states": "DE,RI,VT", "years": "2023", "actions_taken": "1,3", "loan_purposes": "1,32,31"}
HMDA_KEEP = [
    "activity_year", "state_code", "county_code", "derived_loan_product_type", "derived_dwelling_category",
    "derived_ethnicity", "derived_race", "derived_sex", "action_taken", "loan_type", "loan_purpose", "lien_status",
    "loan_amount", "loan_to_value_ratio", "interest_rate", "loan_term", "property_value", "occupancy_type",
    "income", "debt_to_income_ratio", "applicant_credit_score_type", "applicant_age", "submission_of_application",
    "aus-1", "denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4",
    "tract_minority_population_percent", "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage",
]

CFPB_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
CFPB_PRODUCT = "Payday loan, title loan, or personal loan"
CFPB_MAX = 15000   # enough for stable topics; the full product set is ~30k narratives


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get(url, params=None, accept=None, retries=3, timeout=120, library_ua=False):
    # The CFPB endpoints reject custom User-Agent strings, so for them we send the plain
    # python-requests default instead of our project identifier.
    headers = {} if library_ua else dict(HEADERS)
    if accept:
        headers["Accept"] = accept
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:  # retry transient errors
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def _cached(name, fetch, refresh, binary=False):
    """Return raw text/bytes from data/real/raw/<name>, downloading it first if needed."""
    path = os.path.join(RAW_DIR, name)
    if refresh or not os.path.exists(path):
        content = fetch()
        mode = "wb" if binary else "w"
        with open(path, mode, **({} if binary else {"encoding": "utf-8"})) as f:
            f.write(content)
        origin = "live"
    else:
        origin = "cache"
    with open(path, "rb" if binary else "r", **({} if binary else {"encoding": "utf-8"})) as f:
        return f.read(), origin


# --------------------------------------------------------------------------- 1. IMF FSI (India, quarterly)
def collect_imf_india(refresh=False, start="2010"):
    url = IMF_URL.format(country="IND", start=start)
    text, origin = _cached("imf_fsi_india.csv",
                           lambda: _get(url, accept="application/vnd.sdmx.data+csv;version=1.0.0").text, refresh)
    raw = pd.read_csv(io.StringIO(text), low_memory=False)
    q = raw[(raw.FREQUENCY == "Q") & raw.INDICATOR.isin(IMF_INDICATORS)]
    tidy = (q.pivot_table(index="TIME_PERIOD", columns="INDICATOR", values="OBS_VALUE", aggfunc="first")
             .rename(columns=IMF_INDICATORS).sort_index())
    tidy.index = pd.PeriodIndex(tidy.index.str.replace("-", ""), freq="Q").to_timestamp(how="start")
    tidy.index.name = "quarter_start"
    out = os.path.join(REAL_DIR, "india_bank_npl_quarterly.csv")
    tidy.round(4).to_csv(out)
    names = q.drop_duplicates("INDICATOR").set_index("INDICATOR")["SERIES_NAME"].to_dict()
    return {"file": out, "rows": len(tidy), "origin": origin, "endpoint": url,
            "series": names, "first": str(tidy.index.min().date()), "last": str(tidy.index.max().date()),
            "licence": "IMF data, CC BY 4.0 (imf.org/en/About/copyright-and-terms)"}


# --------------------------------------------------------------------------- 2. World Bank panel
def collect_world_bank_panel(refresh=False, years="2000:2025"):
    frames = []
    for ind, col in WB_INDICATORS.items():
        url = WB_URL.format(ind=ind)
        params = {"format": "json", "per_page": 20000, "date": years}
        text, origin = _cached(f"wb_{ind}.json", lambda: _get(url, params=params).text, refresh)
        rows = json.loads(text)[1] or []
        df = pd.DataFrame([{"iso3": r["countryiso3code"], "country": r["country"]["value"],
                            "year": int(r["date"]), col: r["value"]} for r in rows if r["countryiso3code"]])
        frames.append(df.set_index(["iso3", "country", "year"]))
    panel = pd.concat(frames, axis=1).reset_index()
    # drop World Bank aggregates (regions / income groups): they have no NPL lending-rate pair and double count
    meta_text, _ = _cached("wb_countries.json", lambda: _get("https://api.worldbank.org/v2/country",
                                                             params={"format": "json", "per_page": 400}).text, refresh)
    meta = pd.DataFrame(json.loads(meta_text)[1])
    real_countries = set(meta.loc[meta["region"].apply(lambda r: r.get("value") != "Aggregates"), "id"])
    panel = panel[panel.iso3.isin(real_countries)]
    panel["income_group"] = panel.iso3.map(meta.set_index("id")["incomeLevel"].apply(lambda r: r.get("value")))
    out = os.path.join(REAL_DIR, "world_bank_npl_panel.csv")
    panel.sort_values(["iso3", "year"]).to_csv(out, index=False)
    return {"file": out, "rows": len(panel), "countries": int(panel.iso3.nunique()),
            "npl_observations": int(panel.npl_ratio_pct.notna().sum()), "origin": origin,
            "endpoint": WB_URL.format(ind="<indicator>"), "indicators": WB_INDICATORS,
            "licence": "World Bank Open Data, CC BY 4.0"}


# --------------------------------------------------------------------------- 3. FRED (US, long history)
def collect_fred_us(refresh=False):
    series = {}
    origins = set()
    for sid, col in FRED_SERIES.items():
        text, origin = _cached(f"fred_{sid}.csv", lambda: _get(FRED_URL.format(sid=sid)).text, refresh)
        origins.add(origin)
        s = pd.read_csv(io.StringIO(text), parse_dates=[0], index_col=0, na_values=".").iloc[:, 0]
        series[col] = s.resample("QS").mean()   # monthly macro -> quarterly mean; quarterly series unchanged
    df = pd.DataFrame(series)
    df["cpi_inflation_yoy"] = df.pop("cpi_index").pct_change(4, fill_method=None) * 100
    df = df.dropna(subset=["consumer_loan_delinquency_pct"])
    df.index.name = "quarter_start"
    out = os.path.join(REAL_DIR, "us_consumer_credit_quarterly.csv")
    df.round(4).to_csv(out)
    return {"file": out, "rows": len(df), "origin": ",".join(sorted(origins)),
            "endpoint": FRED_URL.format(sid="<series>"), "series": FRED_SERIES,
            "first": str(df.index.min().date()), "last": str(df.index.max().date()),
            "licence": "FRED; Federal Reserve Board charge-off & delinquency release (public domain), BLS"}


# --------------------------------------------------------------------------- 4. HMDA loan-level applications
def collect_hmda(refresh=False):
    raw, origin = _cached("hmda_2023_DE_RI_VT.csv",
                          lambda: _get(HMDA_URL, params=HMDA_QUERY, timeout=300, library_ua=True).content, refresh, binary=True)
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    n_raw = len(df)
    df = df[[c for c in HMDA_KEEP if c in df.columns]]
    out = os.path.join(REAL_DIR, "hmda_applications_2023.csv")
    df.to_csv(out, index=False)
    return {"file": out, "rows_raw": n_raw, "rows": len(df), "columns": len(df.columns), "origin": origin,
            "endpoint": HMDA_URL, "query": HMDA_QUERY,
            "note": "action_taken 1 = loan originated, 3 = application denied; public HMDA data are already "
                    "modified by the CFPB to protect applicant privacy (no names, exact ages or addresses)",
            "licence": "US federal government open data (ffiec.cfpb.gov), public domain"}


# --------------------------------------------------------------------------- 5. CFPB complaint narratives
def _fetch_cfpb(page_size=500):
    # The API ignores `frm`; it pages with `search_after=<score>_<id>` taken from the last hit's sort key.
    hits, after = [], None
    while len(hits) < CFPB_MAX:
        params = {"product": CFPB_PRODUCT, "has_narrative": "true", "size": page_size, "no_aggs": "true"}
        if after:
            params["search_after"] = after
        page = _get(CFPB_URL, params=params, library_ua=True).json()
        batch = page["hits"]["hits"]
        if not batch:
            break
        hits.extend(h["_source"] for h in batch)
        after = "_".join(str(k) for k in batch[-1]["sort"])
        time.sleep(0.5)  # be polite to the public API
    return json.dumps(hits)


def collect_cfpb_complaints(refresh=False):
    text, origin = _cached("cfpb_personal_payday_complaints.json", _fetch_cfpb, refresh)
    rows = json.loads(text)
    df = pd.DataFrame(rows)
    keep = ["complaint_id", "date_received", "product", "sub_product", "issue", "sub_issue", "company",
            "state", "company_response", "timely"]
    df = df[[c for c in keep if c in df.columns]]
    df["date_received"] = pd.to_datetime(df["date_received"]).dt.date
    df = df.drop_duplicates("complaint_id")
    out = os.path.join(REAL_DIR, "cfpb_loan_complaints.csv")
    df.to_csv(out, index=False)
    return {"file": out, "rows": len(df), "origin": origin, "endpoint": CFPB_URL,
            "query": {"product": CFPB_PRODUCT, "has_narrative": True, "max_records": CFPB_MAX},
            "first": str(df.date_received.min()), "last": str(df.date_received.max()),
            "note": "complaints that had a consented narrative; the search API returns the structured fields "
                    "only (product, issue, sub-issue, company response), not the narrative text",
            "licence": "US federal government open data (consumerfinance.gov), public domain"}


COLLECTORS = {
    "imf_india_npl": collect_imf_india,
    "world_bank_panel": collect_world_bank_panel,
    "fred_us_consumer_credit": collect_fred_us,
    "hmda_applications": collect_hmda,
    "cfpb_complaints": collect_cfpb_complaints,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--refresh", action="store_true", help="ignore cached raw files and re-download")
    ap.add_argument("--only", nargs="*", choices=list(COLLECTORS), help="run only these collectors")
    args = ap.parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)
    sources = {}
    if os.path.exists(SOURCES_PATH):
        with open(SOURCES_PATH, encoding="utf-8") as f:
            sources = json.load(f)
    for name, fn in COLLECTORS.items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        try:
            info = fn(refresh=args.refresh)
            info["fetched_at"] = _now()
            info["seconds"] = round(time.time() - t0, 1)
            sources[name] = info
            print(f"[ok]   {name:26s} {info['rows']:>8,} rows  ({info['origin']})  -> {os.path.relpath(info['file'])}")
        except Exception as e:  # keep going: one failing API must not block the others
            print(f"[fail] {name:26s} {type(e).__name__}: {e}")
            sources.setdefault(name, {})["last_error"] = f"{_now()} {type(e).__name__}: {e}"
    for v in sources.values():
        if "file" in v:
            v["file"] = os.path.relpath(v["file"], os.path.dirname(DATA_DIR)).replace("\\", "/")
    with open(SOURCES_PATH, "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2, default=str)
    print(f"provenance -> {os.path.relpath(SOURCES_PATH)}")


if __name__ == "__main__":
    main()
