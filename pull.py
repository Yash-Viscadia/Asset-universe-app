"""
pull.py — Raw data ingestion from three free public APIs.

Sources:
  - OpenFDA /drug/label  → US marketed oncology drugs (pharm class filter)
  - ClinicalTrials.gov v2 → Industry-sponsored oncology trials, phases 1-3
  - EMA EPAR bulk Excel  → EU approved medicines (TA filter)
"""

import io
import logging
import time
from typing import Callable, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

OPENFDA_LABEL  = "https://api.fda.gov/drug/label.json"
CT_STUDIES     = "https://clinicaltrials.gov/api/v2/studies"
EMA_EPAR_URL   = (
    "https://www.ema.europa.eu/sites/default/files/"
    "Medicines_output_european_public_assessment_reports.xlsx"
)

TIMEOUT = 30          # seconds per request
RETRY_DELAY = 2       # base back-off seconds
API_DELAY   = 0.12    # polite inter-request pause (≈8 req/s)

StatusFn = Optional[Callable[[str], None]]


# ── Generic HTTP helper ───────────────────────────────────────────────────────

def _get(url: str, params: dict = None, retries: int = 3) -> Optional[dict]:
    for attempt in range(retries):
        try:
            time.sleep(API_DELAY)
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                wait = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Rate-limited on {url}. Waiting {wait}s.")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as exc:
            if attempt == retries - 1:
                logger.warning(f"Request failed ({url}): {exc}")
                return None
            time.sleep(RETRY_DELAY * (attempt + 1))
    return None


# ── OpenFDA ───────────────────────────────────────────────────────────────────

def pull_openfda_oncology(
    max_records: int = 600,
    status_fn: StatusFn = None,
) -> pd.DataFrame:
    """
    Pull oncology drug labels from OpenFDA.

    Filter: openfda.pharm_class_epc contains 'Antineoplastic Agent [EPC]'
    Returns one row per unique (brand_name, application_number) pair.
    """
    records = []
    limit   = 100
    skip    = 0
    search  = 'openfda.pharm_class_epc:"Antineoplastic Agent [EPC]"'

    while skip < max_records:
        if status_fn:
            status_fn(f"OpenFDA › fetching records {skip}–{skip + limit}")

        data = _get(OPENFDA_LABEL, {"search": search, "limit": limit, "skip": skip})
        if not data or "results" not in data:
            break

        batch = data["results"]
        if not batch:
            break

        for rec in batch:
            openfda = rec.get("openfda", {})
            brand_names   = openfda.get("brand_name", [])
            generic_names = openfda.get("generic_name", [])
            substances    = openfda.get("substance_name", [])
            manufacturers = openfda.get("manufacturer_name", [])
            app_numbers   = openfda.get("application_number", [])
            routes        = openfda.get("route", [])
            pharm_classes = openfda.get("pharm_class_epc", [])

            indication = _extract_fda_indication(rec)

            for brand in brand_names:
                records.append({
                    "brand_name_raw":   brand,
                    "inn_name_raw":     generic_names[0] if generic_names else "",
                    "substance_raw":    substances[0]    if substances    else "",
                    "company_raw":      manufacturers[0] if manufacturers else "",
                    "application_number": app_numbers[0] if app_numbers  else "",
                    "indication_raw":   indication,
                    "route_raw":        "; ".join(routes),
                    "pharm_class_raw":  "; ".join(pharm_classes),
                    "source": "OpenFDA",
                })

        total_available = data.get("meta", {}).get("results", {}).get("total", 0)
        skip += limit
        if skip >= total_available:
            break

    df = (
        pd.DataFrame(records)
        .drop_duplicates(subset=["brand_name_raw", "application_number"])
        .reset_index(drop=True)
    )
    logger.info(f"OpenFDA pull complete: {len(df)} records")
    return df


def _extract_fda_indication(rec: dict) -> str:
    for field in ["indications_and_usage", "purpose", "clinical_pharmacology"]:
        val = rec.get(field)
        if val:
            text = val[0] if isinstance(val, list) else val
            return str(text)[:400].strip()
    return ""


# ── ClinicalTrials.gov v2 ─────────────────────────────────────────────────────

def pull_clinicaltrials_oncology(
    max_records: int = 1500,
    status_fn: StatusFn = None,
) -> pd.DataFrame:
    """
    Pull oncology pipeline assets from ClinicalTrials.gov API v2.

    Filters:
      - Condition: neoplasms (MeSH)
      - Phase: 1, 2, or 3
      - Sponsor class: INDUSTRY
      - Status: RECRUITING or ACTIVE_NOT_RECRUITING

    Returns one row per drug intervention per study.
    """
    records       = []
    fetched       = 0
    next_token    = None

    base_params = {
        "query.cond":      "neoplasms",
        "filter.advanced": (
            "AREA[Phase](PHASE1 OR PHASE2 OR PHASE3) "
            "AND AREA[LeadSponsorClass]INDUSTRY "
            "AND AREA[OverallStatus](RECRUITING OR ACTIVE_NOT_RECRUITING)"
        ),
        "pageSize":        100,
        "format":          "json",
    }

    while fetched < max_records:
        if status_fn:
            status_fn(f"ClinicalTrials › fetching records {fetched}–{fetched + 100}")

        params = dict(base_params)
        if next_token:
            params["pageToken"] = next_token

        data = _get(CT_STUDIES, params)
        if not data:
            break

        studies = data.get("studies", [])
        if not studies:
            break

        for study in studies:
            proto     = study.get("protocolSection", {})
            id_mod    = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design_mod = proto.get("designModule", {})
            sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
            cond_mod  = proto.get("conditionsModule", {})
            arms_mod  = proto.get("armsInterventionsModule", {})

            nct_id    = id_mod.get("nctId", "")
            phase     = _parse_phase(design_mod.get("phases", []))
            sponsor   = sponsor_mod.get("leadSponsor", {}).get("name", "")
            status    = status_mod.get("overallStatus", "")
            conditions = cond_mod.get("conditions", [])
            indication = "; ".join(conditions[:3])

            interventions = arms_mod.get("interventions", [])
            drug_ivs = [
                iv for iv in interventions
                if iv.get("type", "").upper() in {
                    "DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT", "GENETIC"
                }
            ]

            for iv in drug_ivs:
                name = (iv.get("name") or "").strip()
                if len(name) < 2:
                    continue
                records.append({
                    "brand_name_raw":  name,
                    "inn_name_raw":    name,   # CT uses INN/code most often
                    "substance_raw":   name,
                    "company_raw":     sponsor,
                    "indication_raw":  indication,
                    "stage_raw":       phase,
                    "trial_status":    status,
                    "nct_id":          nct_id,
                    "application_number": "",
                    "route_raw":       "",
                    "pharm_class_raw": "",
                    "source": "ClinicalTrials",
                })

        fetched    += len(studies)
        next_token  = data.get("nextPageToken")
        if not next_token:
            break

    df = pd.DataFrame(records).reset_index(drop=True)
    logger.info(f"ClinicalTrials pull complete: {len(df)} raw intervention records")
    return df


def _parse_phase(phases: list) -> str:
    """Map ClinicalTrials phase codes to canonical stage labels."""
    priority = {"PHASE3": 3, "PHASE2": 2, "PHASE1": 1, "EARLY_PHASE1": 1, "PHASE4": 4}
    labels   = {"PHASE3": "Ph3", "PHASE2": "Ph2", "PHASE1": "Ph1",
                "EARLY_PHASE1": "Ph1", "PHASE4": "Ph4", "NA": "N/A"}
    if not phases:
        return "Unknown"
    best = max(phases, key=lambda p: priority.get(p, 0))
    return labels.get(best, "Unknown")


# ── EMA EPAR ──────────────────────────────────────────────────────────────────

def pull_ema_oncology(status_fn: StatusFn = None) -> pd.DataFrame:
    """
    Pull EU oncology medicines from the EMA EPAR bulk Excel download.

    The file (~5 MB) is downloaded once; rows are filtered for oncology
    via therapeutic area text and ATC-L (antineoplastic) code prefix.
    """
    if status_fn:
        status_fn("EMA EPAR › downloading bulk Excel (~5 MB)…")

    try:
        resp = requests.get(EMA_EPAR_URL, timeout=90, stream=True)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.error(f"EMA download failed: {exc}")
        return pd.DataFrame()

    try:
        # Try common skip-row offsets used by EMA file format
        content = resp.content
        raw = None
        for skip in [8, 0, 1, 4]:
            try:
                candidate = pd.read_excel(io.BytesIO(content), skiprows=skip, dtype=str)
                # Validate: should have at least 5 columns and 10 rows
                if candidate.shape[1] >= 5 and candidate.shape[0] >= 10:
                    raw = candidate
                    break
            except Exception:
                continue

        if raw is None:
            logger.error("Could not parse EMA EPAR Excel file")
            return pd.DataFrame()

        # Normalise column names for robust lookup
        raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]

        # ── Identify key columns ──────────────────────────────────────────────
        col_name    = _find_col(raw, ["medicine_name", "product_name", "name", "medicine"])
        col_inn     = _find_col(raw, ["inn_common_name", "active_substance",
                                      "substance_name", "inn", "active"])
        col_company = _find_col(raw, [
            "marketing_authorisation_holder_company_name",
            "marketing_authorisation_holder",
            "holder", "company", "mah"
        ])
        col_ta      = _find_col(raw, ["therapeutic_area", "category",
                                      "therapeutic_area_s_", "ta"])
        col_atc     = _find_col(raw, ["atc_code", "atc"])
        col_date    = _find_col(raw, [
            "marketing_authorisation_date", "date_of_issue",
            "authorisation_date", "authorisation_date_(procedure_start_date)"
        ])
        col_id      = _find_col(raw, ["product_number", "epar_number", "url",
                                      "epar_url", "product_url"])

        # ── Oncology filter ───────────────────────────────────────────────────
        mask = pd.Series(False, index=raw.index)
        if col_ta:
            mask |= raw[col_ta].astype(str).str.lower().str.contains(
                r"oncol|haematol|cancer|neoplas|tumor|tumour|lymph|leukem",
                na=False, regex=True,
            )
        if col_atc:
            mask |= raw[col_atc].astype(str).str.upper().str.startswith("L", na=False)

        df_onc = raw[mask].copy() if mask.any() else raw.copy()
        logger.info(f"EMA EPAR: {len(df_onc)} oncology rows (of {len(raw)} total)")

        # ── Build output ──────────────────────────────────────────────────────
        def safe(row, col):
            return str(row[col]).strip() if col and col in row.index else ""

        records = [
            {
                "brand_name_raw":     safe(row, col_name),
                "inn_name_raw":       safe(row, col_inn),
                "substance_raw":      safe(row, col_inn),
                "company_raw":        safe(row, col_company),
                "indication_raw":     safe(row, col_ta),
                "approval_date":      safe(row, col_date),
                "source_id":          safe(row, col_id),
                "application_number": "",
                "nct_id":             "",
                "stage_raw":          "Marketed",
                "trial_status":       "Approved",
                "route_raw":          "",
                "pharm_class_raw":    "",
                "source": "EMA",
            }
            for _, row in df_onc.iterrows()
        ]

        df = pd.DataFrame(records)
        df = df[df["brand_name_raw"].str.len() > 1].reset_index(drop=True)
        return df

    except Exception as exc:
        logger.error(f"EMA parse error: {exc}")
        return pd.DataFrame()


def _find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first candidate that matches a column, with substring fallback."""
    for c in candidates:
        if c in df.columns:
            return c
    for c in candidates:
        for col in df.columns:
            if c in col:
                return col
    return None
