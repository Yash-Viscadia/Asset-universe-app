"""
standardize.py — Convert source-specific raw DataFrames into a unified schema.

All three sources produce different field names and value formats.
This layer applies:
  - Column renaming to the canonical schema
  - Light text cleaning (strip whitespace, basic title-casing)
  - Dosage/form suffix stripping from brand/INN names (pre-normalization clean)
  - Stage and region standardization
  - Deduplication of obvious exact duplicates within each source
"""

import re
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# ── Canonical schema ──────────────────────────────────────────────────────────

SCHEMA = [
    "brand_name",       # Cleaned brand / product name
    "inn_name",         # INN / generic / active substance
    "substance_name",   # Raw substance (sometimes more precise than inn)
    "company",          # Sponsor / marketing authorisation holder
    "indication",       # Free-text indication
    "therapeutic_area", # Fixed to "Oncology" for this build
    "stage",            # Marketed / Ph3 / Ph2 / Ph1 / Unknown
    "region",           # US / EU / Global
    "source",           # OpenFDA / ClinicalTrials / EMA
    "source_id",        # NDA / NCT / EPAR product number
    "application_number",
    "nct_id",
    "approval_date",
    "trial_status",
    "route",
    "pharm_class",
]

# ── Regex patterns ────────────────────────────────────────────────────────────

# Dosage and form descriptors to strip from names *before* normalization.
# These produce name variants that should collapse to the same brand root.
_DOSAGE_RE = re.compile(
    r"""
    \s+                          # must be preceded by whitespace
    (?:
        \d+(?:\.\d+)?            # numeric dose: 50, 200.5
        \s*(?:mg|mcg|µg|ug|g|ml|l|iu|miu|units?|%|mmol)   # unit
        (?:/\d+(?:\.\d+)?\s*(?:mg|ml|l))?   # optional second part: /5ml
        .*$                      # rest of string
      |
        \d+/\d+                  # ratio: 25/200
        .*$
      |
        (?:xr|sr|er|ir|cr|dr|la|xl|lp|mr|od|bid|qd|24h|12h)  # release modifiers
        \b.*$
      |
        (?:injection|injectable|solution|suspension|infusion|
           capsule[s]?|tablet[s]?|tab|cap|film.coated|
           concentrate|lyophilised|lyophilized|powder|
           patch|cream|gel|ointment|spray|inhaler|nebuliser|
           auto.?injector|pen|pre.?filled|syringe|vial|
           oral|intravenous|iv|sc|subcutaneous|intramuscular|im|
           plus|pro|forte|duo|fix|combo|kit|pack|hct)
        \b.*$
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_PARENS_RE    = re.compile(r"\s*\([^)]*\)")
_MULTI_SPC_RE = re.compile(r"\s+")

# Noise patterns in company names mapped to canonical forms
_COMPANY_NORMS = [
    (re.compile(r"bristol.myers squibb|bms\b", re.I),      "Bristol-Myers Squibb"),
    (re.compile(r"merck sharp|merck & co|merck kgaa", re.I), "Merck"),
    (re.compile(r"eli lilly", re.I),                         "Eli Lilly"),
    (re.compile(r"f\.?\s*hoffmann|roche registration", re.I),"Roche"),
    (re.compile(r"genentech", re.I),                         "Genentech / Roche"),
    (re.compile(r"novartis pharma", re.I),                   "Novartis"),
    (re.compile(r"astrazeneca", re.I),                       "AstraZeneca"),
    (re.compile(r"pfizer\s+(inc|limited|ltd)", re.I),        "Pfizer"),
    (re.compile(r"johnson\s*&\s*johnson", re.I),             "J&J"),
    (re.compile(r"janssen", re.I),                           "Janssen / J&J"),
    (re.compile(r"\babbvie\b", re.I),                        "AbbVie"),
    (re.compile(r"\bamgen\b", re.I),                         "Amgen"),
    (re.compile(r"sanofi.aventis|sanofi genzyme", re.I),     "Sanofi"),
    (re.compile(r"\bgilead\b", re.I),                        "Gilead"),
    (re.compile(r"\bbiogen\b", re.I),                        "Biogen"),
    (re.compile(r"\bbayer\b", re.I),                         "Bayer"),
    (re.compile(r"\bexelixis\b", re.I),                      "Exelixis"),
    (re.compile(r"\bseagen\b", re.I),                        "Seagen"),
    (re.compile(r"\bbms\b", re.I),                           "Bristol-Myers Squibb"),
]


# ── Name cleaners ─────────────────────────────────────────────────────────────

def clean_drug_name(raw) -> str:
    """
    Strip dosage / form suffixes and normalize whitespace.
    Applies up to 3 passes (handles chained suffixes like '50mg XR Tablets').

    Example: 'KEYTRUDA 200MG/4ML SOLUTION FOR INFUSION' → 'Keytruda'
    Example: 'pembrolizumab'                             → 'Pembrolizumab'
    """
    if not raw or (isinstance(raw, float)):
        return ""
    name = str(raw).strip()
    if name.lower() in {"nan", "none", "", "n/a"}:
        return ""

    name = _PARENS_RE.sub("", name)   # strip (200mg) or (IV)

    for _ in range(3):                # iterative stripping
        stripped = _DOSAGE_RE.sub("", name).strip()
        if stripped == name:
            break
        name = stripped

    name = _MULTI_SPC_RE.sub(" ", name).strip()
    return name.title()


def clean_company(raw) -> str:
    if not raw or (isinstance(raw, float)):
        return ""
    name = str(raw).strip()
    for pattern, replacement in _COMPANY_NORMS:
        if pattern.search(name):
            return replacement
    return name.strip()


# ── Per-source standardizers ──────────────────────────────────────────────────

def standardize_openfda(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SCHEMA)

    out = pd.DataFrame()
    out["brand_name"]        = df["brand_name_raw"].apply(clean_drug_name)
    out["inn_name"]          = df["inn_name_raw"].apply(clean_drug_name)
    out["substance_name"]    = df["substance_raw"].apply(clean_drug_name)
    out["company"]           = df["company_raw"].apply(clean_company)
    out["indication"]        = df["indication_raw"].fillna("").astype(str)
    out["therapeutic_area"]  = "Oncology"
    out["stage"]             = "Marketed"
    out["region"]            = "US"
    out["source"]            = "OpenFDA"
    out["source_id"]         = df.get("application_number", pd.Series("")).fillna("")
    out["application_number"]= df.get("application_number", pd.Series("")).fillna("")
    out["nct_id"]            = ""
    out["approval_date"]     = ""
    out["trial_status"]      = "Marketed"
    out["route"]             = df.get("route_raw", pd.Series("")).fillna("")
    out["pharm_class"]       = df.get("pharm_class_raw", pd.Series("")).fillna("")

    out = out[out["brand_name"].str.len() > 1]
    out = out.drop_duplicates(subset=["brand_name", "application_number"])
    return out.reset_index(drop=True)


def standardize_clinicaltrials(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SCHEMA)

    out = pd.DataFrame()
    out["brand_name"]        = df["brand_name_raw"].apply(clean_drug_name)
    out["inn_name"]          = df["inn_name_raw"].apply(clean_drug_name)
    out["substance_name"]    = df["substance_raw"].apply(clean_drug_name)
    out["company"]           = df["company_raw"].apply(clean_company)
    out["indication"]        = df["indication_raw"].fillna("").astype(str)
    out["therapeutic_area"]  = "Oncology"
    out["stage"]             = df["stage_raw"].fillna("Unknown")
    out["region"]            = "Global"
    out["source"]            = "ClinicalTrials"
    out["source_id"]         = df.get("nct_id", pd.Series("")).fillna("")
    out["nct_id"]            = df.get("nct_id", pd.Series("")).fillna("")
    out["application_number"]= ""
    out["approval_date"]     = ""
    out["trial_status"]      = df.get("trial_status", pd.Series("")).fillna("")
    out["route"]             = ""
    out["pharm_class"]       = ""

    out = out[out["brand_name"].str.len() > 1]
    # Deduplicate: same drug in multiple trials counts once at this stage
    out = out.drop_duplicates(subset=["brand_name", "company", "stage"])
    return out.reset_index(drop=True)


def standardize_ema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SCHEMA)

    out = pd.DataFrame()
    out["brand_name"]        = df["brand_name_raw"].apply(clean_drug_name)
    out["inn_name"]          = df["inn_name_raw"].apply(clean_drug_name)
    out["substance_name"]    = df["substance_raw"].apply(clean_drug_name)
    out["company"]           = df["company_raw"].apply(clean_company)
    out["indication"]        = df.get("indication_raw", pd.Series("")).fillna("")
    out["therapeutic_area"]  = "Oncology"
    out["stage"]             = "Marketed"
    out["region"]            = "EU"
    out["source"]            = "EMA"
    out["source_id"]         = df.get("source_id", pd.Series("")).fillna("")
    out["application_number"]= ""
    out["nct_id"]            = ""
    out["approval_date"]     = df.get("approval_date", pd.Series("")).fillna("")
    out["trial_status"]      = "Approved"
    out["route"]             = ""
    out["pharm_class"]       = ""

    out = out[out["brand_name"].str.len() > 1]
    out = out.drop_duplicates(subset=["brand_name", "source_id"])
    return out.reset_index(drop=True)


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_sources(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate standardized source DataFrames into one working universe.
    Ensures all schema columns are present and fills nulls with empty string.
    """
    valid = [df for df in dfs if not df.empty]
    if not valid:
        return pd.DataFrame(columns=SCHEMA)

    combined = pd.concat(valid, ignore_index=True)
    for col in SCHEMA:
        if col not in combined.columns:
            combined[col] = ""

    combined = combined[SCHEMA].fillna("")
    logger.info(
        f"Merge complete: {len(combined)} records — "
        + " | ".join(
            f"{src}: {(combined['source'] == src).sum()}"
            for src in ["OpenFDA", "ClinicalTrials", "EMA"]
        )
    )
    return combined.reset_index(drop=True)
