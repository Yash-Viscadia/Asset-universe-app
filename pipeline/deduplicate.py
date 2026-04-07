"""
deduplicate.py — Collapse the normalized universe to one row per brand.

Grouping logic:
  Primary key : brand_key (= RxNorm root CUI if resolved, else fuzzy_brand_key)
  Secondary   : Within a brand_key group, further split by company where
                the company looks meaningfully different (avoids merging two
                distinct drugs that happen to share a fuzzy-matched key from
                different sponsors).

Canonical record construction (per group):
  brand_name   → RxNorm root name if available; else mode of raw brand names
  inn_name     → mode of non-empty INN values; prefer lowercase/simple forms
  company      → mode of non-empty company values
  stage        → highest-priority stage across all records in the group
  all_stages   → semicolon-joined set of all stages (shows marketed + pipeline)
  region       → sorted union of all regions (US / EU / Global)
  indication   → top-5 deduplicated indication snippets
  sources      → sorted union of all sources
  record_count → how many raw records collapsed into this row
  nct_ids      → all NCT IDs
  application_numbers → all NDA/BLA numbers
  approval_date → earliest non-empty value
  rxnorm_cui   → root CUI
"""

import re
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Higher number = higher priority; Marketed beats Ph3 beats Ph2 etc.
STAGE_PRIORITY = {
    "Marketed": 10,
    "Ph4":      9,
    "Ph3":      8,
    "Ph2":      7,
    "Ph1":      6,
    "Preclinical": 5,
    "N/A":      2,
    "Unknown":  1,
}

# Stages that mean the asset is on the market somewhere
MARKETED_STAGES = {"Marketed"}


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate the brand-normalized DataFrame to one canonical row per brand.

    Groups by brand_key (and optionally company for fuzzy-key records) then
    collapses each group into a single rich row.
    """
    if df.empty:
        return df.copy()

    # For RxNorm-resolved records, group purely on brand_key (the CUI is
    # globally unique). For fuzzy-key records, also include company to avoid
    # merging unrelated drugs from different sponsors.
    df = df.copy()
    df["_group_key"] = df.apply(_make_group_key, axis=1)

    rows = []
    for group_key, group in df.groupby("_group_key", sort=False):
        row = _collapse_group(group)
        if row:
            rows.append(row)

    result = pd.DataFrame(rows).drop(columns=["_group_key"], errors="ignore")

    before = len(df)
    after  = len(result)
    logger.info(
        f"Deduplication: {before} records → {after} unique brands "
        f"(collapsed {before - after} duplicates)"
    )
    return result.reset_index(drop=True)


def _make_group_key(row: pd.Series) -> str:
    """
    Construct the grouping key for a row.

    If the brand_key is a numeric-looking RxNorm CUI (all digits), it's
    globally unique — group on it alone.  Otherwise it's a fuzzy string key;
    append a normalised company name to avoid cross-company merges.
    """
    bk = str(row.get("brand_key", "") or "").strip()
    if not bk:
        return f"__empty__{row.get('brand_name', '')}"

    # RxNorm CUIs are pure integers
    if bk.isdigit():
        return bk

    # Fuzzy key — append normalised company to prevent cross-sponsor collapse
    company_norm = _normalise_company_key(str(row.get("company", "")))
    return f"{bk}|{company_norm}"


def _normalise_company_key(company: str) -> str:
    """Lowercase, drop legal suffixes, keep first meaningful word(s)."""
    c = re.sub(r"\b(inc|ltd|llc|gmbh|plc|sa|ag|bv|co|corp|limited)\b", "", company, flags=re.I)
    c = re.sub(r"[^a-z0-9]", " ", c.lower())
    c = re.sub(r"\s+", " ", c).strip()
    return c[:30]  # cap length


def _collapse_group(group: pd.DataFrame) -> Optional[dict]:
    """Build one canonical record from a group of rows sharing a brand_key."""

    # ── Brand name ────────────────────────────────────────────────────────────
    # Prefer the RxNorm root name (already a clean, authoritative label);
    # fall back to the most frequent raw brand name in the group.
    rxnorm_names = group["rxnorm_root_name"].dropna().astype(str)
    rxnorm_names = rxnorm_names[rxnorm_names.str.len() > 1]
    if not rxnorm_names.empty:
        brand_name = rxnorm_names.mode().iloc[0]
    else:
        brand_name = _mode_nonempty(group["brand_name"])

    if not brand_name:
        return None

    # ── INN ───────────────────────────────────────────────────────────────────
    # Prefer shorter, all-lowercase values (INN convention).
    inn = _best_inn(group["inn_name"])

    # ── Company ───────────────────────────────────────────────────────────────
    company = _mode_nonempty(group["company"])

    # ── Stage ─────────────────────────────────────────────────────────────────
    stages      = [s for s in group["stage"].dropna().unique() if s]
    best_stage  = max(stages, key=lambda s: STAGE_PRIORITY.get(s, 0)) if stages else "Unknown"

    # Sort stages by priority descending for the all_stages display
    sorted_stages = sorted(
        set(stages),
        key=lambda s: -STAGE_PRIORITY.get(s, 0),
    )
    all_stages = "; ".join(sorted_stages)

    # ── Region ────────────────────────────────────────────────────────────────
    regions = sorted({r for r in group["region"].dropna() if r})
    region  = " / ".join(regions)

    # ── Sources ───────────────────────────────────────────────────────────────
    sources = sorted({s for s in group["source"].dropna() if s})

    # ── Indication ────────────────────────────────────────────────────────────
    indication = _aggregate_indications(group["indication"])

    # ── IDs ───────────────────────────────────────────────────────────────────
    nct_ids      = _collect_col(group, "nct_id")
    app_numbers  = _collect_col(group, "application_number")
    approval     = _earliest_date(group.get("approval_date", pd.Series(dtype=str)))

    # ── RxNorm ────────────────────────────────────────────────────────────────
    root_cuis = group["rxnorm_root_cui"].dropna()
    rxnorm_cui = root_cuis.iloc[0] if not root_cuis.empty else None

    brand_key_val = group["brand_key"].iloc[0]

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "brand_name":         brand_name,
        "inn_name":           inn,
        "company":            company,
        "therapeutic_area":   "Oncology",
        # ── Classification ────────────────────────────────────────────────────
        "stage":              best_stage,
        "all_stages":         all_stages,
        "region":             region,
        "indication":         indication,
        # ── Provenance ────────────────────────────────────────────────────────
        "sources":            "; ".join(sources),
        "record_count":       len(group),
        "nct_ids":            nct_ids,
        "application_numbers": app_numbers,
        "approval_date":      approval,
        # ── Normalization metadata ─────────────────────────────────────────────
        "rxnorm_cui":         rxnorm_cui,
        "brand_key":          brand_key_val,
        # ── Enrichment placeholders (filled by enrich.py) ─────────────────────
        "chembl_id":          None,
        "modality":           None,
        "mechanism_of_action": None,
        "target_name":        None,
        "target_class":       None,
        "atc_code":           None,
    }


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _mode_nonempty(series: pd.Series) -> str:
    vals = series.dropna().astype(str)
    vals = vals[vals.str.strip().str.len() > 0]
    if vals.empty:
        return ""
    return vals.mode().iloc[0]


def _best_inn(series: pd.Series) -> str:
    """
    Pick the 'best' INN from a series of candidates.

    Preference order:
      1. All-lowercase strings (classic INN convention, e.g. 'pembrolizumab')
      2. Title-case single-word (e.g. 'Pembrolizumab')
      3. Most frequent non-empty value
    """
    vals = series.dropna().astype(str)
    vals = vals[vals.str.strip().str.len() > 1]
    if vals.empty:
        return ""

    # Exclude values that are clearly brand names (contain spaces with capitals)
    lowercase = vals[vals.str.islower()]
    if not lowercase.empty:
        return lowercase.mode().iloc[0]

    title_single = vals[vals.apply(lambda v: " " not in v and v[0].isupper())]
    if not title_single.empty:
        return title_single.mode().iloc[0]

    return vals.mode().iloc[0]


def _aggregate_indications(series: pd.Series) -> str:
    """
    Collect, split, and deduplicate indication snippets from across the group.

    - Splits on semicolons (our multi-indication delimiter)
    - Drops very short fragments (< 8 chars)
    - Removes near-duplicates by containment check
    - Returns top 5 joined by '; '
    """
    parts = []
    for val in series.dropna():
        val = str(val).strip()
        if not val or val.lower() in {"nan", "none", ""}:
            continue
        for frag in re.split(r"[;|]", val):
            frag = frag.strip()
            if len(frag) >= 8 and frag not in parts:
                parts.append(frag)

    # Remove shorter strings that are contained in a longer one
    deduped = []
    for p in parts:
        p_lower = p.lower()
        if not any(
            p_lower != o.lower() and p_lower in o.lower()
            for o in parts
        ):
            deduped.append(p)

    return "; ".join(deduped[:5])


def _collect_col(group: pd.DataFrame, col: str) -> str:
    if col not in group.columns:
        return ""
    vals = group[col].dropna().astype(str)
    vals = vals[vals.str.strip().str.len() > 0]
    unique = list(dict.fromkeys(vals.tolist()))   # preserve order, deduplicate
    return "; ".join(unique[:10])


def _earliest_date(series: pd.Series) -> str:
    vals = series.dropna().astype(str)
    vals = vals[vals.str.strip().str.len() > 3]
    vals = vals[~vals.str.lower().isin(["nan", "none", "nat"])]
    if vals.empty:
        return ""
    try:
        return str(pd.to_datetime(vals, errors="coerce").dropna().min().date())
    except Exception:
        return vals.iloc[0]
