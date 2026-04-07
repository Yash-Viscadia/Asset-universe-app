"""
orchestrator.py — End-to-end pipeline runner.

Stages in order:
  1. Pull       — fetch raw data from OpenFDA, ClinicalTrials, EMA
  2. Standardize — normalize to common schema
  3. Merge      — combine into one DataFrame
  4. Normalize  — brand-level normalization via RxNorm
  5. Deduplicate — collapse to one row per brand
  6. Enrich     — add ChEMBL MoA / modality / target
  7. Finalize   — add timestamps, sort, clean column order
"""

import logging
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

from .pull       import pull_openfda_oncology, pull_clinicaltrials_oncology, pull_ema_oncology
from .standardize import standardize_openfda, standardize_clinicaltrials, standardize_ema, merge_sources
from .normalize  import normalize_brand_names
from .deduplicate import deduplicate
from .enrich     import enrich_assets

logger = logging.getLogger(__name__)

StatusFn = Optional[Callable[[str], None]]

# Final column order for the output DataFrame
OUTPUT_COLUMNS = [
    "brand_name",
    "inn_name",
    "company",
    "stage",
    "all_stages",
    "region",
    "therapeutic_area",
    "indication",
    "modality",
    "mechanism_of_action",
    "target_name",
    "target_class",
    "atc_code",
    "sources",
    "record_count",
    "rxnorm_cui",
    "chembl_id",
    "nct_ids",
    "application_numbers",
    "approval_date",
    "brand_key",
    "last_refreshed",
]


def run_pipeline(
    status_fn: StatusFn = None,
    openfda_limit: int = 600,
    ct_limit: int = 1500,
) -> pd.DataFrame:
    """
    Execute the full asset universe pipeline.

    Parameters
    ----------
    status_fn : callable, optional
        Called with a status string at each major step — used by the UI
        to update progress messages.
    openfda_limit : int
        Maximum records to fetch from OpenFDA (default 600).
    ct_limit : int
        Maximum records to fetch from ClinicalTrials.gov (default 1500).

    Returns
    -------
    pd.DataFrame
        Deduplicated, enriched oncology asset universe.
    """

    def emit(msg: str):
        logger.info(msg)
        if status_fn:
            status_fn(msg)

    emit("Pipeline started")

    # ── Stage 1: Pull ─────────────────────────────────────────────────────────
    emit("Pulling from OpenFDA…")
    raw_fda = pull_openfda_oncology(max_records=openfda_limit, status_fn=status_fn)
    emit(f"OpenFDA: {len(raw_fda)} raw label records")

    emit("Pulling from ClinicalTrials.gov…")
    raw_ct = pull_clinicaltrials_oncology(max_records=ct_limit, status_fn=status_fn)
    emit(f"ClinicalTrials: {len(raw_ct)} raw intervention records")

    emit("Pulling from EMA EPAR…")
    raw_ema = pull_ema_oncology(status_fn=status_fn)
    emit(f"EMA EPAR: {len(raw_ema)} raw records")

    # ── Stage 2 & 3: Standardize + Merge ─────────────────────────────────────
    emit("Standardizing and merging sources…")
    std_fda = standardize_openfda(raw_fda)
    std_ct  = standardize_clinicaltrials(raw_ct)
    std_ema = standardize_ema(raw_ema)
    merged  = merge_sources(std_fda, std_ct, std_ema)
    emit(
        f"Merged: {len(merged)} records — "
        f"FDA {len(std_fda)} | CT {len(std_ct)} | EMA {len(std_ema)}"
    )

    if merged.empty:
        emit("No records returned from any source — pipeline halted.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # ── Stage 4: Brand Normalization ──────────────────────────────────────────
    emit("Normalizing brand names via RxNorm API…")
    normalized = normalize_brand_names(merged, status_fn=status_fn)

    # ── Stage 5: Deduplicate ──────────────────────────────────────────────────
    emit("Deduplicating to brand level…")
    deduped = deduplicate(normalized)
    emit(f"Deduplication: {len(merged)} records → {len(deduped)} unique brands")

    # ── Stage 6: Enrich ───────────────────────────────────────────────────────
    emit("Enriching with ChEMBL MoA / modality / target data…")
    enriched = enrich_assets(deduped, status_fn=status_fn)

    # ── Stage 7: Finalize ─────────────────────────────────────────────────────
    enriched["therapeutic_area"] = "Oncology"
    enriched["last_refreshed"]   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Ensure all output columns exist
    for col in OUTPUT_COLUMNS:
        if col not in enriched.columns:
            enriched[col] = None

    enriched = enriched[OUTPUT_COLUMNS].fillna("").replace("None", "")

    # Sort: Marketed first, then by stage priority desc, then brand name alpha
    stage_order = {
        "Marketed": 0, "Ph3": 1, "Ph2": 2, "Ph1": 3,
        "Preclinical": 4, "Unknown": 5,
    }
    enriched["_sort_stage"] = enriched["stage"].map(stage_order).fillna(9)
    enriched = enriched.sort_values(["_sort_stage", "brand_name"]).drop(columns=["_sort_stage"])

    emit(f"Pipeline complete ✓ — {len(enriched)} assets in universe")
    return enriched.reset_index(drop=True)
