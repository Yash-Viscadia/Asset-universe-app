"""
enrich.py — Enrich deduplicated assets with ChEMBL mechanism/modality data.

ChEMBL (EMBL-EBI) is a freely accessible database of bioactive molecules.
We look up each asset by INN name (preferred) then brand name as fallback.

Fields added:
  chembl_id          — ChEMBL molecule identifier (e.g. CHEMBL3137343)
  modality           — Small Molecule / Monoclonal Antibody / Biologic / etc.
  mechanism_of_action — e.g. "PD-1 inhibitor", "EGFR kinase inhibitor"
  target_name        — e.g. "Programmed cell death protein 1 (PD-1)"
  target_class       — e.g. "Protein", "Nucleic Acid"
  atc_code           — WHO ATC level-5 code (e.g. L01FF02)
"""

import time
import logging
from functools import lru_cache
from typing import Callable, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
TIMEOUT     = 20
DELAY       = 0.18   # ~5.5 req/s

StatusFn = Optional[Callable[[str], None]]

MODALITY_MAP = {
    "Small molecule":  "Small Molecule",
    "Protein":         "Biologic",
    "Antibody":        "Monoclonal Antibody",
    "Oligonucleotide": "Oligonucleotide",
    "Oligosaccharide": "Oligosaccharide",
    "Enzyme":          "Biologic",
    "Cell":            "Cell Therapy",
    "Gene":            "Gene Therapy",
    "Inorganic":       "Inorganic",
    "Unknown":         "Unknown",
}


# ── ChEMBL API helpers ────────────────────────────────────────────────────────

@lru_cache(maxsize=2000)
def _chembl_molecule(name: str) -> Optional[dict]:
    """
    Look up a ChEMBL molecule record by name.
    Tries pref_name (INN) first, then synonym search.
    """
    name_clean = name.strip().lower()
    if not name_clean or len(name_clean) < 2:
        return None

    for field, val in [
        ("pref_name__iexact",              name_clean),
        ("molecule_synonyms__synonym__iexact", name_clean),
    ]:
        try:
            time.sleep(DELAY)
            r = requests.get(
                f"{CHEMBL_BASE}/molecule.json",
                params={field: val, "format": "json", "limit": 1},
                timeout=TIMEOUT,
            )
            if r.status_code != 200:
                continue
            molecules = r.json().get("molecules", [])
            if molecules:
                return molecules[0]
        except Exception as exc:
            logger.debug(f"ChEMBL molecule lookup failed ({name}, {field}): {exc}")

    return None


@lru_cache(maxsize=2000)
def _chembl_mechanism(chembl_id: str) -> Optional[dict]:
    """Return the first mechanism-of-action record for a ChEMBL molecule."""
    try:
        time.sleep(DELAY)
        r = requests.get(
            f"{CHEMBL_BASE}/mechanism.json",
            params={"molecule_chembl_id": chembl_id, "format": "json", "limit": 3},
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            return None
        mechanisms = r.json().get("mechanisms", [])
        # Prefer records with a mechanism_of_action string
        for m in mechanisms:
            if m.get("mechanism_of_action"):
                return m
        return mechanisms[0] if mechanisms else None
    except Exception as exc:
        logger.debug(f"ChEMBL mechanism lookup failed ({chembl_id}): {exc}")
        return None


# ── Enrichment main function ──────────────────────────────────────────────────

def enrich_assets(df: pd.DataFrame, status_fn: StatusFn = None) -> pd.DataFrame:
    """
    Iterate over deduplicated assets and fill in ChEMBL-derived fields.
    We look up by INN first (more stable for ChEMBL), then brand name.
    """
    df = df.copy()
    total = len(df)

    for i, idx in enumerate(df.index):
        if status_fn and i % 15 == 0:
            status_fn(f"ChEMBL enrichment › {i}/{total} assets…")

        inn   = str(df.at[idx, "inn_name"]   or "").strip()
        brand = str(df.at[idx, "brand_name"] or "").strip()

        molecule = None
        for candidate in [inn, brand]:
            if len(candidate) > 1:
                molecule = _chembl_molecule(candidate)
                if molecule:
                    break

        if not molecule:
            continue

        chembl_id = molecule.get("molecule_chembl_id")
        df.at[idx, "chembl_id"] = chembl_id
        df.at[idx, "modality"]  = _parse_modality(molecule)
        df.at[idx, "atc_code"]  = _parse_atc(molecule)

        if chembl_id:
            mech = _chembl_mechanism(chembl_id)
            if mech:
                df.at[idx, "mechanism_of_action"] = mech.get("mechanism_of_action", "")
                df.at[idx, "target_name"]         = mech.get("target_name", "")
                df.at[idx, "target_class"]        = _parse_target_class(mech)

    matched = df["chembl_id"].notna().sum()
    logger.info(f"ChEMBL enrichment complete: {matched}/{total} assets matched")
    if status_fn:
        status_fn(f"ChEMBL enrichment complete: {matched}/{total} assets matched")

    return df


# ── Field parsers ─────────────────────────────────────────────────────────────

def _parse_modality(mol: dict) -> str:
    raw = mol.get("molecule_type", "Unknown") or "Unknown"
    return MODALITY_MAP.get(raw, raw)


def _parse_atc(mol: dict) -> str:
    atc_list = mol.get("atc_classifications", []) or []
    if not atc_list:
        return ""
    # Return the level-5 code (most specific) of the first classification
    return atc_list[0].get("level5", "")


def _parse_target_class(mech: dict) -> str:
    target_type = mech.get("target_type", "") or ""
    class_map = {
        "SINGLE PROTEIN":  "Protein",
        "PROTEIN COMPLEX":  "Protein Complex",
        "PROTEIN FAMILY":   "Protein Family",
        "NUCLEIC-ACID":     "Nucleic Acid",
        "CELL-LINE":        "Cell Line",
        "ORGANISM":         "Organism",
        "TISSUE":           "Tissue",
    }
    return class_map.get(target_type.upper(), target_type)
