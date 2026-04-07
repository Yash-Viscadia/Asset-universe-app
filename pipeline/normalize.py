"""
normalize.py — Brand-level normalization via the NLM RxNorm API.

The core problem this solves:
  "Keytruda 200mg", "KEYTRUDA", "Keytruda injection" and "pembrolizumab"
  should all resolve to a single canonical brand record.

Strategy:
  1. Strip dosage/form from name (already done in standardize.py)
  2. Query RxNorm for a CUI (Concept Unique Identifier)
  3. If CUI exists, determine its Term Type (TTY)
  4. Traverse the RxNorm graph UP to the Brand Name (TTY=BN) root concept
  5. Assign brand_key = root CUI  (the universal dedup key)
  6. For names not found in RxNorm (e.g. trial code names like MK-3475),
     generate a rule-based fuzzy_key as fallback

RxNorm TTY hierarchy relevant to us:
  IN  → Ingredient (e.g. pembrolizumab)
  PIN → Precise ingredient
  BN  → Brand Name (e.g. Keytruda)   ← the root we want
  SBD → Semantic Branded Drug (e.g. Keytruda 200 MG/4 ML Intravenous Solution)
  SBDC→ Semantic Branded Drug Component
  SBP → Semantic Branded Pack

Traversal:
  SBD  --[tradename_of]-→  BN
  SBDC --[tradename_of]-→  BN
  IN   --[has_tradename]-→ BN   (gives us branded children, pick first)
"""

import re
import time
import logging
from functools import lru_cache
from typing import Optional, Tuple, Callable

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
TIMEOUT    = 12
DELAY      = 0.1   # 10 req/s — well within NLM's informal limit

StatusFn = Optional[Callable[[str], None]]

# RxNorm TTY buckets
TTY_BRAND_ROOT    = "BN"
TTY_BRANDED_DRUGS = {"SBD", "SBDC", "SBP", "BPCK"}
TTY_INGREDIENTS   = {"IN", "PIN", "MIN"}


# ── RxNorm API primitives ─────────────────────────────────────────────────────

@lru_cache(maxsize=3000)
def _rxnorm_get(path: str) -> Optional[dict]:
    """Cached GET against the RxNav REST API."""
    try:
        time.sleep(DELAY)
        r = requests.get(f"{RXNAV_BASE}/{path}", timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as exc:
        logger.debug(f"RxNorm request failed ({path}): {exc}")
        return None


@lru_cache(maxsize=3000)
def rxnorm_cui_for_name(name: str) -> Optional[str]:
    """
    Return the best RxNorm CUI for a drug name.
    Tries exact match first; falls back to approximate match.
    """
    if not name or len(name.strip()) < 2:
        return None

    # Exact match (allsrc=0: RxNorm concepts only, not all sources)
    data = _rxnorm_get(f"rxcui.json?name={requests.utils.quote(name)}&allsrc=0")
    if data:
        ids = data.get("idGroup", {}).get("rxnormId", [])
        if ids:
            return ids[0]

    # Approximate match (broader, picks first result)
    data = _rxnorm_get(f"approximateTerm.json?term={requests.utils.quote(name)}&maxEntries=1&option=0")
    if data:
        candidates = data.get("approximateGroup", {}).get("candidate", [])
        if candidates:
            return candidates[0].get("rxcui")

    return None


@lru_cache(maxsize=3000)
def rxnorm_properties(cui: str) -> dict:
    """Return the RxNorm property dict {name, tty, language, ...} for a CUI."""
    data = _rxnorm_get(f"rxcui/{cui}/properties.json")
    if data:
        return data.get("properties", {})
    return {}


@lru_cache(maxsize=3000)
def rxnorm_related(cui: str, rela: str) -> list:
    """
    Return list of related CUIs via a named relationship.
    rela examples: 'tradename_of', 'has_tradename', 'ingredient_of'
    """
    data = _rxnorm_get(f"rxcui/{cui}/related.json?rela={rela}")
    if not data:
        return []
    groups = data.get("relatedGroup", {}).get("conceptGroup", [])
    cuis = []
    for g in groups:
        for c in g.get("conceptProperties", []):
            cui_val = c.get("rxcui")
            if cui_val:
                cuis.append(cui_val)
    return cuis


# ── Brand root traversal ──────────────────────────────────────────────────────

@lru_cache(maxsize=3000)
def brand_name_root(cui: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Traverse the RxNorm graph from any drug CUI up to the BN (Brand Name) root.

    Returns: (brand_root_cui, brand_root_name)
      - brand_root_cui  is the dedup key
      - brand_root_name is the canonical display name

    Traversal rules:
      BN   → already the root; return self
      SBD/SBDC/SBP → follow 'tradename_of' to get to BN
      IN/PIN       → follow 'has_tradename' to get branded form(s),
                     take first; if none, return ingredient CUI/name
      anything else → return self (best effort)
    """
    props = rxnorm_properties(cui)
    tty   = props.get("tty", "")
    name  = props.get("name", "")

    if tty == TTY_BRAND_ROOT:
        return (cui, name)

    if tty in TTY_BRANDED_DRUGS:
        related = rxnorm_related(cui, "tradename_of")
        if related:
            root_cui = related[0]
            root_props = rxnorm_properties(root_cui)
            return (root_cui, root_props.get("name", name))

    if tty in TTY_INGREDIENTS:
        # From ingredient, find the brand via has_tradename
        related = rxnorm_related(cui, "has_tradename")
        if related:
            root_cui = related[0]
            root_props = rxnorm_properties(root_cui)
            if root_props.get("tty") == TTY_BRAND_ROOT:
                return (root_cui, root_props.get("name", name))

    # Fallback: return self as-is
    return (cui, name)


# ── Fuzzy brand key (RxNorm fallback) ────────────────────────────────────────

# Patterns that indicate a trial code name (not a brand name):
#   MK-3475, ABT-199, BGB-3111, MEDI4736, AMG-232
_CODE_NAME_RE = re.compile(
    r"""^
        [A-Z]{1,6}          # 1–6 uppercase letter prefix (sponsor initials)
        [-_]?               # optional separator
        \d{3,6}             # 3–6 digit code
        (?:[-_][A-Z0-9]+)?  # optional suffix
    $""",
    re.VERBOSE,
)

_STRIP_MODIFIERS = re.compile(
    r"\s+(xr|sr|er|ir|cr|dr|la|xl|lp|mr|od|plus|pro|forte|duo|hct|fix)\b.*$",
    re.IGNORECASE,
)
_NON_ALNUM  = re.compile(r"[^a-z0-9\s]")
_MULTI_SPC  = re.compile(r"\s+")


def fuzzy_brand_key(name: str) -> str:
    """
    Generate a stable, normalised string key for use when RxNorm fails.

    Logic:
      - If the name looks like a trial code (MK-3475, ABT-199): keep digits and
        letters, lowercased — codes are already unique identifiers.
      - Otherwise: lowercase, strip release modifiers, remove punctuation,
        collapse whitespace. This collapses common brand variants.

    Examples:
      'Keytruda SR'     → 'keytruda'
      'KEYTRUDA'        → 'keytruda'
      'MK-3475'         → 'mk3475'
      'pembrolizumab'   → 'pembrolizumab'
    """
    if not name:
        return ""

    stripped = name.strip()

    # Detect code names (keep as-is after lowercasing and removing dashes)
    if _CODE_NAME_RE.match(stripped.upper()):
        key = re.sub(r"[-_]", "", stripped.lower())
        return key

    # Brand / INN: strip modifiers then clean
    key = stripped.lower()
    key = _STRIP_MODIFIERS.sub("", key).strip()
    key = _NON_ALNUM.sub(" ", key)
    key = _MULTI_SPC.sub(" ", key).strip()
    return key


# ── Main normalization pass ───────────────────────────────────────────────────

def normalize_brand_names(
    df: pd.DataFrame,
    status_fn: StatusFn = None,
) -> pd.DataFrame:
    """
    Annotate every row in the merged DataFrame with:
      rxnorm_cui        — CUI for the raw brand name (may be None)
      rxnorm_root_cui   — CUI of the BN root (dedup key if RxNorm matched)
      rxnorm_root_name  — Canonical brand name from RxNorm
      rxnorm_tty        — Term type of the raw CUI
      brand_key         — Final dedup key (rxnorm_root_cui OR fuzzy_brand_key)

    We batch-process unique brand names to avoid redundant API calls.
    """
    df = df.copy()

    unique_brands = df["brand_name"].dropna().unique()
    total = len(unique_brands)

    # Maps: brand_name → lookup results
    cui_of:       dict[str, Optional[str]]             = {}
    tty_of:       dict[str, Optional[str]]             = {}
    root_cui_of:  dict[str, Optional[str]]             = {}
    root_name_of: dict[str, Optional[str]]             = {}

    for i, brand in enumerate(unique_brands):
        if status_fn and i % 25 == 0:
            status_fn(
                f"RxNorm › {i}/{total} brand names resolved "
                f"({sum(1 for v in cui_of.values() if v)} matched so far)"
            )

        cui = rxnorm_cui_for_name(brand)
        cui_of[brand] = cui

        if cui:
            props = rxnorm_properties(cui)
            tty_of[brand] = props.get("tty")

            root_cui, root_name = brand_name_root(cui)
            root_cui_of[brand]  = root_cui
            root_name_of[brand] = root_name
        else:
            tty_of[brand]      = None
            root_cui_of[brand]  = None
            root_name_of[brand] = None

    # Apply to DataFrame columns
    df["rxnorm_cui"]       = df["brand_name"].map(cui_of)
    df["rxnorm_tty"]       = df["brand_name"].map(tty_of)
    df["rxnorm_root_cui"]  = df["brand_name"].map(root_cui_of)
    df["rxnorm_root_name"] = df["brand_name"].map(root_name_of)

    # brand_key: prefer RxNorm root CUI; fall back to fuzzy key
    df["brand_key"] = df.apply(
        lambda row: (
            str(row["rxnorm_root_cui"])
            if row["rxnorm_root_cui"]
            else fuzzy_brand_key(row["brand_name"])
        ),
        axis=1,
    )

    matched = df["rxnorm_cui"].notna().sum()
    logger.info(
        f"Brand normalization complete: {matched}/{len(df)} rows matched in RxNorm "
        f"({len(df['brand_key'].unique())} unique brand keys)"
    )

    if status_fn:
        status_fn(
            f"RxNorm complete: {matched}/{len(df)} matched "
            f"→ {df['brand_key'].nunique()} unique brand keys"
        )

    return df
