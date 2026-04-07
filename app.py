"""
app.py — Oncology Asset Universe · Streamlit UI

Layout
------
Sidebar : pipeline controls + filters + download
Main    : KPI bar → filterable asset table → brand-key debug expander
"""

import io
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from orchestrator import run_pipeline

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oncology Asset Universe",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "universe":    None,
    "last_run":    None,
    "run_log":     [],
    "running":     False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

CACHE_PATH = "/tmp/oncology_universe.parquet"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cached() -> bool:
    """Load universe from /tmp parquet if it exists (persists within container)."""
    if os.path.exists(CACHE_PATH):
        try:
            st.session_state.universe = pd.read_parquet(CACHE_PATH)
            mtime = os.path.getmtime(CACHE_PATH)
            st.session_state.last_run = datetime.fromtimestamp(mtime).strftime(
                "%Y-%m-%d %H:%M UTC"
            )
            return True
        except Exception:
            pass
    return False


def save_cache(df: pd.DataFrame):
    try:
        df.to_parquet(CACHE_PATH, index=False)
    except Exception as exc:
        logging.warning(f"Cache save failed: {exc}")


def apply_filters(
    df: pd.DataFrame,
    stages: list,
    companies: list,
    modalities: list,
    sources: list,
    search: str,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if stages:
        mask &= df["stage"].isin(stages)
    if companies:
        mask &= df["company"].isin(companies)
    if modalities:
        mask &= df["modality"].isin(modalities)
    if sources:
        # sources column is a semicolon-joined string; match any
        mask &= df["sources"].apply(
            lambda s: any(src in s for src in sources)
        )
    if search:
        q = search.lower()
        text_cols = ["brand_name", "inn_name", "company", "indication",
                     "mechanism_of_action", "target_name"]
        text_mask = df[text_cols].fillna("").apply(
            lambda col: col.str.lower().str.contains(q, regex=False)
        ).any(axis=1)
        mask &= text_mask
    return df[mask].reset_index(drop=True)


def to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Assets", index=False)
        mkt = df[df["stage"] == "Marketed"]
        if not mkt.empty:
            mkt.to_excel(writer, sheet_name="Marketed", index=False)
        pipe = df[df["stage"].isin(["Ph1", "Ph2", "Ph3"])]
        if not pipe.empty:
            pipe.to_excel(writer, sheet_name="Pipeline", index=False)
    return buf.getvalue()


def stage_badge(stage: str) -> str:
    colours = {
        "Marketed": "🟢",
        "Ph3": "🔵",
        "Ph2": "🟡",
        "Ph1": "🟠",
        "Preclinical": "⚪",
    }
    return colours.get(stage, "⚫")


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_and_store():
    """Run the full pipeline with live status updates in the UI."""
    st.session_state.running = True
    log: list[str] = []

    status_box   = st.empty()
    progress_bar = st.progress(0)

    # Map keywords in status messages → approximate progress %
    PROGRESS_MAP = [
        ("OpenFDA",        5),
        ("ClinicalTrials", 20),
        ("EMA",            30),
        ("Standardiz",     35),
        ("Merged",         38),
        ("RxNorm",         45),
        ("RxNorm complete",60),
        ("Deduplicat",     65),
        ("ChEMBL",         70),
        ("ChEMBL enrichment complete", 92),
        ("Pipeline complete",          100),
    ]

    def on_status(msg: str):
        log.append(msg)
        status_box.info(f"⟳  {msg}")
        for keyword, pct in PROGRESS_MAP:
            if keyword.lower() in msg.lower():
                progress_bar.progress(pct)
                break

    try:
        df = run_pipeline(status_fn=on_status)
        progress_bar.progress(100)
        status_box.success(
            f"✓  Pipeline complete — **{len(df)} unique oncology assets** in universe"
        )
        st.session_state.universe = df
        st.session_state.run_log  = log
        st.session_state.last_run = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        save_cache(df)
    except Exception as exc:
        status_box.error(f"Pipeline error: {exc}")
        st.exception(exc)
    finally:
        st.session_state.running = False


# ── Load cached data on first visit ───────────────────────────────────────────
if st.session_state.universe is None:
    load_cached()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧬 Oncology Asset Universe")
    st.caption("OpenFDA · ClinicalTrials.gov · EMA EPAR · ChEMBL")
    st.divider()

    # Pipeline control
    btn_label = "⟳  Refresh Pipeline" if st.session_state.universe is not None else "▶  Run Pipeline"
    if st.button(btn_label, type="primary", use_container_width=True,
                 disabled=st.session_state.running):
        run_and_store()
        st.rerun()

    if st.session_state.last_run:
        st.caption(f"Last run: {st.session_state.last_run}")

    st.divider()

    # Filters (only when data is loaded)
    df_full: pd.DataFrame = st.session_state.universe

    sel_stages     = []
    sel_companies  = []
    sel_modalities = []
    sel_sources    = []
    search_query   = ""

    if df_full is not None and not df_full.empty:
        st.subheader("Filters")

        search_query = st.text_input("🔍  Search", placeholder="drug name, target, MoA…")

        all_stages = sorted(
            df_full["stage"].dropna().unique(),
            key=lambda s: {"Marketed": 0, "Ph3": 1, "Ph2": 2, "Ph1": 3}.get(s, 9)
        )
        sel_stages = st.multiselect("Stage", all_stages, default=all_stages)

        all_companies = sorted([c for c in df_full["company"].dropna().unique() if c])
        sel_companies = st.multiselect("Company", all_companies)

        all_modalities = sorted([m for m in df_full["modality"].dropna().unique() if m])
        sel_modalities = st.multiselect("Modality", all_modalities)

        all_sources_flat = sorted(
            {s.strip()
             for row in df_full["sources"].dropna()
             for s in row.split(";")
             if s.strip()}
        )
        sel_sources = st.multiselect("Source", all_sources_flat)

        st.divider()

        # Download
        filtered_for_dl = apply_filters(
            df_full, sel_stages, sel_companies, sel_modalities, sel_sources, search_query
        )
        st.download_button(
            "⬇  Export Excel",
            data=to_excel(filtered_for_dl),
            file_name="oncology_asset_universe.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


# ── Main area ─────────────────────────────────────────────────────────────────

if st.session_state.universe is None:
    # Empty state
    st.markdown("## Oncology Asset Universe")
    st.markdown(
        "This tool automatically pulls, standardizes, and deduplicates oncology drug "
        "assets across **OpenFDA**, **ClinicalTrials.gov**, and **EMA EPAR**, then "
        "enriches each asset with mechanism-of-action and modality data from **ChEMBL**."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1** — Click **Run Pipeline** in the sidebar to pull fresh data from all sources.")
    with col2:
        st.info("**Step 2** — Brand normalization via RxNorm collapses dosage variants into one record per brand.")
    with col3:
        st.info("**Step 3** — Filter, search, and export the deduplicated universe as Excel.")

    st.markdown("---")
    st.markdown("**Estimated run time:** 8–15 minutes (API rate limits) · All sources are free / no API key required.")
    st.stop()


# ── Apply filters ─────────────────────────────────────────────────────────────

df = apply_filters(
    st.session_state.universe,
    sel_stages, sel_companies, sel_modalities, sel_sources, search_query,
)

# ── KPI metrics ───────────────────────────────────────────────────────────────

total_assets  = len(df)
marketed      = (df["stage"] == "Marketed").sum()
pipeline_ct   = df["stage"].isin(["Ph1", "Ph2", "Ph3"]).sum()
chembl_matched = df["chembl_id"].astype(str).str.len().gt(3).sum()
dual_listed   = df["sources"].str.contains(";").sum()  # appears in 2+ sources

st.markdown("### Oncology Asset Universe")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Assets",     total_assets)
k2.metric("🟢 Marketed",       marketed)
k3.metric("🔵 Pipeline",       pipeline_ct,
          help="Ph1 + Ph2 + Ph3")
k4.metric("⚗ ChEMBL Matched", chembl_matched,
          help="Assets with MoA / modality data")
k5.metric("⊕ Multi-source",   dual_listed,
          help="Assets confirmed across 2+ databases")

st.divider()

# ── Source breakdown ──────────────────────────────────────────────────────────
src_counts = {
    "OpenFDA":        df["sources"].str.contains("OpenFDA").sum(),
    "ClinicalTrials": df["sources"].str.contains("ClinicalTrials").sum(),
    "EMA":            df["sources"].str.contains("EMA").sum(),
}
c1, c2, c3 = st.columns(3)
for col, (src, cnt) in zip([c1, c2, c3], src_counts.items()):
    col.caption(f"**{src}**: {cnt} assets")

# ── Asset table ───────────────────────────────────────────────────────────────

# Select and rename display columns
DISPLAY_COLS = {
    "brand_name":          "Brand",
    "inn_name":            "INN / Generic",
    "company":             "Company",
    "stage":               "Stage",
    "all_stages":          "All Stages",
    "region":              "Region",
    "modality":            "Modality",
    "mechanism_of_action": "MoA",
    "target_name":         "Target",
    "indication":          "Indication",
    "sources":             "Sources",
    "rxnorm_cui":          "RxNorm CUI",
    "record_count":        "Records collapsed",
}

display_df = df[[c for c in DISPLAY_COLS if c in df.columns]].rename(columns=DISPLAY_COLS)

st.dataframe(
    display_df,
    use_container_width=True,
    height=520,
    column_config={
        "Brand":              st.column_config.TextColumn(width="medium"),
        "INN / Generic":      st.column_config.TextColumn(width="medium"),
        "Company":            st.column_config.TextColumn(width="medium"),
        "Stage":              st.column_config.TextColumn(width="small"),
        "All Stages":         st.column_config.TextColumn(width="medium"),
        "Region":             st.column_config.TextColumn(width="small"),
        "Modality":           st.column_config.TextColumn(width="medium"),
        "MoA":                st.column_config.TextColumn(width="large"),
        "Target":             st.column_config.TextColumn(width="large"),
        "Indication":         st.column_config.TextColumn(width="large"),
        "Sources":            st.column_config.TextColumn(width="medium"),
        "RxNorm CUI":         st.column_config.TextColumn(width="small"),
        "Records collapsed":  st.column_config.NumberColumn(width="small",
                                help="How many raw records were merged into this brand row"),
    },
)

st.caption(f"Showing **{len(display_df)}** assets · filtered from {len(st.session_state.universe)} total")

# ── Dedup transparency expander ───────────────────────────────────────────────

with st.expander("🔍  Deduplication detail — see how records were collapsed"):
    st.markdown(
        """
        Each row in the table above represents one **canonical brand** assembled from
        potentially many raw source records.

        **Brand key** — The universal dedup identifier:
        - If the brand is found in **RxNorm**, the key is its **CUI** (e.g. `1234567`).
          All dosage variants (50mg, 100mg XR, IV solution…) map to the same BN-root CUI.
        - If not in RxNorm (common for trial code names like MK-3475), a **fuzzy string key**
          is generated: lowercased, punctuation removed, release-modifier suffixes stripped.
          The company is appended to prevent cross-sponsor collisions.

        **Stage resolution** — When a brand appears as both Marketed (OpenFDA) and in
        active trials (ClinicalTrials), `stage` = **Marketed** (highest priority),
        and `all_stages` shows the full picture.
        """
    )

    debug_cols = {
        "brand_name":  "Brand",
        "brand_key":   "Brand Key",
        "rxnorm_cui":  "RxNorm CUI",
        "stage":       "Stage",
        "all_stages":  "All Stages",
        "sources":     "Sources",
        "record_count":"Records collapsed",
        "nct_ids":     "NCT IDs",
        "application_numbers": "NDA/BLA #",
    }
    debug_df = df[[c for c in debug_cols if c in df.columns]].rename(columns=debug_cols)
    st.dataframe(debug_df, use_container_width=True, height=360)


# ── Pipeline log expander ─────────────────────────────────────────────────────

if st.session_state.run_log:
    with st.expander("📋  Pipeline run log"):
        for line in st.session_state.run_log:
            st.text(line)
