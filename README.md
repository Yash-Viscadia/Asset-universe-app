# Oncology Asset Universe

Automated pipeline that pulls, standardizes, deduplicates, and enriches
oncology drug assets from four free public data sources into a single
brand-level universe — refreshable in one click.

---

## Data sources

| Source | Content | Filter used |
|---|---|---|
| **OpenFDA** `/drug/label` | US marketed oncology drugs | `pharm_class_epc = "Antineoplastic Agent [EPC]"` |
| **ClinicalTrials.gov** v2 | Industry-sponsored Ph1–3 oncology trials | Condition = Neoplasms, Sponsor = Industry |
| **EMA EPAR** (bulk Excel) | EU approved oncology medicines | Therapeutic area contains "oncol/haematol/cancer" or ATC code = L |
| **ChEMBL** (enrichment) | MoA, modality, target, ATC code | Looked up by INN per asset after dedup |

Brand normalization uses **RxNorm** (NLM, free) to collapse dosage variants
into a single brand-root CUI, then **ChEMBL** (EMBL-EBI, free) for mechanism
and modality data.

---

## Pipeline stages

```
Raw pulls (3 sources)
  ↓
Merge & standardize       — common schema, clean names, strip dosage suffixes
  ↓
Brand normalization        — RxNorm CUI lookup + brand-root traversal
  ↓
Deduplicate               — group by brand_key (CUI or fuzzy string) → 1 row per brand
  ↓
Enrich                    — ChEMBL MoA / modality / target
  ↓
Master universe output    — filterable table + Excel export
```

**Run time:** ~8–15 minutes on first run (API rate limits apply).
Subsequent visits within the same Streamlit container session load from cache
(near-instant). Click **Refresh Pipeline** to force a fresh pull.

---

## Deploy to Streamlit Community Cloud

1. **Fork or push this repo to GitHub** (repo must be public for the free tier).

2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.

3. Connect your GitHub account, select the repo, set:
   - **Main file path:** `app.py`
   - **Branch:** `main`

4. Click **Deploy**. Streamlit installs `requirements.txt` automatically.

5. First visit triggers the pipeline (8–15 min). Progress is shown live.
   Subsequent visitors within the same container get the cached result.

No API keys or secrets are required — all four sources are publicly accessible.

---

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Extending the pipeline

| Goal | Where to change |
|---|---|
| Add a new TA filter | `pipeline/pull.py` — adjust search params in each pull function |
| Add a new data source | Add `pull_X()` in `pull.py`, `standardize_X()` in `standardize.py`, call both in `orchestrator.py` |
| Change dedup grouping logic | `pipeline/deduplicate.py` — `_make_group_key()` and `_collapse_group()` |
| Add sales / revenue data | Requires a commercial source (Evaluate Pharma, IQVIA); add as an enrichment step after dedup |
| Persist data between deployments | Replace `/tmp/oncology_universe.parquet` cache with an external store (Supabase free tier, GitHub-committed Parquet via GitHub Actions) |
