#!/usr/bin/env python3
"""
Multi-Source Ingestion Script
Indexes data from all four public FDA sources into the vector store:

  1. openFDA Drug Labels (JSON)     → data/sample_labels/*.json
  2. DailyMed SPL files (XML)       → data/sample_labels/*.xml
  3. ClinicalTrials.gov (JSON)      → data/sample_clinicaltrials/*.json
  4. FDA Orange Book (TXT)          → data/sample_orangebook/*.txt

Usage:
    python scripts/ingest.py                  # Index all sources
    python scripts/ingest.py --reset          # Clear vector store and re-index
    python scripts/ingest.py --drug "Ozempic" # Fetch and index a specific drug
    python scripts/ingest.py --stats          # Show current index statistics
    python scripts/ingest.py --labels-only    # Only openFDA JSON + DailyMed XML
    python scripts/ingest.py --trials-only    # Only ClinicalTrials.gov
    python scripts/ingest.py --orangebook-only # Only FDA Orange Book
    python scripts/ingest.py --test           # Run a quick test query after indexing
"""
import sys
import os
import json
import re
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.vector_store import VectorStoreManager
from src.processing.chunker import SectionAwareChunker
from src.processing.metadata import MetadataExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
LABELS_DIR = DATA_DIR / "sample_labels"
TRIALS_DIR = DATA_DIR / "sample_clinicaltrials"
ORANGE_DIR = DATA_DIR / "sample_orangebook"

# Section priority scores for non-openFDA sources
SECTION_PRIORITIES = {
    "boxed_warning": 10, "contraindications": 9, "warnings_and_precautions": 8,
    "warnings": 8, "adverse_reactions": 7, "drug_interactions": 7,
    "dosage_and_administration": 6, "indications_and_usage": 6,
    "indications": 6, "clinical_pharmacology": 4, "clinical_studies": 5,
    "use_in_specific_populations": 5, "overdosage": 4, "description": 3,
    "how_supplied": 2, "patient_counseling_information": 3,
}


def show_stats(vector_store):
    """Display current vector store statistics across all sources."""
    doc_count = vector_store.get_document_count()
    drugs = vector_store.get_all_drug_names()

    print("\n" + "=" * 60)
    print("VECTOR STORE STATISTICS (ALL SOURCES)")
    print("=" * 60)
    print(f"  Total chunks indexed: {doc_count}")
    print(f"  Unique drugs: {len(drugs)}")
    if drugs:
        print(f"  Drugs: {', '.join(sorted(drugs)[:20])}")
        if len(drugs) > 20:
            print(f"  ... and {len(drugs) - 20} more")
    print("=" * 60 + "\n")


# ============================================================
# Source 1: openFDA Drug Labels (JSON) — existing pipeline
# ============================================================

def ingest_openfda_labels(vector_store, reset=False):
    """Index openFDA JSON labels using the existing IngestionPipeline."""
    if reset:
        logger.info("Resetting vector store...")
        vector_store.reset()

    pipeline = IngestionPipeline(
        vector_store=vector_store,
        chunker=SectionAwareChunker(),
        metadata_extractor=MetadataExtractor(),
    )

    print("\n--- Source 1: openFDA Drug Labels (JSON) ---")
    stats = pipeline.ingest_from_local()

    print(f"  Labels fetched/parsed: {stats.labels_fetched}/{stats.labels_parsed}")
    print(f"  Chunks created/indexed: {stats.chunks_created}/{stats.chunks_indexed}")
    print(f"  Errors: {stats.errors}")

    pipeline.close()
    return stats


# ============================================================
# Source 2: DailyMed SPL XML
# ============================================================

def ingest_dailymed_xml(vector_store):
    """Parse DailyMed SPL XML files and index into the vector store."""
    xml_files = list(LABELS_DIR.glob("*.xml"))
    if not xml_files:
        print("\n--- Source 2: DailyMed SPL XML ---")
        print("  No XML files found. Run: python scripts/seed_data.py")
        return 0

    print(f"\n--- Source 2: DailyMed SPL XML ({len(xml_files)} files) ---")
    total_chunks = 0

    for xml_file in xml_files:
        try:
            content = xml_file.read_text(encoding="utf-8")

            # Extract drug name from filename: dailymed_metformin_abc12345.xml
            name_match = re.match(r"dailymed_(.+?)_[a-f0-9]+\.xml", xml_file.name)
            drug_name = name_match.group(1).replace("_", " ").title() if name_match else xml_file.stem

            # Parse sections from SPL XML
            sections = _parse_spl_xml(content)
            has_boxed = any("boxed" in k or "black_box" in k for k in sections)

            for section_name, section_text in sections.items():
                if len(section_text.strip()) < 50:
                    continue

                chunk_id = f"dailymed_{xml_file.stem}_{section_name}"
                display_name = section_name.replace("_", " ").title()
                priority = SECTION_PRIORITIES.get(section_name, 3)

                metadata = {
                    "source_type": "dailymed_spl",
                    "drug_name": drug_name,
                    "section_type": section_name,
                    "section_display_name": display_name,
                    "label_id": xml_file.stem,
                    "generic_name": drug_name,
                    "manufacturer": "",
                    "approval_date": "",
                    "therapeutic_area": "",
                    "normalized_therapeutic_area": "",
                    "has_boxed_warning": has_boxed,
                    "safety_flags": "",
                    "section_priority": priority,
                    "chunk_index": 0,
                    "total_chunks": 1,
                }

                # Truncate very long sections to 2000 chars per chunk
                text_content = f"[{display_name}] {section_text[:2000]}"

                vector_store.add_documents(
                    texts=[text_content],
                    metadatas=[metadata],
                    ids=[chunk_id],
                )
                total_chunks += 1

            logger.info(f"  {xml_file.name}: {len(sections)} sections indexed")

        except Exception as e:
            logger.warning(f"  Failed to parse {xml_file.name}: {e}")

    print(f"  Total DailyMed chunks indexed: {total_chunks}")
    return total_chunks


def _parse_spl_xml(xml_content):
    """Extract text sections from SPL XML using regex (no lxml dependency)."""
    sections = {}

    # SPL sections are in <section> tags with <title> and <text> children
    section_pattern = re.compile(
        r'<section[^>]*>.*?<title[^>]*>(.*?)</title>.*?<text[^>]*>(.*?)</text>',
        re.DOTALL | re.IGNORECASE
    )

    for match in section_pattern.finditer(xml_content):
        title = re.sub(r'<[^>]+>', '', match.group(1)).strip()
        text = re.sub(r'<[^>]+>', ' ', match.group(2))
        text = re.sub(r'\s+', ' ', text).strip()

        if title and text and len(text) > 30:
            key = title.lower().replace(" ", "_").replace(",", "").replace("&", "and")
            key = re.sub(r'[^a-z0-9_]', '', key)
            sections[key] = text

    return sections


# ============================================================
# Source 3: ClinicalTrials.gov
# ============================================================

def ingest_clinical_trials(vector_store):
    """Parse and index locally downloaded ClinicalTrials.gov JSON files."""
    trial_files = list(TRIALS_DIR.glob("*.json"))
    if not trial_files:
        print("\n--- Source 3: ClinicalTrials.gov ---")
        print("  No trial files found. Run: python scripts/seed_data.py")
        return 0

    print(f"\n--- Source 3: ClinicalTrials.gov ({len(trial_files)} files) ---")
    total_chunks = 0

    for trial_file in trial_files:
        try:
            data = json.loads(trial_file.read_text(encoding="utf-8"))
            studies = data.get("studies", [])
            file_chunks = 0

            for study in studies:
                proto = study.get("protocolSection", {})
                ident = proto.get("identificationModule", {})
                status_mod = proto.get("statusModule", {})
                desc = proto.get("descriptionModule", {})
                design = proto.get("designModule", {})
                elig = proto.get("eligibilityModule", {})
                outcomes = proto.get("outcomesModule", {})
                arms = proto.get("armsInterventionsModule", {})

                nct_id = ident.get("nctId", "unknown")
                title = ident.get("officialTitle", ident.get("briefTitle", "Untitled"))
                phase = ", ".join(design.get("phases", [])) or "Unknown"
                trial_status = status_mod.get("overallStatus", "Unknown")

                # Extract drug names from interventions
                interventions = arms.get("interventions", [])
                drug_names = [i.get("name", "") for i in interventions
                              if i.get("type", "").upper() == "DRUG"]
                drug_name_str = ", ".join(drug_names) if drug_names else (
                    trial_file.stem.replace("trials_", "").replace("_", " ").title()
                )

                base_meta = {
                    "source_type": "clinical_trial",
                    "drug_name": drug_name_str,
                    "nct_id": nct_id,
                    "trial_status": trial_status,
                    "trial_phase": phase,
                    "label_id": nct_id,
                    "generic_name": drug_name_str,
                    "manufacturer": "",
                    "approval_date": "",
                    "therapeutic_area": "",
                    "normalized_therapeutic_area": "",
                    "has_boxed_warning": False,
                    "safety_flags": "",
                    "section_priority": 4,
                    "chunk_index": 0,
                    "total_chunks": 1,
                }

                # Chunk 1: Trial Summary
                summary = desc.get("briefSummary", "")
                if summary and len(summary) > 50:
                    meta = {**base_meta,
                            "section_type": "trial_summary",
                            "section_display_name": "Trial Summary"}
                    vector_store.add_documents(
                        texts=[f"[Trial Summary - {nct_id}] {title}. Phase: {phase}. Status: {trial_status}. {summary}"],
                        metadatas=[meta],
                        ids=[f"trial_{nct_id}_summary"],
                    )
                    file_chunks += 1

                # Chunk 2: Eligibility Criteria
                criteria = elig.get("eligibilityCriteria", "")
                if criteria and len(criteria) > 50:
                    meta = {**base_meta,
                            "section_type": "eligibility_criteria",
                            "section_display_name": "Eligibility Criteria"}
                    vector_store.add_documents(
                        texts=[f"[Eligibility Criteria - {nct_id}] {criteria[:2000]}"],
                        metadatas=[meta],
                        ids=[f"trial_{nct_id}_eligibility"],
                    )
                    file_chunks += 1

                # Chunk 3: Primary Outcomes
                primary = outcomes.get("primaryOutcomes", [])
                if primary:
                    outcomes_text = "; ".join(
                        f"{o.get('measure', 'N/A')} (timeframe: {o.get('timeFrame', 'N/A')})"
                        for o in primary
                    )
                    meta = {**base_meta,
                            "section_type": "primary_outcomes",
                            "section_display_name": "Primary Outcomes"}
                    vector_store.add_documents(
                        texts=[f"[Primary Outcomes - {nct_id}] {outcomes_text}"],
                        metadatas=[meta],
                        ids=[f"trial_{nct_id}_outcomes"],
                    )
                    file_chunks += 1

                # Chunk 4: Detailed Description (if available)
                detailed = desc.get("detailedDescription", "")
                if detailed and len(detailed) > 100:
                    meta = {**base_meta,
                            "section_type": "trial_description",
                            "section_display_name": "Detailed Description"}
                    vector_store.add_documents(
                        texts=[f"[Detailed Description - {nct_id}] {detailed[:2000]}"],
                        metadatas=[meta],
                        ids=[f"trial_{nct_id}_detailed"],
                    )
                    file_chunks += 1

            total_chunks += file_chunks
            logger.info(f"  {trial_file.name}: {len(studies)} studies, {file_chunks} chunks")

        except Exception as e:
            logger.warning(f"  Failed to process {trial_file.name}: {e}")

    print(f"  Total trial chunks indexed: {total_chunks}")
    return total_chunks


# ============================================================
# Source 4: FDA Orange Book
# ============================================================

def ingest_orange_book(vector_store):
    """Parse and index FDA Orange Book patent/exclusivity data."""
    # Find Products.txt (case-insensitive)
    products_file = None
    for candidate in ["products.txt", "Products.txt", "PRODUCTS.TXT"]:
        p = ORANGE_DIR / candidate
        if p.exists():
            products_file = p
            break

    if not products_file:
        print("\n--- Source 4: FDA Orange Book ---")
        print("  No Products.txt found. Run: python scripts/seed_data.py")
        return 0

    print(f"\n--- Source 4: FDA Orange Book ---")
    total_chunks = 0

    try:
        lines = products_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) < 2:
            print("  Products.txt appears empty")
            return 0

        # Parse tilde-delimited header
        header = [h.strip().lower() for h in lines[0].split("~")]

        # Map column names to indices
        col_map = {}
        for i, col in enumerate(header):
            if "ingredient" in col:
                col_map["ingredient"] = i
            elif "trade_name" in col:
                col_map["trade_name"] = i
            elif "applicant" in col:
                col_map["applicant"] = i
            elif "appl_no" in col:
                col_map["appl_no"] = i
            elif "te_code" in col:
                col_map["te_code"] = i
            elif "type" in col and "type" not in col_map:
                col_map["type"] = i

        # Index products (first 500 for demo; full set is 30K+ rows)
        max_rows = 500
        for line in lines[1:max_rows + 1]:
            fields = line.split("~")
            if len(fields) < max(col_map.values(), default=0) + 1:
                continue

            ingredient = fields[col_map["ingredient"]].strip() if "ingredient" in col_map else ""
            trade_name = fields[col_map.get("trade_name", 1)].strip() if "trade_name" in col_map else ""
            applicant = fields[col_map.get("applicant", 2)].strip() if "applicant" in col_map else ""
            appl_no = fields[col_map.get("appl_no", 3)].strip() if "appl_no" in col_map else str(total_chunks)
            te_code = fields[col_map.get("te_code", 4)].strip() if "te_code" in col_map else ""

            if not ingredient:
                continue

            text = (
                f"[Orange Book Product] {ingredient} "
                f"(Trade Name: {trade_name}). "
                f"Applicant: {applicant}. "
                f"Application Number: {appl_no}. "
                f"Therapeutic Equivalence: {te_code}."
            )

            metadata = {
                "source_type": "orange_book",
                "drug_name": trade_name or ingredient,
                "generic_name": ingredient,
                "section_type": "orange_book_product",
                "section_display_name": "Orange Book Product",
                "label_id": f"ob_{appl_no}",
                "manufacturer": applicant,
                "approval_date": "",
                "therapeutic_area": "",
                "normalized_therapeutic_area": "",
                "has_boxed_warning": False,
                "safety_flags": "",
                "section_priority": 2,
                "chunk_index": 0,
                "total_chunks": 1,
            }

            vector_store.add_documents(
                texts=[text],
                metadatas=[metadata],
                ids=[f"ob_{appl_no}_{total_chunks}"],
            )
            total_chunks += 1

        print(f"  Orange Book products indexed: {total_chunks}")

    except Exception as e:
        logger.warning(f"  Orange Book ingestion failed: {e}")

    # Also index Patent.txt if present
    patent_chunks = _ingest_orange_book_patents(vector_store)
    total_chunks += patent_chunks

    return total_chunks


def _ingest_orange_book_patents(vector_store):
    """Index patent data from the Orange Book Patent.txt file."""
    patent_file = None
    for candidate in ["patent.txt", "Patent.txt", "PATENT.TXT"]:
        p = ORANGE_DIR / candidate
        if p.exists():
            patent_file = p
            break

    if not patent_file:
        return 0

    try:
        lines = patent_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) < 2:
            return 0

        header = [h.strip().lower() for h in lines[0].split("~")]
        col_map = {}
        for i, col in enumerate(header):
            if "appl_no" in col:
                col_map["appl_no"] = i
            elif "patent_no" in col:
                col_map["patent_no"] = i
            elif "patent_expire" in col:
                col_map["patent_expire"] = i
            elif "drug_substance" in col:
                col_map["drug_substance"] = i
            elif "drug_product" in col:
                col_map["drug_product"] = i

        count = 0
        for line in lines[1:300]:  # First 300 patents
            fields = line.split("~")
            if len(fields) < max(col_map.values(), default=0) + 1:
                continue

            appl_no = fields[col_map.get("appl_no", 0)].strip() if "appl_no" in col_map else ""
            patent_no = fields[col_map.get("patent_no", 1)].strip() if "patent_no" in col_map else ""
            expiry = fields[col_map.get("patent_expire", 2)].strip() if "patent_expire" in col_map else ""

            if not patent_no:
                continue

            text = f"[Orange Book Patent] Application {appl_no}, Patent {patent_no}, Expiry: {expiry}."
            metadata = {
                "source_type": "orange_book_patent",
                "drug_name": f"Application {appl_no}",
                "generic_name": "",
                "section_type": "orange_book_patent",
                "section_display_name": "Orange Book Patent",
                "label_id": f"obp_{appl_no}_{patent_no}",
                "manufacturer": "",
                "approval_date": "",
                "therapeutic_area": "",
                "normalized_therapeutic_area": "",
                "has_boxed_warning": False,
                "safety_flags": "",
                "section_priority": 2,
                "chunk_index": 0,
                "total_chunks": 1,
            }

            vector_store.add_documents(
                texts=[text], metadatas=[metadata], ids=[f"obp_{appl_no}_{patent_no}"],
            )
            count += 1

        if count:
            print(f"  Orange Book patents indexed: {count}")
        return count

    except Exception as e:
        logger.warning(f"  Patent ingestion failed: {e}")
        return 0


# ============================================================
# Main Orchestration
# ============================================================

def ingest_all(reset=False):
    """Ingest data from ALL four sources into the vector store."""
    vector_store = VectorStoreManager()

    print("\n" + "=" * 60)
    print("MULTI-SOURCE INGESTION PIPELINE")
    print("Sources: openFDA + DailyMed + ClinicalTrials + Orange Book")
    print("=" * 60)

    ingest_openfda_labels(vector_store, reset=reset)
    ingest_dailymed_xml(vector_store)
    ingest_clinical_trials(vector_store)
    ingest_orange_book(vector_store)

    print("\n" + "=" * 60)
    print("ALL SOURCES INDEXED!")
    show_stats(vector_store)


def test_query(vector_store):
    """Run a quick verification query."""
    print("\nTest Query: 'What are the contraindications for GLP-1 drugs?'")
    results = vector_store.query("What are the contraindications for GLP-1 drugs?", n_results=5)

    if results["documents"]:
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            source = meta.get("source_type", "unknown")
            drug = meta.get("drug_name", "Unknown")
            section = meta.get("section_display_name", "")
            print(f"\n  Result {i + 1} [{source}]: {drug} — {section}")
            print(f"  Preview: {doc[:180]}...")
    else:
        print("  No results. Run ingestion first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-source FDA ingestion into vector store")
    parser.add_argument("--drug", help="Fetch and index a specific drug from API")
    parser.add_argument("--reset", action="store_true", help="Clear vector store before indexing")
    parser.add_argument("--stats", action="store_true", help="Show index statistics only")
    parser.add_argument("--test", action="store_true", help="Run a test query after indexing")
    parser.add_argument("--labels-only", action="store_true", help="Only index labels (JSON + XML)")
    parser.add_argument("--trials-only", action="store_true", help="Only index clinical trials")
    parser.add_argument("--orangebook-only", action="store_true", help="Only index Orange Book")

    args = parser.parse_args()

    if args.stats:
        vs = VectorStoreManager()
        show_stats(vs)
    elif args.drug:
        vs = VectorStoreManager()
        pipeline = IngestionPipeline(
            vector_store=vs,
            chunker=SectionAwareChunker(),
            metadata_extractor=MetadataExtractor(),
        )
        stats = pipeline.ingest_single_drug(args.drug, include_trials=True)
        print(f"Results for {args.drug}: Labels={stats.labels_fetched}, Chunks={stats.chunks_indexed}")
        show_stats(vs)
        pipeline.close()
    elif args.labels_only:
        vs = VectorStoreManager()
        ingest_openfda_labels(vs, reset=args.reset)
        ingest_dailymed_xml(vs)
        show_stats(vs)
    elif args.trials_only:
        vs = VectorStoreManager()
        ingest_clinical_trials(vs)
        show_stats(vs)
    elif args.orangebook_only:
        vs = VectorStoreManager()
        ingest_orange_book(vs)
        show_stats(vs)
    else:
        ingest_all(reset=args.reset)
        if args.test:
            vs = VectorStoreManager()
            test_query(vs)
