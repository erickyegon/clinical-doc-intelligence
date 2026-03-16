#!/usr/bin/env python3
"""
FDA Multi-Source Data Seeder
Downloads real data from ALL four public FDA data sources:

  1. openFDA Drug Labels    → api.fda.gov/drug/label.json (70K+ structured labels)
  2. DailyMed SPL files     → dailymed.nlm.nih.gov (raw XML prescribing info)
  3. ClinicalTrials.gov     → clinicaltrials.gov/api/v2 (400K+ trial records)
  4. FDA Orange Book        → fda.gov (patent and exclusivity data)

Usage:
    python scripts/seed_data.py                    # Download all sources
    python scripts/seed_data.py --class "Statins"  # Specific drug class
    python scripts/seed_data.py --drug "Ozempic"   # Specific drug
    python scripts/seed_data.py --list-sources     # Show all sources
    python scripts/seed_data.py --skip-trials      # Skip ClinicalTrials.gov
    python scripts/seed_data.py --skip-orangebook  # Skip Orange Book download

Populates:
    data/sample_labels/          (openFDA JSON + DailyMed XML)
    data/sample_clinicaltrials/  (trial JSON records)
    data/sample_orangebook/      (Patent.txt, Products.txt, Exclusivity.txt)
"""
import sys
import os
import json
import argparse
import logging
import re
import time
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.fda_labels import FDALabelClient, LocalLabelLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Directories
# ============================================================
DATA_DIR = Path("data")
LABELS_DIR = DATA_DIR / "sample_labels"
TRIALS_DIR = DATA_DIR / "sample_clinicaltrials"
ORANGE_DIR = DATA_DIR / "sample_orangebook"

for d in (LABELS_DIR, TRIALS_DIR, ORANGE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Drug classes and priority drugs
# ============================================================
DEFAULT_DRUG_CLASSES = {
    "SGLT2 Inhibitors": {
        "search_term": "Sodium-Glucose Transporter 2 (SGLT2) Inhibitor [EPC]",
        "example_drugs": ["Jardiance", "Farxiga", "Invokana"],
        "relevance": "Diabetes - high HEDIS/STARS relevance for MA plans",
    },
    "GLP-1 Receptor Agonists": {
        "search_term": "Glucagon-Like Peptide 1 (GLP-1) Receptor Agonist [EPC]",
        "example_drugs": ["Ozempic", "Trulicity", "Mounjaro"],
        "relevance": "Diabetes/Obesity - top pharma RWE focus area",
    },
    "Statins": {
        "search_term": "HMG-CoA Reductase Inhibitor",
        "example_drugs": ["Lipitor", "Crestor", "Zocor"],
        "relevance": "Cardiovascular - HEDIS statin therapy measures",
    },
    "ACE Inhibitors": {
        "search_term": "Angiotensin Converting Enzyme Inhibitor",
        "example_drugs": ["Lisinopril", "Enalapril", "Ramipril"],
        "relevance": "Cardiovascular/Renal - core managed care medications",
    },
    "PD-1 Inhibitors": {
        "search_term": "Programmed Death Receptor-1 Blocking Antibody",
        "example_drugs": ["Keytruda", "Opdivo"],
        "relevance": "Oncology - highest-revenue drug class in pharma",
    },
    "DPP-4 Inhibitors": {
        "search_term": "Dipeptidyl Peptidase 4 Inhibitor",
        "example_drugs": ["Januvia", "Tradjenta", "Onglyza"],
        "relevance": "Diabetes - formulary comparison use case",
    },
}

PRIORITY_DRUGS = [
    "metformin", "semaglutide", "empagliflozin", "atorvastatin",
    "lisinopril", "amlodipine", "pembrolizumab", "sitagliptin",
    "dapagliflozin", "liraglutide", "rosuvastatin", "canagliflozin",
]


# ============================================================
# Source 1: openFDA Drug Labels
# ============================================================

def seed_by_class(client, loader, class_name, class_config, max_per_class=15):
    """Download labels by therapeutic class from openFDA."""
    search_term = class_config["search_term"]
    logger.info(f"Downloading {class_name} ({search_term})...")

    raw_labels = client.fetch_labels_by_drug_class(search_term, max_results=max_per_class)
    if raw_labels:
        safe_name = class_name.lower().replace(" ", "_").replace("-", "_")
        loader.save_labels(raw_labels, f"class_{safe_name}.json")
        drug_names = set()
        for raw in raw_labels:
            label = client.parse_label(raw)
            drug_names.add(label.drug_name)
        logger.info(f"  -> {len(raw_labels)} labels, drugs: {', '.join(sorted(drug_names)[:10])}")
    else:
        logger.warning(f"  -> No labels found for {class_name}")

    return raw_labels


def seed_by_drug(client, loader, drug_name):
    """Download labels for a specific drug from openFDA."""
    logger.info(f"Downloading {drug_name}...")

    raw_labels = client.search_labels(drug_name=drug_name, limit=3)
    if raw_labels:
        safe_name = drug_name.lower().replace(" ", "_").replace("-", "_")
        loader.save_labels(raw_labels, f"drug_{safe_name}.json")
        label = client.parse_label(raw_labels[0])
        logger.info(f"  -> {label.drug_name} ({label.generic_name}): {len(label.sections)} sections")
    else:
        logger.warning(f"  -> No labels found for {drug_name}")

    return raw_labels


# ============================================================
# Source 2: DailyMed SPL XML
# ============================================================

class DailyMedClient:
    """Downloads raw SPL XML prescribing information from DailyMed."""

    BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

    def fetch_spl_by_drug(self, drug_name, max_results=2):
        """Search by drug name, get set IDs, download full XML."""
        import httpx

        downloaded = []
        try:
            search_url = f"{self.BASE_URL}/spls.json"
            params = {"drug_name": drug_name, "pagesize": max_results}
            resp = httpx.get(search_url, params=params, timeout=30.0)
            resp.raise_for_status()
            data = resp.json().get("data", [])

            for item in data:
                setid = item.get("setid")
                if not setid:
                    continue

                xml_url = f"{self.BASE_URL}/spls/{setid}.xml"
                xml_resp = httpx.get(xml_url, timeout=30.0)
                xml_resp.raise_for_status()

                safe_name = drug_name.lower().replace(" ", "_")
                filepath = LABELS_DIR / f"dailymed_{safe_name}_{setid[:8]}.xml"
                filepath.write_text(xml_resp.text, encoding="utf-8")
                downloaded.append(filepath.name)

            if downloaded:
                logger.info(f"  -> DailyMed: {len(downloaded)} SPL XML for {drug_name}")
            else:
                logger.info(f"  -> DailyMed: no results for {drug_name}")

        except Exception as e:
            logger.warning(f"  -> DailyMed failed for {drug_name}: {e}")

        return downloaded


# ============================================================
# Source 3: ClinicalTrials.gov API V2
# ============================================================

class ClinicalTrialsClient:
    """Downloads trial records from ClinicalTrials.gov API V2."""

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def fetch_trials_by_drug(self, drug_name, max_results=8):
        """Search trials by drug intervention name."""
        import httpx

        try:
            params = {
                "query.intr": drug_name,
                "pageSize": max_results,
                "format": "json",
            }
            resp = httpx.get(self.BASE_URL, params=params, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()

            safe_name = drug_name.lower().replace(" ", "_")
            filepath = TRIALS_DIR / f"trials_{safe_name}.json"
            filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")

            count = len(data.get("studies", []))
            if count:
                logger.info(f"  -> ClinicalTrials.gov: {count} studies for {drug_name}")
            return count

        except Exception as e:
            logger.warning(f"  -> ClinicalTrials.gov failed for {drug_name}: {e}")
            return 0


# ============================================================
# Source 4: FDA Orange Book
# ============================================================

class OrangeBookClient:
    """Downloads and extracts the official FDA Orange Book ZIP."""

    PAGE_URL = "https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files"

    def download_latest(self):
        """Find and download the latest Orange Book ZIP from the FDA website."""
        import httpx

        logger.info("Downloading FDA Orange Book data...")
        try:
            # Find the ZIP link on the FDA page
            page_resp = httpx.get(self.PAGE_URL, timeout=30.0, follow_redirects=True)
            page_resp.raise_for_status()

            match = re.search(r'href="(https?://[^"]+\.zip)"', page_resp.text, re.IGNORECASE)
            if not match:
                logger.warning("  -> Could not find Orange Book ZIP link on FDA page.")
                logger.warning(f"     Download manually from: {self.PAGE_URL}")
                return False

            zip_url = match.group(1)
            logger.info(f"  -> Found ZIP: {zip_url}")

            zip_path = ORANGE_DIR / "orangebook.zip"
            with httpx.stream("GET", zip_url, timeout=60.0, follow_redirects=True) as stream:
                stream.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in stream.iter_bytes(chunk_size=8192):
                        f.write(chunk)

            # Extract all files
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(ORANGE_DIR)

            # Clean up ZIP
            zip_path.unlink(missing_ok=True)

            extracted = [f.name for f in ORANGE_DIR.iterdir() if f.suffix in (".txt", ".csv")]
            logger.info(f"  -> Orange Book extracted: {', '.join(extracted)}")
            return True

        except Exception as e:
            logger.warning(f"  -> Orange Book download failed: {e}")
            logger.warning(f"     Download manually from: {self.PAGE_URL}")
            return False


# ============================================================
# Main Orchestration
# ============================================================

def seed_all(max_per_class=15, skip_trials=False, skip_orangebook=False):
    """Download from ALL four public FDA data sources."""
    print("=" * 70)
    print("FDA MULTI-SOURCE DATA SEEDER")
    print("4 Public Datasets: openFDA + DailyMed + ClinicalTrials + Orange Book")
    print("=" * 70)

    total_labels = 0
    total_drugs = set()

    # --- Phase 1: openFDA Labels by Class ---
    print("\n--- Phase 1: openFDA Drug Labels by Therapeutic Class ---")
    client = FDALabelClient()
    loader = LocalLabelLoader()

    for class_name, config in DEFAULT_DRUG_CLASSES.items():
        try:
            labels = seed_by_class(client, loader, class_name, config, max_per_class)
            total_labels += len(labels)
            for raw in labels:
                parsed = client.parse_label(raw)
                total_drugs.add(parsed.drug_name)
        except Exception as e:
            logger.error(f"Failed: {class_name}: {e}")

    # --- Phase 2: openFDA Labels by Drug Name ---
    print("\n--- Phase 2: openFDA Priority Individual Drugs ---")
    for drug_name in PRIORITY_DRUGS:
        try:
            labels = seed_by_drug(client, loader, drug_name)
            total_labels += len(labels)
            for raw in labels:
                parsed = client.parse_label(raw)
                total_drugs.add(parsed.drug_name)
        except Exception as e:
            logger.error(f"Failed: {drug_name}: {e}")
    client.close()

    # --- Phase 3: DailyMed SPL XML ---
    print("\n--- Phase 3: DailyMed Raw SPL XML Files ---")
    dailymed = DailyMedClient()
    dailymed_count = 0
    for drug in PRIORITY_DRUGS[:8]:
        files = dailymed.fetch_spl_by_drug(drug, max_results=2)
        dailymed_count += len(files)
        time.sleep(0.5)  # Rate limiting courtesy

    # --- Phase 4: ClinicalTrials.gov ---
    if not skip_trials:
        print("\n--- Phase 4: ClinicalTrials.gov Trial Records ---")
        trials_client = ClinicalTrialsClient()
        trials_count = 0
        for drug in PRIORITY_DRUGS:
            count = trials_client.fetch_trials_by_drug(drug, max_results=8)
            trials_count += count
            time.sleep(0.3)  # Rate limiting courtesy
    else:
        print("\n--- Phase 4: ClinicalTrials.gov SKIPPED (--skip-trials) ---")
        trials_count = 0

    # --- Phase 5: FDA Orange Book ---
    if not skip_orangebook:
        print("\n--- Phase 5: FDA Orange Book (Patents & Exclusivity) ---")
        orange = OrangeBookClient()
        orange.download_latest()
    else:
        print("\n--- Phase 5: Orange Book SKIPPED (--skip-orangebook) ---")

    # --- Summary ---
    label_files = len(list(LABELS_DIR.glob("*.json")))
    xml_files = len(list(LABELS_DIR.glob("*.xml")))
    trial_files = len(list(TRIALS_DIR.glob("*.json")))
    orange_files = len(list(ORANGE_DIR.glob("*.*")))

    print("\n" + "=" * 70)
    print("SEEDING COMPLETE — ALL SOURCES")
    print("=" * 70)
    print(f"  openFDA Labels:       {total_labels} labels, {len(total_drugs)} unique drugs ({label_files} JSON files)")
    print(f"  DailyMed SPL XML:     {dailymed_count} files ({xml_files} total XML in {LABELS_DIR})")
    print(f"  ClinicalTrials.gov:   {trials_count} studies ({trial_files} files in {TRIALS_DIR})")
    print(f"  Orange Book:          {orange_files} files in {ORANGE_DIR}")
    print()
    print("Next step: python scripts/ingest.py")
    print("=" * 70)


def list_sources():
    """Display all supported data sources."""
    print("\nSupported Public FDA Data Sources:")
    print("=" * 65)
    print(f"  1. openFDA Drug Labels     -> {LABELS_DIR}/*.json")
    print(f"     70,000+ structured drug labels")
    print(f"     API: api.fda.gov/drug/label.json")
    print()
    print(f"  2. DailyMed SPL Files      -> {LABELS_DIR}/*.xml")
    print(f"     Raw XML prescribing information (SPL format)")
    print(f"     API: dailymed.nlm.nih.gov/dailymed/services/v2")
    print()
    print(f"  3. ClinicalTrials.gov      -> {TRIALS_DIR}/*.json")
    print(f"     400,000+ clinical trial records")
    print(f"     API: clinicaltrials.gov/api/v2")
    print()
    print(f"  4. FDA Orange Book         -> {ORANGE_DIR}/*.txt")
    print(f"     Patent and exclusivity data for generic drug comparisons")
    print(f"     Source: fda.gov/drugs/orange-book-data-files")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FDA data from all 4 public sources")
    parser.add_argument("--class", dest="drug_class", help="Download specific drug class")
    parser.add_argument("--drug", help="Download specific drug by name")
    parser.add_argument("--max-per-class", type=int, default=15)
    parser.add_argument("--list-sources", action="store_true", help="List all data sources")
    parser.add_argument("--skip-trials", action="store_true", help="Skip ClinicalTrials.gov")
    parser.add_argument("--skip-orangebook", action="store_true", help="Skip Orange Book download")

    args = parser.parse_args()

    if args.list_sources:
        list_sources()
    elif args.drug:
        client = FDALabelClient()
        loader = LocalLabelLoader()
        seed_by_drug(client, loader, args.drug)
        client.close()
    elif args.drug_class:
        client = FDALabelClient()
        loader = LocalLabelLoader()
        if args.drug_class in DEFAULT_DRUG_CLASSES:
            seed_by_class(client, loader, args.drug_class,
                          DEFAULT_DRUG_CLASSES[args.drug_class], args.max_per_class)
        else:
            seed_by_class(client, loader, args.drug_class,
                          {"search_term": args.drug_class}, args.max_per_class)
        client.close()
    else:
        seed_all(args.max_per_class, skip_trials=args.skip_trials, skip_orangebook=args.skip_orangebook)
