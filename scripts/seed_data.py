#!/usr/bin/env python3
"""
FDA Drug Label Data Seeder
Downloads real drug labels from the openFDA API for demonstration and development.

Usage:
    python scripts/seed_data.py                    # Download all default classes
    python scripts/seed_data.py --class "Statin"   # Download specific class
    python scripts/seed_data.py --drug "Ozempic"   # Download specific drug
    python scripts/seed_data.py --list-classes      # Show available classes

This script populates data/sample_labels/ with real FDA label JSON files
that the ingestion pipeline can index into the vector store.
"""
import sys
import os
import json
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.fda_labels import FDALabelClient, LocalLabelLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Drug classes relevant to U.S. healthcare analytics / pharma RWE roles
# These cover diabetes, cardiovascular, oncology, and managed care focus areas
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

# Individual high-value drugs to fetch by name (covers gaps in class search)
PRIORITY_DRUGS = [
    "metformin",
    "semaglutide",
    "empagliflozin",
    "atorvastatin",
    "lisinopril",
    "amlodipine",
    "pembrolizumab",
    "sitagliptin",
    "dapagliflozin",
    "liraglutide",
    "rosuvastatin",
    "canagliflozin",
]


def seed_by_class(client: FDALabelClient, loader: LocalLabelLoader, 
                   class_name: str, class_config: dict, max_per_class: int = 15):
    """Download labels for a drug class."""
    search_term = class_config["search_term"]
    logger.info(f"Downloading {class_name} ({search_term})...")
    
    raw_labels = client.fetch_labels_by_drug_class(search_term, max_results=max_per_class)
    
    if raw_labels:
        safe_name = class_name.lower().replace(" ", "_").replace("-", "_")
        loader.save_labels(raw_labels, f"class_{safe_name}.json")
        
        # Parse and show what we got
        drug_names = set()
        for raw in raw_labels:
            label = client.parse_label(raw)
            drug_names.add(label.drug_name)
        
        logger.info(f"  → {len(raw_labels)} labels, drugs: {', '.join(sorted(drug_names)[:10])}")
    else:
        logger.warning(f"  → No labels found for {class_name}")
    
    return raw_labels


def seed_by_drug(client: FDALabelClient, loader: LocalLabelLoader, 
                  drug_name: str):
    """Download labels for a specific drug."""
    logger.info(f"Downloading {drug_name}...")
    raw_labels = client.search_labels(drug_name=drug_name, limit=3)
    
    if raw_labels:
        safe_name = drug_name.lower().replace(" ", "_").replace("-", "_")
        loader.save_labels(raw_labels, f"drug_{safe_name}.json")
        
        label = client.parse_label(raw_labels[0])
        sections = list(label.sections.keys())
        logger.info(f"  → {label.drug_name} ({label.generic_name}): {len(sections)} sections")
    else:
        logger.warning(f"  → No labels found for {drug_name}")
    
    return raw_labels


def seed_all(max_per_class: int = 15):
    """Download all default drug classes and priority drugs."""
    client = FDALabelClient()
    loader = LocalLabelLoader()
    
    total_labels = 0
    total_drugs = set()
    
    print("=" * 60)
    print("FDA DRUG LABEL DATA SEEDER")
    print("=" * 60)
    print()
    
    # Seed by class
    print("Phase 1: Downloading by therapeutic class...")
    print("-" * 40)
    for class_name, config in DEFAULT_DRUG_CLASSES.items():
        try:
            labels = seed_by_class(client, loader, class_name, config, max_per_class)
            total_labels += len(labels)
            for raw in labels:
                parsed = client.parse_label(raw)
                total_drugs.add(parsed.drug_name)
        except Exception as e:
            logger.error(f"Failed to seed {class_name}: {e}")
    
    print()
    print("Phase 2: Downloading priority individual drugs...")
    print("-" * 40)
    for drug_name in PRIORITY_DRUGS:
        try:
            labels = seed_by_drug(client, loader, drug_name)
            total_labels += len(labels)
            for raw in labels:
                parsed = client.parse_label(raw)
                total_drugs.add(parsed.drug_name)
        except Exception as e:
            logger.error(f"Failed to seed {drug_name}: {e}")
    
    client.close()
    
    print()
    print("=" * 60)
    print(f"SEEDING COMPLETE")
    print(f"  Total labels downloaded: {total_labels}")
    print(f"  Unique drugs: {len(total_drugs)}")
    print(f"  Data directory: data/sample_labels/")
    print(f"  Files created: {len(list(loader.data_dir.glob('*.json')))}")
    print("=" * 60)
    print()
    print("Next step: Run the ingestion pipeline to index these labels:")
    print("  python scripts/ingest.py")


def list_classes():
    """Show available drug classes and their relevance."""
    print("\nAvailable Drug Classes for Seeding:")
    print("=" * 70)
    for name, config in DEFAULT_DRUG_CLASSES.items():
        print(f"\n  {name}")
        print(f"    Search term: {config['search_term']}")
        print(f"    Example drugs: {', '.join(config['example_drugs'])}")
        print(f"    Relevance: {config['relevance']}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FDA drug labels for development")
    parser.add_argument("--class", dest="drug_class", help="Download specific drug class")
    parser.add_argument("--drug", help="Download specific drug by name")
    parser.add_argument("--max-per-class", type=int, default=15, help="Max labels per class")
    parser.add_argument("--list-classes", action="store_true", help="List available classes")
    
    args = parser.parse_args()
    
    if args.list_classes:
        list_classes()
    elif args.drug_class:
        client = FDALabelClient()
        loader = LocalLabelLoader()
        if args.drug_class in DEFAULT_DRUG_CLASSES:
            seed_by_class(client, loader, args.drug_class, 
                         DEFAULT_DRUG_CLASSES[args.drug_class], args.max_per_class)
        else:
            # Try as a raw search term
            seed_by_class(client, loader, args.drug_class,
                         {"search_term": args.drug_class}, args.max_per_class)
        client.close()
    elif args.drug:
        client = FDALabelClient()
        loader = LocalLabelLoader()
        seed_by_drug(client, loader, args.drug)
        client.close()
    else:
        seed_all(args.max_per_class)
