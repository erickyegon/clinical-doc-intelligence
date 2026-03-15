#!/usr/bin/env python3
"""
Ingestion Script
Indexes downloaded FDA drug labels from data/sample_labels/ into the vector store.

Usage:
    python scripts/ingest.py                    # Index all local labels
    python scripts/ingest.py --drug "Ozempic"   # Index + fetch a specific drug from API
    python scripts/ingest.py --reset            # Clear vector store and re-index
    python scripts/ingest.py --stats            # Show current index statistics
"""
import sys
import os
import json
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.vector_store import VectorStoreManager
from src.processing.chunker import SectionAwareChunker
from src.processing.metadata import MetadataExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def show_stats(vector_store: VectorStoreManager):
    """Display current vector store statistics."""
    doc_count = vector_store.get_document_count()
    drugs = vector_store.get_all_drug_names()
    
    print("\n" + "=" * 50)
    print("VECTOR STORE STATISTICS")
    print("=" * 50)
    print(f"  Total chunks indexed: {doc_count}")
    print(f"  Unique drugs: {len(drugs)}")
    if drugs:
        print(f"  Drugs: {', '.join(drugs[:20])}")
        if len(drugs) > 20:
            print(f"  ... and {len(drugs) - 20} more")
    print("=" * 50 + "\n")


def ingest_local(reset: bool = False):
    """Index all labels from local data directory."""
    vector_store = VectorStoreManager()
    
    if reset:
        logger.info("Resetting vector store...")
        vector_store.reset()
    
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        chunker=SectionAwareChunker(),
        metadata_extractor=MetadataExtractor(),
    )
    
    print("\n" + "=" * 50)
    print("INGESTION PIPELINE - LOCAL DATA")
    print("=" * 50 + "\n")
    
    stats = pipeline.ingest_from_local()
    
    print("\n" + "-" * 50)
    print("INGESTION RESULTS:")
    print(f"  Labels fetched:  {stats.labels_fetched}")
    print(f"  Labels parsed:   {stats.labels_parsed}")
    print(f"  Chunks created:  {stats.chunks_created}")
    print(f"  Chunks indexed:  {stats.chunks_indexed}")
    print(f"  Errors:          {stats.errors}")
    print("-" * 50 + "\n")
    
    show_stats(vector_store)
    pipeline.close()


def ingest_drug(drug_name: str):
    """Fetch and index a specific drug from the API."""
    vector_store = VectorStoreManager()
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        chunker=SectionAwareChunker(),
        metadata_extractor=MetadataExtractor(),
    )
    
    print(f"\nFetching and indexing: {drug_name}")
    stats = pipeline.ingest_single_drug(drug_name, include_trials=True)
    
    print(f"\nResults for {drug_name}:")
    print(f"  Labels: {stats.labels_fetched}, Chunks: {stats.chunks_indexed}, Errors: {stats.errors}")
    
    show_stats(vector_store)
    pipeline.close()


def test_query(vector_store: VectorStoreManager):
    """Run a quick test query to verify indexing."""
    print("\nTest Query: 'What are the contraindications?'")
    results = vector_store.query("What are the contraindications?", n_results=3)
    
    if results["documents"]:
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            print(f"\n  Result {i+1}: {meta.get('drug_name', 'Unknown')} - {meta.get('section_display_name', 'Unknown')}")
            print(f"  Preview: {doc[:150]}...")
    else:
        print("  No results found. Is the vector store populated?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index FDA drug labels into vector store")
    parser.add_argument("--drug", help="Fetch and index a specific drug from API")
    parser.add_argument("--reset", action="store_true", help="Clear vector store before indexing")
    parser.add_argument("--stats", action="store_true", help="Show index statistics only")
    parser.add_argument("--test", action="store_true", help="Run a test query after indexing")
    
    args = parser.parse_args()
    
    if args.stats:
        vs = VectorStoreManager()
        show_stats(vs)
    elif args.drug:
        ingest_drug(args.drug)
    else:
        ingest_local(reset=args.reset)
        if args.test:
            vs = VectorStoreManager()
            test_query(vs)
