"""
Document Ingestion Pipeline
Orchestrates the full ingestion flow: fetch → parse → chunk → embed → index.

Module 7: End-to-End RAG System Architecture
"""
import logging
import json
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

from src.ingestion.fda_labels import FDALabelClient, LocalLabelLoader, DrugLabel
from src.ingestion.clinical_trials import ClinicalTrialsClient
from src.processing.chunker import SectionAwareChunker
from src.processing.metadata import MetadataExtractor
from src.retrieval.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Track ingestion pipeline metrics."""
    labels_fetched: int = 0
    labels_parsed: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    errors: int = 0
    drug_classes_processed: list = None

    def __post_init__(self):
        if self.drug_classes_processed is None:
            self.drug_classes_processed = []

    def to_dict(self):
        return {
            "labels_fetched": self.labels_fetched,
            "labels_parsed": self.labels_parsed,
            "chunks_created": self.chunks_created,
            "chunks_indexed": self.chunks_indexed,
            "errors": self.errors,
            "drug_classes_processed": self.drug_classes_processed,
        }


class IngestionPipeline:
    """
    Full ingestion pipeline from FDA data sources to vector store.
    
    Supports two modes:
    1. Live API mode: Fetches from openFDA API in real-time
    2. Local mode: Loads pre-downloaded JSON files
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        chunker: Optional[SectionAwareChunker] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ):
        self.vector_store = vector_store
        self.chunker = chunker or SectionAwareChunker()
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.fda_client = FDALabelClient()
        self.trials_client = ClinicalTrialsClient()
        self.local_loader = LocalLabelLoader()
        self.stats = IngestionStats()

    def ingest_from_api(
        self,
        drug_classes: list[str],
        labels_per_class: int = 20,
        save_locally: bool = True,
    ) -> IngestionStats:
        """
        Ingest drug labels from the openFDA API for specified drug classes.
        
        Args:
            drug_classes: List of pharmacologic class names
                e.g. ["SGLT2 Inhibitor", "GLP-1 Receptor Agonist", "DPP-4 Inhibitor"]
            labels_per_class: Max labels to fetch per class
            save_locally: Whether to save fetched data for offline reuse
        """
        self.stats = IngestionStats()

        for drug_class in drug_classes:
            logger.info(f"Fetching labels for class: {drug_class}")
            try:
                raw_labels = self.fda_client.fetch_labels_by_drug_class(
                    drug_class, max_results=labels_per_class
                )
                self.stats.labels_fetched += len(raw_labels)
                self.stats.drug_classes_processed.append(drug_class)

                if save_locally and raw_labels:
                    safe_name = drug_class.lower().replace(" ", "_").replace("/", "_")
                    self.local_loader.save_labels(raw_labels, f"{safe_name}.json")

                self._process_raw_labels(raw_labels)

            except Exception as e:
                logger.error(f"Failed to ingest class {drug_class}: {e}")
                self.stats.errors += 1

        logger.info(f"Ingestion complete: {self.stats.to_dict()}")
        return self.stats

    def ingest_from_local(self, data_dir: Optional[Path] = None) -> IngestionStats:
        """Ingest from pre-downloaded local JSON files."""
        self.stats = IngestionStats()
        loader = LocalLabelLoader(data_dir) if data_dir else self.local_loader

        raw_labels = loader.load_all()
        self.stats.labels_fetched = len(raw_labels)
        self._process_raw_labels(raw_labels)

        logger.info(f"Local ingestion complete: {self.stats.to_dict()}")
        return self.stats

    def ingest_single_drug(self, drug_name: str, include_trials: bool = True) -> IngestionStats:
        """
        Ingest all available data for a single drug.
        Fetches FDA label + optionally associated clinical trials.
        """
        self.stats = IngestionStats()

        # Fetch FDA label
        raw_labels = self.fda_client.search_labels(drug_name=drug_name, limit=5)
        self.stats.labels_fetched = len(raw_labels)
        self._process_raw_labels(raw_labels)

        # Fetch associated clinical trials
        if include_trials:
            trials = self.trials_client.search_trials_for_drug(drug_name, max_results=10)
            for trial in trials:
                try:
                    chunks = self._chunk_trial(trial)
                    self.stats.chunks_created += len(chunks)
                    if chunks:
                        self.vector_store.add_documents(
                            texts=[c["content"] for c in chunks],
                            metadatas=[c["metadata"] for c in chunks],
                            ids=[c["id"] for c in chunks],
                        )
                        self.stats.chunks_indexed += len(chunks)
                except Exception as e:
                    logger.error(f"Failed to process trial {trial.nct_id}: {e}")
                    self.stats.errors += 1

        return self.stats

    def _process_raw_labels(self, raw_labels: list[dict]):
        """Parse, chunk, and index a batch of raw label records."""
        for raw in raw_labels:
            try:
                label = self.fda_client.parse_label(raw)
                self.stats.labels_parsed += 1

                chunks = self.chunker.chunk_label(label)
                self.stats.chunks_created += len(chunks)

                if chunks:
                    # Enrich metadata
                    for chunk in chunks:
                        chunk["metadata"] = self.metadata_extractor.enrich(
                            chunk["metadata"], label
                        )

                    self.vector_store.add_documents(
                        texts=[c["content"] for c in chunks],
                        metadatas=[c["metadata"] for c in chunks],
                        ids=[c["id"] for c in chunks],
                    )
                    self.stats.chunks_indexed += len(chunks)

            except Exception as e:
                logger.error(f"Failed to process label: {e}")
                self.stats.errors += 1

    def _chunk_trial(self, trial) -> list[dict]:
        """Create chunks from a clinical trial record."""
        chunks = []
        base_id = f"trial_{trial.nct_id}"
        base_meta = {
            "source_type": "clinical_trial",
            "nct_id": trial.nct_id,
            "drug_name": ", ".join(i["name"] for i in trial.interventions) if trial.interventions else "Unknown",
            "trial_status": trial.status,
            "trial_phase": trial.phase or "Unknown",
        }

        if trial.brief_summary:
            chunks.append({
                "id": f"{base_id}_summary",
                "content": f"Trial Summary ({trial.nct_id}): {trial.brief_summary}",
                "metadata": {**base_meta, "section_type": "trial_summary"},
            })

        if trial.eligibility_criteria:
            chunks.append({
                "id": f"{base_id}_eligibility",
                "content": f"Eligibility Criteria ({trial.nct_id}): {trial.eligibility_criteria}",
                "metadata": {**base_meta, "section_type": "eligibility_criteria"},
            })

        if trial.primary_outcomes:
            outcomes_text = "; ".join(
                f"{o['measure']} (timeframe: {o['timeframe']})"
                for o in trial.primary_outcomes
            )
            chunks.append({
                "id": f"{base_id}_outcomes",
                "content": f"Primary Outcomes ({trial.nct_id}): {outcomes_text}",
                "metadata": {**base_meta, "section_type": "primary_outcomes"},
            })

        return chunks

    def get_stats(self) -> dict:
        return self.stats.to_dict()

    def close(self):
        self.fda_client.close()
        self.trials_client.close()
