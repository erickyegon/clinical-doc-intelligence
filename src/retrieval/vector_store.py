"""
Vector Store Manager
Handles document storage, retrieval, and management using ChromaDB.
Supports both local and server-mode deployments.

Module 7: Embeddings & Vector Databases
"""
import logging
from typing import Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings

from config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages the vector store lifecycle: create, add, query, delete.
    Uses ChromaDB with persistent storage for production use.
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_dir = str(persist_dir or CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or CHROMA_COLLECTION

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity for text
        )

        logger.info(
            f"Vector store initialized: collection='{self.collection_name}', "
            f"documents={self.collection.count()}"
        )

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> int:
        """
        Add documents to the vector store.
        ChromaDB handles embedding generation internally.
        
        Returns number of documents added.
        """
        if not texts:
            return 0

        # ChromaDB metadata values must be str, int, float, or bool
        clean_metadatas = [self._clean_metadata(m) for m in metadatas]

        try:
            self.collection.upsert(
                documents=texts,
                metadatas=clean_metadatas,
                ids=ids,
            )
            logger.info(f"Added/updated {len(texts)} documents to vector store")
            return len(texts)
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def query(
        self,
        query_text: str,
        n_results: int = RETRIEVAL_TOP_K,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> dict:
        """
        Query the vector store with optional metadata filtering.
        
        Args:
            query_text: The search query
            n_results: Number of results to return
            where: Metadata filter (e.g., {"drug_name": "Jardiance"})
            where_document: Document content filter
            
        Returns:
            Dict with keys: ids, documents, metadatas, distances
        """
        kwargs = {
            "query_texts": [query_text],
            "n_results": min(n_results, self.collection.count() or 1),
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        try:
            results = self.collection.query(**kwargs)
            # Flatten the nested lists (ChromaDB returns list of lists)
            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
            }
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

    def query_with_metadata_filter(
        self,
        query_text: str,
        drug_name: Optional[str] = None,
        section_type: Optional[str] = None,
        therapeutic_area: Optional[str] = None,
        n_results: int = RETRIEVAL_TOP_K,
    ) -> dict:
        """
        Convenience method for common filtered queries.
        Builds ChromaDB where clause from named parameters.
        """
        conditions = []
        if drug_name:
            conditions.append({"drug_name": {"$eq": drug_name}})
        if section_type:
            conditions.append({"section_type": {"$eq": section_type}})
        if therapeutic_area:
            conditions.append({"normalized_therapeutic_area": {"$eq": therapeutic_area}})

        where = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        return self.query(query_text=query_text, n_results=n_results, where=where)

    def get_all_drug_names(self) -> list[str]:
        """Get list of all unique drug names in the store."""
        try:
            # Get a sample of all metadata
            results = self.collection.get(limit=10000, include=["metadatas"])
            drug_names = set()
            for meta in results.get("metadatas", []):
                if meta and "drug_name" in meta:
                    drug_names.add(meta["drug_name"])
            return sorted(drug_names)
        except Exception as e:
            logger.error(f"Failed to get drug names: {e}")
            return []

    def get_document_count(self) -> int:
        """Get total number of documents in the collection."""
        return self.collection.count()

    def delete_by_drug(self, drug_name: str) -> int:
        """Delete all chunks for a specific drug."""
        try:
            results = self.collection.get(
                where={"drug_name": {"$eq": drug_name}},
                include=[],
            )
            ids = results.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} chunks for {drug_name}")
            return len(ids)
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return 0

    def reset(self):
        """Delete and recreate the collection (use with caution)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Collection '{self.collection_name}' has been reset")

    def _clean_metadata(self, metadata: dict) -> dict:
        """Ensure all metadata values are ChromaDB-compatible types."""
        clean = {}
        for k, v in metadata.items():
            if v is None:
                clean[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, list):
                clean[k] = ", ".join(str(item) for item in v)
            else:
                clean[k] = str(v)
        return clean
