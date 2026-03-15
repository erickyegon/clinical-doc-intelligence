"""
Hybrid Search and Re-Ranking Pipeline
Combines dense vector search with metadata filtering and cross-encoder re-ranking.

Module 7: Retrieval, Ranking & Re-Ranking
Module 8: Advanced RAG
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

from config.settings import RETRIEVAL_TOP_K, RERANK_TOP_K

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A single retrieved document with score and metadata."""
    content: str
    metadata: dict
    score: float  # 0.0 (worst) to 1.0 (best)
    doc_id: str = ""
    rerank_score: Optional[float] = None

    @property
    def drug_name(self) -> str:
        return self.metadata.get("drug_name", "Unknown")

    @property
    def section_type(self) -> str:
        return self.metadata.get("section_type", "Unknown")

    @property
    def citation(self) -> str:
        return f"[{self.drug_name} | {self.metadata.get('section_display_name', self.section_type)} | {self.metadata.get('label_id', 'N/A')}]"


@dataclass
class RetrievalResult:
    """Complete result of a retrieval operation."""
    documents: list[RetrievedDocument] = field(default_factory=list)
    query: str = ""
    rewritten_query: Optional[str] = None
    retrieval_method: str = "hybrid"
    total_candidates: int = 0
    filters_applied: dict = field(default_factory=dict)

    @property
    def top_documents(self) -> list[RetrievedDocument]:
        """Return documents sorted by best score."""
        return sorted(self.documents, key=lambda d: d.rerank_score or d.score, reverse=True)


class HybridRetriever:
    """
    Multi-stage retrieval pipeline:
    1. Dense vector search (semantic similarity)
    2. Metadata filtering (drug name, section type, therapeutic area)
    3. Section priority boosting (safety sections ranked higher)
    4. MMR diversification (avoid redundant chunks)
    5. Cross-encoder re-ranking (precision refinement)
    """

    def __init__(
        self,
        vector_store,
        reranker=None,
        initial_k: int = RETRIEVAL_TOP_K,
        final_k: int = RERANK_TOP_K,
        mmr_lambda: float = 0.7,  # Balance relevance vs diversity
    ):
        self.vector_store = vector_store
        self.reranker = reranker
        self.initial_k = initial_k
        self.final_k = final_k
        self.mmr_lambda = mmr_lambda

    def retrieve(
        self,
        query: str,
        drug_name: Optional[str] = None,
        section_type: Optional[str] = None,
        therapeutic_area: Optional[str] = None,
        rewritten_query: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Execute the full retrieval pipeline.
        
        Args:
            query: Original user query
            drug_name: Filter to specific drug
            section_type: Filter to specific label section
            therapeutic_area: Filter by therapeutic area
            rewritten_query: Query-rewritten version for better retrieval
        """
        search_query = rewritten_query or query

        # Stage 1: Dense vector search with metadata filtering
        raw_results = self.vector_store.query_with_metadata_filter(
            query_text=search_query,
            drug_name=drug_name,
            section_type=section_type,
            therapeutic_area=therapeutic_area,
            n_results=self.initial_k,
        )

        # Convert to RetrievedDocument objects
        candidates = []
        for i, doc_text in enumerate(raw_results["documents"]):
            distance = raw_results["distances"][i] if raw_results["distances"] else 1.0
            # Convert cosine distance to similarity score
            similarity = 1.0 - distance

            candidates.append(RetrievedDocument(
                content=doc_text,
                metadata=raw_results["metadatas"][i] if raw_results["metadatas"] else {},
                score=similarity,
                doc_id=raw_results["ids"][i] if raw_results["ids"] else f"doc_{i}",
            ))

        # Stage 2: Section priority boosting
        candidates = self._boost_by_priority(candidates)

        # Stage 3: MMR diversification
        candidates = self._mmr_diversify(candidates, self.initial_k)

        # Stage 4: Cross-encoder re-ranking (if available)
        if self.reranker and candidates:
            candidates = self.reranker.rerank(query, candidates, top_k=self.final_k)
        else:
            candidates = candidates[:self.final_k]

        result = RetrievalResult(
            documents=candidates,
            query=query,
            rewritten_query=rewritten_query,
            total_candidates=len(raw_results["documents"]),
            filters_applied={
                "drug_name": drug_name,
                "section_type": section_type,
                "therapeutic_area": therapeutic_area,
            },
        )

        logger.info(
            f"Retrieved {len(candidates)} documents for query: '{query[:80]}...' "
            f"(from {result.total_candidates} candidates)"
        )
        return result

    def _boost_by_priority(
        self, candidates: list[RetrievedDocument], boost_factor: float = 0.1
    ) -> list[RetrievedDocument]:
        """
        Boost scores for safety-critical sections.
        This ensures Black Box Warnings and Contraindications surface first.
        """
        for doc in candidates:
            priority = doc.metadata.get("section_priority", 3)
            # Normalize priority to 0-0.1 range and add to score
            doc.score += (priority / 10.0) * boost_factor

            # Extra boost for safety flags
            safety_flags = doc.metadata.get("safety_flags", "")
            if "black_box" in safety_flags:
                doc.score += 0.05
        return candidates

    def _mmr_diversify(
        self, candidates: list[RetrievedDocument], k: int
    ) -> list[RetrievedDocument]:
        """
        Maximal Marginal Relevance: balance relevance with diversity.
        Prevents returning 5 chunks from the same drug section.
        """
        if len(candidates) <= k:
            return candidates

        selected = [candidates[0]]  # Start with highest scoring
        remaining = candidates[1:]

        while len(selected) < k and remaining:
            best_score = -1
            best_idx = 0

            for i, candidate in enumerate(remaining):
                # Relevance component
                relevance = candidate.score

                # Diversity component: penalize if same drug+section already selected
                max_similarity = 0
                for sel in selected:
                    sim = self._metadata_similarity(candidate.metadata, sel.metadata)
                    max_similarity = max(max_similarity, sim)

                # MMR score
                mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_similarity

                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _metadata_similarity(self, meta1: dict, meta2: dict) -> float:
        """
        Calculate metadata overlap between two documents.
        High overlap = same drug & section = less diverse.
        """
        score = 0.0
        if meta1.get("drug_name") == meta2.get("drug_name"):
            score += 0.5
        if meta1.get("section_type") == meta2.get("section_type"):
            score += 0.3
        if meta1.get("label_id") == meta2.get("label_id"):
            score += 0.2
        return score


class CrossEncoderReranker:
    """
    Cross-encoder re-ranking for precision refinement.
    
    In production, uses a fine-tuned cross-encoder model.
    Falls back to score-based ranking if model unavailable.
    
    Module 7: Re-Ranking
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Attempt to load cross-encoder model. Graceful fallback if unavailable."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded reranker model: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Re-ranking will use score-based fallback. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning(f"Failed to load reranker model: {e}. Using fallback.")

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = RERANK_TOP_K,
    ) -> list[RetrievedDocument]:
        """
        Re-rank documents using cross-encoder model.
        Falls back to original scores if model unavailable.
        """
        if not documents:
            return []

        if self.model is None:
            # Fallback: sort by original score
            sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
            return sorted_docs[:top_k]

        # Prepare pairs for cross-encoder
        pairs = [(query, doc.content) for doc in documents]

        try:
            scores = self.model.predict(pairs)
            for doc, score in zip(documents, scores):
                doc.rerank_score = float(score)

            reranked = sorted(documents, key=lambda d: d.rerank_score, reverse=True)
            return reranked[:top_k]
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}. Using original scores.")
            sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
            return sorted_docs[:top_k]
