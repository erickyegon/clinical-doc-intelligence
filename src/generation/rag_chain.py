"""
RAG Generation Pipeline
Handles query rewriting, context assembly, LLM generation, and citation tracking.

Module 6: Prompt Engineering
Module 7: Prompting with Retrieved Context
Module 8: Context Engineering & Memory Management
"""
import json
import logging
import time
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from string import Template

from jinja2 import Environment, BaseLoader

from config.settings import PROJECT_ROOT
from src.retrieval.hybrid_search import RetrievalResult, RetrievedDocument

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Tracks source attribution for a generated answer."""
    drug_name: str
    section_type: str
    section_display_name: str
    label_id: str
    chunk_content_preview: str  # First 200 chars
    relevance_score: float


@dataclass
class RAGResponse:
    """Complete RAG response with citations and metadata."""
    answer: str
    citations: list[Citation] = field(default_factory=list)
    query: str = ""
    rewritten_query: Optional[str] = None
    confidence: float = 0.0
    model_used: str = ""
    total_tokens: int = 0
    latency_ms: float = 0.0
    context_documents: int = 0
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [
                {
                    "drug_name": c.drug_name,
                    "section": c.section_display_name,
                    "label_id": c.label_id,
                    "relevance_score": round(c.relevance_score, 3),
                }
                for c in self.citations
            ],
            "metadata": {
                "query": self.query,
                "rewritten_query": self.rewritten_query,
                "confidence": round(self.confidence, 3),
                "model_used": self.model_used,
                "total_tokens": self.total_tokens,
                "latency_ms": round(self.latency_ms, 1),
                "context_documents": self.context_documents,
            },
            "warning": self.warning,
        }


class PromptManager:
    """
    Manages prompt templates from YAML configuration.
    Module 6: Production-Grade Prompt Management
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or PROJECT_ROOT / "config" / "prompts.yaml"
        self.templates = {}
        self.jinja_env = Environment(loader=BaseLoader())
        self._load_templates()

    def _load_templates(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.templates = yaml.safe_load(f)
            logger.info(f"Loaded prompt templates from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load prompt templates: {e}")
            self.templates = {}

    def get_system_prompt(self, prompt_type: str) -> str:
        return self.templates.get("system_prompts", {}).get(prompt_type, "")

    def render_user_prompt(self, prompt_type: str, **kwargs) -> str:
        template_str = self.templates.get("user_prompts", {}).get(prompt_type, "")
        if not template_str:
            return ""
        try:
            template = self.jinja_env.from_string(template_str)
            return template.render(**kwargs)
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            return template_str


class RAGChain:
    """
    End-to-end RAG pipeline: query rewrite → retrieve → generate → cite.
    
    Module 7: End-to-End RAG System Architecture
    Module 8: Context Engineering
    """

    def __init__(self, retriever, model_router, prompt_manager: Optional[PromptManager] = None):
        self.retriever = retriever
        self.model_router = model_router
        self.prompt_manager = prompt_manager or PromptManager()

    async def query(
        self,
        user_query: str,
        drug_name: Optional[str] = None,
        section_type: Optional[str] = None,
        therapeutic_area: Optional[str] = None,
        enable_rewrite: bool = True,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline.
        
        1. Rewrite query for better retrieval
        2. Retrieve relevant documents
        3. Assemble context with citation markers
        4. Generate answer
        5. Extract and validate citations
        """
        start_time = time.time()

        # Step 1: Query rewriting (Module 6: Reasoning-Based Prompting)
        rewritten_query = None
        if enable_rewrite:
            rewritten_query = await self._rewrite_query(user_query)

        # Step 2: Retrieve documents
        retrieval_result = self.retriever.retrieve(
            query=user_query,
            drug_name=drug_name,
            section_type=section_type,
            therapeutic_area=therapeutic_area,
            rewritten_query=rewritten_query,
        )

        # Step 3: Check if we have sufficient context
        if not retrieval_result.documents:
            return RAGResponse(
                answer="I could not find relevant information in the FDA drug label database for your query. Please try rephrasing your question or specifying a drug name.",
                query=user_query,
                rewritten_query=rewritten_query,
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                warning="No relevant documents found",
            )

        # Step 4: Assemble context and generate
        context_docs = self._prepare_context_documents(retrieval_result.top_documents)
        
        system_prompt = self.prompt_manager.get_system_prompt("clinical_qa")
        user_prompt = self.prompt_manager.render_user_prompt(
            "clinical_qa",
            documents=context_docs,
            query=user_query,
        )

        # Step 5: Call LLM via model router
        llm_response = await self.model_router.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Step 6: Build citations
        citations = self._build_citations(retrieval_result.top_documents)

        # Step 7: Assemble response
        latency = (time.time() - start_time) * 1000

        response = RAGResponse(
            answer=llm_response.get("content", ""),
            citations=citations,
            query=user_query,
            rewritten_query=rewritten_query,
            confidence=self._estimate_confidence(retrieval_result),
            model_used=llm_response.get("model", ""),
            total_tokens=llm_response.get("total_tokens", 0),
            latency_ms=latency,
            context_documents=len(context_docs),
        )

        logger.info(
            f"RAG query completed: {len(citations)} citations, "
            f"confidence={response.confidence:.2f}, latency={latency:.0f}ms"
        )
        return response

    async def compare_documents(
        self,
        query: str,
        drug_names: list[str],
    ) -> RAGResponse:
        """
        Compare labels across multiple drugs.
        Module 8: Document Comparison Engine
        """
        start_time = time.time()

        # Retrieve for each drug separately
        drug_documents = {}
        all_docs = []
        for drug in drug_names:
            result = self.retriever.retrieve(query=query, drug_name=drug)
            drug_documents[drug] = result.top_documents
            all_docs.extend(result.top_documents)

        system_prompt = self.prompt_manager.get_system_prompt("document_comparison")
        user_prompt = self.prompt_manager.render_user_prompt(
            "document_comparison",
            drug_documents={
                name: self._prepare_context_documents(docs)
                for name, docs in drug_documents.items()
            },
            query=query,
        )

        llm_response = await self.model_router.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        citations = self._build_citations(all_docs)
        latency = (time.time() - start_time) * 1000

        return RAGResponse(
            answer=llm_response.get("content", ""),
            citations=citations,
            query=query,
            model_used=llm_response.get("model", ""),
            total_tokens=llm_response.get("total_tokens", 0),
            latency_ms=latency,
            context_documents=len(all_docs),
        )

    async def _rewrite_query(self, query: str) -> Optional[str]:
        """Rewrite user query for improved retrieval."""
        try:
            prompt = self.prompt_manager.render_user_prompt(
                "query_rewrite", query=query
            )
            if not prompt:
                return None

            response = await self.model_router.generate(
                system_prompt="You are a medical terminology expert. Rewrite queries to improve clinical document retrieval.",
                user_prompt=prompt,
                max_tokens=200,
                temperature=0.0,
            )
            rewritten = response.get("content", "").strip()
            if rewritten and rewritten != query:
                logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
        return None

    def _prepare_context_documents(self, documents: list[RetrievedDocument]) -> list[dict]:
        """Format retrieved documents for prompt injection."""
        return [
            {
                "content": doc.content,
                "metadata": doc.metadata,
                "citation": doc.citation,
            }
            for doc in documents
        ]

    def _build_citations(self, documents: list[RetrievedDocument]) -> list[Citation]:
        """Build citation objects from retrieved documents."""
        return [
            Citation(
                drug_name=doc.drug_name,
                section_type=doc.section_type,
                section_display_name=doc.metadata.get("section_display_name", doc.section_type),
                label_id=doc.metadata.get("label_id", ""),
                chunk_content_preview=doc.content[:200],
                relevance_score=doc.rerank_score or doc.score,
            )
            for doc in documents
        ]

    def _estimate_confidence(self, result: RetrievalResult) -> float:
        """
        Estimate answer confidence based on retrieval quality.
        Module 10: Evaluation Strategies
        """
        if not result.documents:
            return 0.0

        top_scores = [d.rerank_score or d.score for d in result.top_documents[:3]]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0

        # Factor in document diversity
        unique_drugs = len(set(d.drug_name for d in result.documents))
        unique_sections = len(set(d.section_type for d in result.documents))

        # Higher confidence if multiple relevant sections found
        coverage_bonus = min(0.1, unique_sections * 0.02)

        return min(1.0, avg_score + coverage_bonus)
