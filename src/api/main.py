"""
Clinical Document Intelligence Platform - FastAPI Application
Production-grade REST API for FDA drug label intelligence.

Module 16: End-to-End Project with Deployment
"""
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import API_HOST, API_PORT, LOG_LEVEL
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.hybrid_search import HybridRetriever, CrossEncoderReranker
from src.generation.rag_chain import RAGChain, PromptManager
from src.orchestration.model_router import ModelRouter, TokenTracker
from src.guardrails.validators import (
    InputGuardrails, OutputGuardrails,
    QueryRequest, ComparisonRequest,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# === Global components (initialized on startup) ===
vector_store: Optional[VectorStoreManager] = None
retriever: Optional[HybridRetriever] = None
rag_chain: Optional[RAGChain] = None
model_router: Optional[ModelRouter] = None
input_guardrails = InputGuardrails()
output_guardrails = OutputGuardrails()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: initialize and teardown components."""
    global vector_store, retriever, rag_chain, model_router

    logger.info("Initializing Clinical Document Intelligence Platform...")

    # Initialize vector store
    vector_store = VectorStoreManager()
    doc_count = vector_store.get_document_count()
    logger.info(f"Vector store ready: {doc_count} documents indexed")

    # Initialize retriever with optional reranker
    try:
        reranker = CrossEncoderReranker()
    except Exception:
        reranker = None
        logger.warning("Reranker not available, using score-based ranking")

    retriever = HybridRetriever(
        vector_store=vector_store,
        reranker=reranker,
    )

    # Initialize model router and RAG chain
    token_tracker = TokenTracker()
    model_router = ModelRouter(token_tracker=token_tracker)
    prompt_manager = PromptManager()
    rag_chain = RAGChain(
        retriever=retriever,
        model_router=model_router,
        prompt_manager=prompt_manager,
    )

    # Initialize multi-agent coordinator (Module 9)
    try:
        from src.agents.coordinator import MultiAgentCoordinator
        from src.agents.api_routes import set_coordinator
        from src.ingestion.fda_labels import FDALabelClient
        from src.ingestion.clinical_trials import ClinicalTrialsClient

        coordinator = MultiAgentCoordinator(
            retriever=retriever,
            model_router=model_router,
            fda_client=FDALabelClient(),
            trials_client=ClinicalTrialsClient(),
        )
        set_coordinator(coordinator)
        logger.info("Multi-agent coordinator initialized")
    except Exception as e:
        logger.warning(f"Agent coordinator not available: {e}")

    logger.info("Platform initialized successfully")
    yield

    # Cleanup
    if model_router:
        await model_router.close()
    logger.info("Platform shut down")


# === FastAPI App ===

app = FastAPI(
    title="Clinical Document Intelligence Platform",
    description=(
        "AI-powered FDA drug label intelligence system. "
        "Query, compare, and analyze drug labels with citation-grounded answers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register agent routes (Module 9)
from src.agents.api_routes import router as agent_router
app.include_router(agent_router)


# === Request/Response Schemas ===

class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    metadata: dict
    warning: Optional[str] = None
    disclaimer: str = ""


class ComparisonResponse(BaseModel):
    answer: str
    citations: list[dict]
    drugs_compared: list[str]
    metadata: dict


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    available_drugs: list[str]
    version: str


class StatsResponse(BaseModel):
    token_usage: dict
    documents_indexed: int
    available_drugs: int


# === API Endpoints ===

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health and index status."""
    drugs = vector_store.get_all_drug_names() if vector_store else []
    return HealthResponse(
        status="healthy",
        documents_indexed=vector_store.get_document_count() if vector_store else 0,
        available_drugs=drugs[:20],
        version="1.0.0",
    )


@app.post("/query", response_model=QueryResponse, tags=["Intelligence"])
async def query_documents(request: QueryRequest):
    """
    Query FDA drug labels with citation-grounded answers.
    
    Supports:
    - General clinical questions
    - Drug-specific queries (filter by drug_name)
    - Section-specific queries (filter by section_type)
    - Therapeutic area queries
    """
    # Input guardrails
    validation = input_guardrails.validate(request.query)
    if not validation.passed:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Query blocked by guardrails",
                "reason": validation.blocked_reason,
                "risk_level": validation.risk_level.value,
            },
        )

    query_text = validation.sanitized_input or request.query

    try:
        response = await rag_chain.query(
            user_query=query_text,
            drug_name=request.drug_name,
            section_type=request.section_type,
            therapeutic_area=request.therapeutic_area,
            enable_rewrite=request.enable_rewrite,
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

    # Output guardrails
    output_validation = output_guardrails.validate(response)
    warning = output_guardrails.add_confidence_warning(response)
    disclaimer = output_guardrails.CLINICAL_DISCLAIMER

    if output_validation.issues:
        logger.warning(f"Output guardrail issues: {output_validation.issues}")

    return QueryResponse(
        answer=response.answer,
        citations=[c.__dict__ for c in response.citations] if response.citations else [],
        metadata={
            "query": response.query,
            "rewritten_query": response.rewritten_query,
            "confidence": round(response.confidence, 3),
            "model_used": response.model_used,
            "total_tokens": response.total_tokens,
            "latency_ms": round(response.latency_ms, 1),
            "context_documents": response.context_documents,
            "guardrail_issues": output_validation.issues,
        },
        warning=warning,
        disclaimer=disclaimer,
    )


@app.post("/compare", response_model=ComparisonResponse, tags=["Intelligence"])
async def compare_drugs(request: ComparisonRequest):
    """
    Compare FDA drug labels across multiple products.
    
    Use cases:
    - Compare black box warnings across a drug class
    - Compare contraindication profiles
    - Side-by-side dosing comparison
    """
    validation = input_guardrails.validate(request.query)
    if not validation.passed:
        raise HTTPException(status_code=400, detail=validation.blocked_reason)

    try:
        response = await rag_chain.compare_documents(
            query=validation.sanitized_input or request.query,
            drug_names=request.drug_names,
        )
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

    return ComparisonResponse(
        answer=response.answer,
        citations=[c.__dict__ for c in response.citations],
        drugs_compared=request.drug_names,
        metadata={
            "confidence": round(response.confidence, 3),
            "model_used": response.model_used,
            "latency_ms": round(response.latency_ms, 1),
        },
    )


@app.get("/drugs", tags=["Data"])
async def list_drugs():
    """List all drugs available in the knowledge base."""
    drugs = vector_store.get_all_drug_names() if vector_store else []
    return {"drugs": drugs, "total": len(drugs)}


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get system statistics including token usage and costs."""
    return StatsResponse(
        token_usage=model_router.get_cost_summary() if model_router else {},
        documents_indexed=vector_store.get_document_count() if vector_store else 0,
        available_drugs=len(vector_store.get_all_drug_names()) if vector_store else 0,
    )


# === Run ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level=LOG_LEVEL.lower(),
    )
