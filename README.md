# Clinical Document Intelligence Platform

[![CI — Test & Validate](https://github.com/erickyegon/clinical-doc-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/erickyegon/clinical-doc-intelligence/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**AI-powered FDA drug label intelligence system for regulatory analysis, formulary decision support, and competitive intelligence.**

> Built by [Erick K. Yegon, PhD](https://linkedin.com/in/erickyegon) — Director-level Data Scientist & Epidemiologist with 17+ years in global health analytics.

---

## The Problem

Pharmaceutical HEOR teams, formulary committees, and regulatory analysts spend **40+ hours per drug** manually reviewing FDA labels, comparing safety profiles across drug classes, and tracking label changes. This work is repetitive, error-prone, and doesn't scale.

## The Solution

A production-grade RAG (Retrieval-Augmented Generation) system that:

- **Ingests** 70,000+ FDA drug labels from the openFDA API and clinical trial data from ClinicalTrials.gov
- **Indexes** content with section-aware chunking that never splits safety-critical sections (Black Box Warnings, Contraindications)
- **Retrieves** relevant information using hybrid search: dense vector retrieval → metadata filtering → section priority boosting → MMR diversification → cross-encoder re-ranking
- **Generates** citation-grounded answers with source traceability back to specific label sections
- **Compares** drug labels across products for competitive intelligence and formulary decisions
- **Guards** against PHI exposure, prompt injection, unsupported clinical claims, and hallucination

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI REST API                              │
│              /query    /compare    /drugs    /health                 │
├────────────┬────────────────────────────────────┬───────────────────┤
│  INPUT     │        RAG PIPELINE                │    OUTPUT         │
│ GUARDRAILS │                                    │   GUARDRAILS      │
│            │  ┌──────────────────────────────┐  │                   │
│ • PHI      │  │  Query Rewrite (Module 6)    │  │ • Confidence      │
│   Detection│  │  ↓                           │  │   threshold       │
│ • Injection│  │  Hybrid Retrieval (Module 7) │  │ • Citation        │
│   Defense  │  │  • Dense vector search       │  │   validation      │
│ • Schema   │  │  • Metadata filtering        │  │ • Unsupported     │
│   Validate │  │  • Section priority boost    │  │   claim detection │
│            │  │  • MMR diversification       │  │ • Clinical        │
│            │  │  • Cross-encoder re-rank     │  │   disclaimer      │
│            │  │  ↓                           │  │                   │
│            │  │  Context Assembly (Module 8)  │  │                   │
│            │  │  ↓                           │  │                   │
│            │  │  LLM Generation (Module 3)   │  │                   │
│            │  │  • Provider switching         │  │                   │
│            │  │  • Cost tracking             │  │                   │
│            │  │  • Automatic fallback        │  │                   │
│            │  │  ↓                           │  │                   │
│            │  │  Citation Extraction         │  │                   │
│            │  └──────────────────────────────┘  │                   │
├────────────┴────────────────────────────────────┴───────────────────┤
│                     DATA LAYER                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐ │
│  │ openFDA API  │  │ ClinicalTrial│  │ ChromaDB Vector Store      │ │
│  │ 70K+ labels  │  │ .gov API V2  │  │ Section-aware chunks       │ │
│  │              │  │ 400K+ trials │  │ Metadata: drug, section,   │ │
│  │              │  │              │  │ therapeutic area, safety    │ │
│  └──────────────┘  └──────────────┘  └────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                    EVALUATION (Module 10)                             │
│  Faithfulness | Relevance | Citation Accuracy | Safety Completeness  │
│  Latency | Token Cost | Error Rate                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Clinical Q&A with Citations
Ask questions about any FDA-approved drug and receive answers grounded in official label data with section-level citations.

```
Query: "What are the contraindications for empagliflozin?"

Answer: Empagliflozin (JARDIANCE) is contraindicated in patients with severe 
renal impairment (eGFR < 30 mL/min/1.73 m²), end-stage renal disease, or 
patients on dialysis. It is also contraindicated in patients with known 
hypersensitivity to empagliflozin or any excipients.
[Source: JARDIANCE | Contraindications | label_abc123]
```

### 2. Cross-Drug Comparison
Compare safety profiles, dosing, and contraindications across drugs in a class.

```
Query: "Compare black box warnings for SGLT2 inhibitors"
Drugs: ["JARDIANCE", "FARXIGA", "INVOKANA"]
```

### 3. Safety-First Retrieval
Black Box Warnings and Contraindications are **never split** during chunking and receive priority boosting during retrieval, ensuring safety-critical information always surfaces first.

### 4. Multi-Provider LLM Routing
Automatic provider switching across OpenAI, Groq, and AWS Bedrock with per-query cost tracking and fallback logic.

### 5. Production Guardrails
- **Input**: PHI detection (SSN, MRN, DOB patterns), prompt injection defense, query sanitization
- **Output**: Confidence thresholds, citation validation, unsupported clinical claim detection, mandatory disclaimers

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| API | FastAPI | Production REST API with OpenAPI docs |
| Frontend | Streamlit | Interactive clinical intelligence dashboard |
| LLM | OpenAI GPT-4o / Groq Llama 3.1 / AWS Bedrock | Multi-provider with fallback |
| Vector DB | ChromaDB | Section-aware document storage |
| Embeddings | all-MiniLM-L6-v2 (local) / text-embedding-3-small | Cost-tiered embedding |
| Re-ranking | cross-encoder/ms-marco-MiniLM (optional) | Precision refinement |
| Prompts | Jinja2 + YAML | Production-grade prompt management |
| Guardrails | Pydantic + custom validators | Input/output safety |
| Evaluation | Custom RAGAS-inspired metrics | Domain-specific clinical eval |
| Data | openFDA API, ClinicalTrials.gov API V2 | Real U.S. regulatory data |
| Cache | Redis (optional) | Response and embedding caching |
| Deployment | Docker + AWS ECS/Fargate | Cloud-native containerization |

---

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/erickyegon/clinical-doc-intelligence.git
cd clinical-doc-intelligence

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys (Groq free tier works for development)
```

### 2. Download FDA Drug Labels

```bash
# Download labels for key therapeutic classes (diabetes, cardiovascular, oncology)
python scripts/seed_data.py

# Or download a specific drug
python scripts/seed_data.py --drug "Ozempic"
```

### 3. Index into Vector Store

```bash
python scripts/ingest.py

# Verify indexing
python scripts/ingest.py --stats
```

### 4. Start the API

```bash
uvicorn src.api.main:app --reload

# API docs at http://localhost:8000/docs
```

### 5. Launch the Interactive Frontend

```bash
# In a second terminal (keep the API running)
streamlit run app.py

# Opens at http://localhost:8501
```

### 5. Query

```bash
# Clinical Q&A
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the contraindications for Jardiance?"}'

# Drug Comparison
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare adverse reactions",
    "drug_names": ["JARDIANCE", "FARXIGA"]
  }'
```

### Docker

```bash
docker-compose up        # Development with hot reload
docker-compose --profile production up  # Production build
```

---

## Testing

```bash
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Evaluation

The platform includes a built-in evaluation framework with domain-specific metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| Retrieval Precision | Correct drugs in retrieved docs | > 0.85 |
| Faithfulness | Answer grounded in retrieved context | > 0.90 |
| Citation Accuracy | All citations valid and traceable | > 0.95 |
| Safety Completeness | Safety info included when relevant | > 0.95 |
| Latency (p95) | End-to-end response time | < 5s |
| Cost per Query | Average LLM cost | < $0.03 |

---

## Project Structure

```
clinical-doc-intel/
├── app.py                       # Streamlit interactive frontend
├── .streamlit/config.toml       # Streamlit theme & config
├── .github/workflows/ci.yml     # GitHub Actions CI/CD pipeline
├── Makefile                     # One-command operations
├── config/
│   ├── settings.py              # Central configuration
│   └── prompts.yaml             # Jinja2 prompt templates
├── src/
│   ├── ingestion/
│   │   ├── fda_labels.py        # openFDA API client
│   │   ├── clinical_trials.py   # ClinicalTrials.gov V2 client
│   │   └── pipeline.py          # Ingestion orchestration
│   ├── processing/
│   │   ├── chunker.py           # Section-aware chunking
│   │   └── metadata.py          # Metadata enrichment
│   ├── retrieval/
│   │   ├── vector_store.py      # ChromaDB management
│   │   └── hybrid_search.py     # Multi-stage retrieval + re-ranking
│   ├── generation/
│   │   └── rag_chain.py         # RAG pipeline with citations
│   ├── orchestration/
│   │   └── model_router.py      # Multi-provider LLM routing
│   ├── guardrails/
│   │   └── validators.py        # Input/output safety validation
│   ├── evaluation/
│   │   └── evaluator.py         # RAG evaluation framework
│   └── api/
│       └── main.py              # FastAPI application
├── scripts/
│   ├── seed_data.py             # Download FDA label data
│   └── ingest.py                # Index data into vector store
├── tests/
│   └── test_platform.py         # Comprehensive test suite
├── docs/adr/
│   ├── 001-vector-store-selection.md
│   └── 002-llm-provider-strategy.md
├── deployment/
│   └── (AWS ECS task definitions, CI/CD configs)
├── Dockerfile                   # Multi-stage production build
├── docker-compose.yml           # Local dev + production profiles
├── requirements.txt
├── .env.example
└── README.md
```

---

## Course Module Coverage

This project implements concepts from across the Full Stack Generative AI BootCamp:

| Module | Topic | Implementation |
|--------|-------|---------------|
| 1 | Foundations of GenAI | Embeddings, vector space, similarity |
| 3 | API for Accessing LLMs | Multi-provider router (OpenAI/Groq/Bedrock) |
| 5 | LLM Hosting & API | FastAPI endpoint exposure |
| 6 | Prompt Engineering | YAML prompt library, Jinja2 templates, CoT |
| 7 | RAG Systems | Full pipeline: ingest → chunk → embed → retrieve → generate |
| 8 | Advanced RAG | Query rewriting, re-ranking, caching, multimodal |
| 10 | Evaluation | Faithfulness, relevance, safety metrics, cost tracking |
| 11 | Guardrails | PHI detection, injection defense, output validation |
| 12 | MCP | Standardized tool interfaces (extensible) |
| 16 | E2E Deployment | Docker, FastAPI, cloud-native architecture |

---

## Related Projects

- **[Medicare RAF Pipeline](https://github.com/erickyegon/medicare-raf-prototypes)** — Risk adjustment factor prediction for Medicare Advantage (ATT −$391/member)
- **RWE Evidence Synthesis Agent** — Multi-agent system for systematic literature review automation (coming soon)
- **HEDIS/STARS Quality Agent** — Automated care gap identification for managed care (coming soon)

---

## License

MIT

---

## Contact

**Erick K. Yegon, PhD**  
Director-Level Data Science & Epidemiology  
📧 keyegonaws@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/erickyegon) | [GitHub](https://github.com/erickyegon) | [ORCID](https://orcid.org/0000-0002-7055-4848)
