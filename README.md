# Clinical Document Intelligence Platform

[![CI — Test & Validate](https://github.com/erickyegon/clinical-doc-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/erickyegon/clinical-doc-intelligence/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**AI-powered FDA drug label intelligence system for regulatory analysis, formulary decision support, and competitive intelligence.**

Author: Erick Kiprotich Yegon, epidemiologist and data scientist (real-world evidence, HEOR, causal inference) · Portfolio: https://erickyegon.github.io · LinkedIn: https://linkedin.com/in/erickyegon

> **Status: research prototype, not a validated clinical tool.** There is no evaluation benchmark yet and no measured accuracy, faithfulness or citation results. The metrics in `src/evaluation/evaluator.py` are heuristic screens (word overlap, field-present and keyword checks), not clinical validation, and the "confidence" value is derived from retrieval scores. Answers depend on the label versions you download. Do not use this for patient care or regulatory decisions.

---

## The Problem

Pharmaceutical HEOR teams, formulary committees, and regulatory analysts spend substantial time manually reviewing FDA labels, comparing safety profiles across drug classes, and tracking label changes. This work is repetitive, error-prone, and doesn't scale.

## The Solution

A RAG (Retrieval-Augmented Generation) prototype that:

- **Ingests** FDA drug labels from the openFDA API (you choose the drugs to download with `scripts/seed_data.py`) and clinical trial data from ClinicalTrials.gov
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
│ • PHI      │  │  Query Rewrite               │  │ • Confidence      │
│   Detection│  │  ↓                           │  │   threshold       │
│ • Injection│  │  Hybrid Retrieval            │  │ • Citation        │
│   Defense  │  │  • Dense vector search       │  │   validation      │
│ • Schema   │  │  • Metadata filtering        │  │ • Unsupported     │
│   Validate │  │  • Section priority boost    │  │   claim detection │
│            │  │  • MMR diversification       │  │ • Clinical        │
│            │  │  • Cross-encoder re-rank     │  │   disclaimer      │
│            │  │  ↓                           │  │                   │
│            │  │  Context Assembly             │  │                   │
│            │  │  ↓                           │  │                   │
│            │  │  LLM Generation              │  │                   │
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
│  │              │  │ trials       │  │ Metadata: drug, section,   │ │
│  │              │  │              │  │ therapeutic area, safety    │ │
│  └──────────────┘  └──────────────┘  └────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                    EVALUATION                                         │
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

### 5. Guardrails
- **Input**: PHI detection (SSN, MRN, DOB patterns), prompt injection defense, query sanitization
- **Output**: Retrieval-score thresholds (not a calibrated answer confidence), citation validation, unsupported clinical claim detection, mandatory disclaimers

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| API | FastAPI | REST API with OpenAPI docs |
| Frontend | Streamlit | Interactive clinical intelligence dashboard |
| LLM | OpenAI GPT-4o / Groq Llama 3.1 / AWS Bedrock | Multi-provider with fallback |
| Vector DB | ChromaDB | Section-aware document storage |
| Embeddings | all-MiniLM-L6-v2 (local) / text-embedding-3-small | Cost-tiered embedding |
| Re-ranking | cross-encoder/ms-marco-MiniLM (optional) | Precision refinement |
| Prompts | Jinja2 + YAML | Prompt management |
| Guardrails | Pydantic + custom validators | Input/output safety |
| Evaluation | Custom heuristic screening metrics (no benchmark yet) | Keyword and word-overlap checks |
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

**Status: no benchmark and no measured results.** `src/evaluation/evaluator.py` holds scoring code and a 27-case sample set (`SAMPLE_EVAL_DATASET`). That set is not referenced by any script or test, its reference answers (called expert-written in a code comment) carry no recorded source, label version or reviewer, and its header comment (28 cases, a "temporal" category) does not match its contents. The scores the code can produce are heuristic screens:

| Metric (as named in the code) | What the code actually computes | Target (aspirational, never measured) |
|--------|-------------|--------|
| Retrieval precision | Share of cited drug names that match the expected drug names | > 0.85 |
| Faithfulness | Word overlap between the answer and the first 200 characters of each cited chunk | > 0.90 |
| Citation accuracy | Share of citations with a non-empty drug, section and label ID (does not check that the citation supports the claim) | > 0.95 |
| Safety completeness | Number of six warning keywords found in the answer, for safety questions | > 0.95 |
| Clinical accuracy | Share of required keywords present in the answer | not set |
| Latency (p95), cost per query | Measured per query when the evaluator is run | < 5 s, < $0.03 |

A protocol with a frozen held-out test set, a baseline system, human-verified scoring and a failure analysis has not been built.

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

## Related Projects

- **[Medicare RAF Pipeline](https://github.com/erickyegon/medicare-raf-prototypes)** — Risk adjustment factor prediction for Medicare Advantage (simulated claims)

---

## License

MIT
