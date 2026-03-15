# ============================================================
# Clinical Document Intelligence Platform — Makefile
# One-command operations for development, testing, and deployment
# ============================================================

.PHONY: help setup seed ingest serve ui test test-all lint clean docker docker-prod

# Default: show available commands
help: ## Show this help message
	@echo ""
	@echo "Clinical Document Intelligence Platform"
	@echo "========================================"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ============================================================
# Setup & Data
# ============================================================

setup: ## Install dependencies and prepare environment
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from template — edit with your API keys"; fi
	@mkdir -p data/sample_labels data/eval vector_store
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo "  1. Edit .env with your API keys (Groq free tier works)"
	@echo "  2. Run: make seed"
	@echo "  3. Run: make ingest"
	@echo "  4. Run: make serve"

seed: ## Download FDA drug labels from openFDA API
	python scripts/seed_data.py

seed-drug: ## Download a specific drug (usage: make seed-drug DRUG=Ozempic)
	python scripts/seed_data.py --drug "$(DRUG)"

ingest: ## Index downloaded labels into the vector store
	python scripts/ingest.py

ingest-reset: ## Clear vector store and re-index all labels
	python scripts/ingest.py --reset

ingest-stats: ## Show vector store statistics
	python scripts/ingest.py --stats

# ============================================================
# Run
# ============================================================

serve: ## Start the FastAPI backend (port 8000)
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

ui: ## Start the Streamlit frontend (port 8501)
	streamlit run app.py

run: ## Start both backend and frontend (requires two terminals)
	@echo "Terminal 1: make serve"
	@echo "Terminal 2: make ui"
	@echo ""
	@echo "Or use: make docker"

# ============================================================
# Testing & Quality
# ============================================================

test: ## Run unit tests (no network required)
	python -m pytest tests/ -v --tb=short \
		-k "not VectorStore and not HybridRetrieval and not Integration"

test-all: ## Run all tests including integration tests
	python -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	python -m pytest tests/ -v --tb=short \
		-k "not VectorStore and not HybridRetrieval and not Integration" \
		--cov=src --cov-report=term-missing

lint: ## Run linter (requires: pip install ruff)
	ruff check src/ tests/ scripts/ --select E,F,W --ignore E501

# ============================================================
# Docker
# ============================================================

docker: ## Build and run with docker-compose (development mode)
	docker-compose up --build

docker-prod: ## Build and run production deployment
	docker-compose --profile production up --build

docker-build: ## Build production Docker image only
	docker build --target production -t clinical-doc-intel:latest .

# ============================================================
# Utilities
# ============================================================

clean: ## Remove cached files and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .ruff_cache coverage.xml htmlcov/

clean-data: ## Remove downloaded labels and vector store (re-download required)
	rm -rf data/sample_labels/*.json
	rm -rf vector_store/
	@echo "Data cleaned. Run 'make seed' and 'make ingest' to repopulate."

status: ## Show system status (requires backend running)
	@echo "=== Health ===" && curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "Backend not running"
	@echo ""
	@echo "=== Stats ===" && curl -s http://localhost:8000/stats | python -m json.tool 2>/dev/null || echo "Backend not running"

query: ## Run a quick test query (requires backend running)
	@curl -s -X POST http://localhost:8000/query \
		-H "Content-Type: application/json" \
		-d '{"query": "What are the contraindications for empagliflozin?", "drug_name": "JARDIANCE"}' \
		| python -m json.tool
