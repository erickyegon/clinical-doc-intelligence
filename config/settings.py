"""
Central configuration for the Clinical Document Intelligence Platform.
Uses environment variables with sensible defaults for local development.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_LABELS_DIR = DATA_DIR / "sample_labels"
EVAL_DIR = DATA_DIR / "eval"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "vector_store"

# === LLM Provider Configuration (Module 3: Provider Switching) ===
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai | groq | bedrock
LLM_CONFIGS = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "base_url": "https://api.openai.com/v1",
        "max_tokens": 4096,
        "temperature": 0.1,
    },
    "groq": {
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "model": os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
        "base_url": "https://api.groq.com/openai/v1",
        "max_tokens": 4096,
        "temperature": 0.1,
    },
    "bedrock": {
        "region": os.getenv("AWS_REGION", "us-east-1"),
        "model": os.getenv("BEDROCK_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0"),
        "max_tokens": 4096,
        "temperature": 0.1,
    },
}

# === Embedding Configuration ===
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")  # local | openai
EMBEDDING_CONFIGS = {
    "local": {
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
    },
    "openai": {
        "model_name": "text-embedding-3-small",
        "dimension": 1536,
        "api_key": os.getenv("OPENAI_API_KEY", ""),
    },
}

# === Vector Store ===
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "fda_drug_labels")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

# === Chunking ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# === FDA Data ===
OPENFDA_BASE_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")
DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
CLINICALTRIALS_BASE_URL = "https://clinicaltrials.gov/api/v2"

# === Cache (Module 8: Performance Optimization) ===
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "false").lower() == "true"

# === Guardrails (Module 11) ===
MAX_QUERY_LENGTH = 2000
MIN_CONFIDENCE_THRESHOLD = 0.3
MAX_TOKENS_PER_QUERY = 8000
BLOCKED_PATTERNS = [
    r"(?i)patient\s+name",
    r"(?i)social\s+security",
    r"(?i)date\s+of\s+birth.*\d{2}",
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
]

# === Evaluation (Module 10) ===
EVAL_LLM_MODEL = "gpt-4o"
EVAL_SAMPLE_SIZE = 50

# === API ===
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))

# === Logging ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_TOKEN_TRACKING = os.getenv("ENABLE_TOKEN_TRACKING", "true").lower() == "true"
