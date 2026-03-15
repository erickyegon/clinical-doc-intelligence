# ADR-001: Vector Store Selection — ChromaDB

## Status: Accepted

## Date: 2026-03-15

## Context
The Clinical Document Intelligence Platform requires a vector database to store 
and retrieve embedded FDA drug label chunks. We evaluated three options:

| Criteria | ChromaDB | Pinecone | Qdrant |
|----------|----------|----------|--------|
| Local development | ✅ Embedded | ❌ Cloud only | ✅ Docker |
| Metadata filtering | ✅ Built-in | ✅ Built-in | ✅ Built-in |
| Cost | Free | $70+/mo | Free (self-hosted) |
| Production readiness | Moderate | High | High |
| Setup complexity | Minimal | Minimal | Moderate |
| Max documents | ~1M local | Billions | Billions |

## Decision
**ChromaDB** for development and initial deployment, with a migration path to 
**Qdrant** or **Pinecone** for production scale.

## Rationale
1. **Zero-config local development**: ChromaDB runs embedded in the Python process 
   with persistent storage. No Docker containers, no API keys, no cloud accounts needed 
   to start building and testing.

2. **Sufficient scale for our use case**: FDA drug labels number ~70,000 total. With 
   an average of 8-12 chunks per label, we're looking at ~500K-800K documents — well 
   within ChromaDB's local capabilities.

3. **Same query interface**: The retrieval layer uses an abstraction (`VectorStoreManager`) 
   that can be swapped to Pinecone or Qdrant by changing one class. Metadata filtering 
   syntax is similar across all three.

4. **Cost efficiency during development**: No cloud costs during the build phase. 
   Pinecone's free tier limits to 100K vectors, which may be insufficient for full 
   FDA label coverage.

## Migration Path
When scaling to production on AWS:
- **Option A**: Qdrant on ECS (self-hosted, cost-controlled)
- **Option B**: Pinecone serverless (managed, pay-per-query)
- **Option C**: Amazon OpenSearch with k-NN (AWS-native, integrates with Bedrock)

The `VectorStoreManager` abstraction means migration requires changing only the 
store initialization and query translation — retrieval and generation layers are unaffected.

## Consequences
- Local storage means vector data is not shared across machines without explicit export
- ChromaDB's cosine similarity implementation is Python-native; may be slower than 
  Qdrant's Rust-based engine at scale
- No built-in backup/restore — must implement via filesystem snapshots
