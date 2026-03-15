# ADR-002: LLM Provider Strategy — Multi-Provider with Cost-Tiered Routing

## Status: Accepted

## Date: 2026-03-15

## Context
The platform makes LLM calls for three distinct tasks with different quality/cost requirements:

1. **Query rewriting**: Low-stakes text transformation (expand abbreviations, add synonyms)
2. **Answer generation**: High-stakes clinical reasoning requiring accuracy and safety
3. **Evaluation/confidence scoring**: Moderate-stakes quality assessment

Each task has a different tolerance for error and a different cost sensitivity.

## Decision
Implement a **multi-provider model router** with automatic fallback:

| Task | Primary Model | Fallback | Rationale |
|------|--------------|----------|-----------|
| Query rewriting | Groq (Llama 3.1 8B) | OpenAI GPT-4o-mini | Low-stakes, speed matters |
| Answer generation | OpenAI GPT-4o | Groq (Llama 3.1 70B) | Accuracy critical for clinical content |
| Evaluation | OpenAI GPT-4o | — | Judge model should be strongest available |

## Rationale
1. **Cost optimization**: Query rewriting at $0.05/M tokens (Groq 8B) vs $2.50/M (GPT-4o) 
   saves ~98% on a high-volume operation with minimal quality impact.

2. **Availability**: Groq has aggressive rate limits; OpenAI has occasional outages. 
   Automatic fallback ensures the platform stays operational.

3. **Enterprise readiness**: The same router supports AWS Bedrock (Claude) for 
   organizations that require VPC-isolated inference — a key selling point for 
   pharma/healthcare clients with compliance requirements.

4. **Token tracking**: Every call is logged with model, token count, and estimated cost. 
   This data feeds the evaluation dashboard and enables cost-per-query reporting.

## Cost Projection
For a typical deployment processing 1,000 queries/day:

| Component | Model | Tokens/query | Daily cost |
|-----------|-------|-------------|------------|
| Query rewrite | Llama 8B (Groq) | ~300 | $0.02 |
| Retrieval (embedding) | Local MiniLM | 0 | $0.00 |
| Answer generation | GPT-4o | ~2,000 | $25.00 |
| **Total** | | | **~$25/day** |

## Consequences
- Requires managing multiple API keys and rate limits
- Response quality varies between providers — evaluation framework must account for this
- Groq's free tier may be insufficient for production volume; paid plan needed at scale
