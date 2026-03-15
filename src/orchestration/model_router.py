"""
LLM Model Router with Provider Switching
Abstracts LLM calls across OpenAI, Groq, and AWS Bedrock.
Implements cost tracking, fallback logic, and rate limiting.

Module 3: Provider Switching & Abstraction Layer
Module 8: Caching & Performance Optimization
"""
import json
import time
import logging
from typing import Optional

import httpx

from config.settings import LLM_PROVIDER, LLM_CONFIGS, ENABLE_CACHE

logger = logging.getLogger(__name__)


class TokenTracker:
    """Track token usage and cost across providers."""

    # Approximate pricing per 1M tokens (as of early 2025)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 3.00, "output": 15.00},
    }

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.request_count = 0
        self.by_model = {}

    def track(self, model: str, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.request_count += 1

        pricing = self.PRICING.get(model, {"input": 1.0, "output": 2.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        self.total_cost += cost

        if model not in self.by_model:
            self.by_model[model] = {"input": 0, "output": 0, "cost": 0.0, "requests": 0}
        self.by_model[model]["input"] += input_tokens
        self.by_model[model]["output"] += output_tokens
        self.by_model[model]["cost"] += cost
        self.by_model[model]["requests"] += 1

    def get_summary(self) -> dict:
        return {
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "request_count": self.request_count,
            "by_model": self.by_model,
        }


class ModelRouter:
    """
    Routes LLM requests across providers with automatic fallback.
    
    Provider priority: primary → fallback → error
    All providers use OpenAI-compatible API format (except Bedrock).
    """

    FALLBACK_ORDER = ["openai", "groq"]

    def __init__(
        self,
        primary_provider: Optional[str] = None,
        token_tracker: Optional[TokenTracker] = None,
    ):
        self.primary_provider = primary_provider or LLM_PROVIDER
        self.token_tracker = token_tracker or TokenTracker()
        self.configs = LLM_CONFIGS
        self._clients = {}

    def _get_client(self, provider: str) -> httpx.AsyncClient:
        """Get or create an async HTTP client for a provider."""
        if provider not in self._clients:
            config = self.configs.get(provider, {})
            headers = {"Content-Type": "application/json"}

            if provider in ("openai", "groq"):
                headers["Authorization"] = f"Bearer {config.get('api_key', '')}"

            self._clients[provider] = httpx.AsyncClient(
                timeout=60.0,
                headers=headers,
            )
        return self._clients[provider]

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        provider: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> dict:
        """
        Generate a response using the specified or default provider.
        Automatically falls back to next provider on failure.
        
        Returns:
            dict with keys: content, model, total_tokens, input_tokens, output_tokens, provider
        """
        providers_to_try = [provider or self.primary_provider]
        for fallback in self.FALLBACK_ORDER:
            if fallback not in providers_to_try:
                providers_to_try.append(fallback)

        last_error = None
        for prov in providers_to_try:
            try:
                config = self.configs.get(prov)
                if not config:
                    continue

                if prov == "bedrock":
                    result = await self._call_bedrock(
                        config, system_prompt, user_prompt, max_tokens, temperature
                    )
                else:
                    result = await self._call_openai_compatible(
                        prov, config, system_prompt, user_prompt,
                        max_tokens, temperature, response_format
                    )

                # Track tokens
                self.token_tracker.track(
                    model=result["model"],
                    input_tokens=result.get("input_tokens", 0),
                    output_tokens=result.get("output_tokens", 0),
                )

                result["provider"] = prov
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Provider {prov} failed: {e}. Trying fallback...")
                continue

        error_msg = f"All LLM providers failed. Last error: {last_error}"
        logger.error(error_msg)
        return {
            "content": f"Error: Unable to generate response. {error_msg}",
            "model": "none",
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "provider": "none",
        }

    async def _call_openai_compatible(
        self, provider, config, system_prompt, user_prompt,
        max_tokens, temperature, response_format,
    ) -> dict:
        """Call OpenAI-compatible API (works for OpenAI, Groq, OpenRouter)."""
        client = await self._get_async_client(provider)

        payload = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens or config.get("max_tokens", 4096),
            "temperature": temperature if temperature is not None else config.get("temperature", 0.1),
        }
        if response_format:
            payload["response_format"] = response_format

        response = await client.post(
            f"{config['base_url']}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        usage = data.get("usage", {})
        return {
            "content": data["choices"][0]["message"]["content"],
            "model": data.get("model", config["model"]),
            "total_tokens": usage.get("total_tokens", 0),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }

    async def _call_bedrock(self, config, system_prompt, user_prompt, max_tokens, temperature) -> dict:
        """Call AWS Bedrock (Claude models). Requires boto3."""
        try:
            import boto3
            bedrock = boto3.client("bedrock-runtime", region_name=config.get("region", "us-east-1"))
        except ImportError:
            raise RuntimeError("boto3 required for Bedrock. Install with: pip install boto3")

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or config.get("max_tokens", 4096),
            "temperature": temperature if temperature is not None else config.get("temperature", 0.1),
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        response = bedrock.invoke_model(
            modelId=config["model"],
            body=json.dumps(payload),
        )
        data = json.loads(response["body"].read())

        return {
            "content": data["content"][0]["text"],
            "model": config["model"],
            "total_tokens": data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0),
            "input_tokens": data.get("usage", {}).get("input_tokens", 0),
            "output_tokens": data.get("usage", {}).get("output_tokens", 0),
        }

    async def _get_async_client(self, provider: str) -> httpx.AsyncClient:
        """Get or create async client."""
        if provider not in self._clients:
            config = self.configs.get(provider, {})
            headers = {"Content-Type": "application/json"}
            if provider in ("openai", "groq"):
                headers["Authorization"] = f"Bearer {config.get('api_key', '')}"
            self._clients[provider] = httpx.AsyncClient(timeout=60.0, headers=headers)
        return self._clients[provider]

    def get_cost_summary(self) -> dict:
        return self.token_tracker.get_summary()

    async def close(self):
        for client in self._clients.values():
            await client.aclose()
