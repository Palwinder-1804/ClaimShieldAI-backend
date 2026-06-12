import hashlib
import logging
from typing import Optional
from fastapi import HTTPException, status
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.services.session_service import redis_client

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # Initialize LangChain OpenAI models if API key is present
        self.api_key = settings.OPENAI_API_KEY
        if self.api_key and "your-openai-api-key" not in self.api_key:
            self.model_primary = ChatOpenAI(
                model="gpt-4o",
                openai_api_key=self.api_key,
                max_tokens=1000,
                temperature=0.2
            )
            self.model_fallback = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=self.api_key,
                max_tokens=1000,
                temperature=0.2
            )
        else:
            self.model_primary = None
            self.model_fallback = None

    async def call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Executes a prompt against OpenAI LLM.
        Includes caching in Redis, and fallback: GPT-4o -> GPT-4o-mini -> Redis Backup Cache -> HTTPException.
        """
        # Calculate prompt hash
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_key = f"llm_cache:{prompt_hash}"
        backup_key = f"llm_cache_backup:{prompt_hash}"

        # 1. Try checking primary active cache in Redis
        try:
            cached_val = await redis_client.get(cache_key)
            if cached_val:
                logger.info(f"LLM cache HIT for hash: {prompt_hash}")
                return cached_val
        except Exception as e:
            logger.warning(f"Redis cache fetch error: {e}")

        # If cache miss or Redis failure, attempt live models
        if not self.model_primary or not self.model_fallback:
            # Fallback to local stub if no OpenAI API Key configured
            logger.warning("OPENAI_API_KEY not set. Using local mock response fallback.")
            return await self._mock_fallback_response(prompt, prompt_hash)

        # 2. Try GPT-4o (Primary Model)
        try:
            logger.info("Attempting GPT-4o call...")
            response = await self.model_primary.ainvoke(prompt)
            result_text = str(response.content)
            await self._save_to_cache(cache_key, backup_key, result_text)
            return result_text
        except Exception as e_primary:
            logger.error(f"GPT-4o failed: {e_primary}. Falling back to GPT-4o-mini...")

            # 3. Try GPT-4o-mini (Fallback Model)
            try:
                response = await self.model_fallback.ainvoke(prompt)
                result_text = str(response.content)
                await self._save_to_cache(cache_key, backup_key, result_text)
                return result_text
            except Exception as e_fallback:
                logger.error(f"GPT-4o-mini failed: {e_fallback}. Falling back to stale backup cache...")

                # 4. Try stale backup cache lookup
                try:
                    backup_val = await redis_client.get(backup_key)
                    if backup_val:
                        logger.info("Stale backup cache successfully retrieved.")
                        return backup_val
                except Exception as e_backup:
                    logger.error(f"Redis backup retrieval failed: {e_backup}")

                # 5. Fail completely if no fallback works
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="LLM service calls failed. GPT-4o, GPT-4o-mini, and cache fallbacks were unreachable."
                )

    async def _save_to_cache(self, cache_key: str, backup_key: str, value: str) -> None:
        """
        Saves values into Redis: cache_key with 1-hour TTL, and backup_key with no TTL.
        """
        try:
            await redis_client.setex(cache_key, 3600, value)
            await redis_client.set(backup_key, value)
        except Exception as e:
            logger.warning(f"Failed to write to Redis cache: {e}")

    async def _mock_fallback_response(self, prompt: str, prompt_hash: str) -> str:
        """
        Generates structured mock responses for development testing when OpenAI keys are missing.
        """
        prompt_lower = prompt.lower()
        if "is_covered" in prompt_lower:
            return (
                "{\n"
                "  \"is_covered\": true,\n"
                "  \"coverage_evaluation\": \"Claim documents matches local health policy guidelines. Treatments are standard and reasonable.\",\n"
                "  \"matching_clauses_summary\": \"Coverage Section A - Standard medical treatments, matching treatment constraints.\",\n"
                "  \"discrepancies\": [],\n"
                "  \"deductible_applied\": 100.0,\n"
                "  \"maximum_payout\": 1500.0\n"
                "}"
            )
        elif "report" in prompt_lower or "executive summary" in prompt_lower:
            return (
                f"# Claim Audit Report (Mock)\n\n"
                f"**Claim ID Hash:** {prompt_hash[:10]}\n"
                f"**Coverage Assessment:** Claims validated. Standard treatment coverage confirmed.\n"
                f"**Fraud Assessment:** Combined anomaly scoring is low. No duplicates identified."
            )
        else:
            return "Mock Response: This is a placeholder insurance response because OPENAI_API_KEY is not configured."

llm_service = LLMService()
