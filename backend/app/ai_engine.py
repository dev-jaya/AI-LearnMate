"""Backward-compatible helpers for the current Gemini provider architecture."""
import asyncio

from .ai_providers import get_provider


def local_questions(topic, difficulty, count):
    """Synchronous compatibility wrapper around the configured AI provider."""
    return asyncio.run(
        get_provider().generate_questions(topic, topic, "", difficulty, count, {})
    )


async def llm_questions(topic, difficulty, count):
    """Async compatibility wrapper around the configured AI provider."""
    return await get_provider().generate_questions(topic, topic, "", difficulty, count, {})
