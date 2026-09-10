"""Backward-compatible imports for the dynamic provider architecture."""
import asyncio

from .ai_providers import FallbackProvider, get_provider

def local_questions(topic, difficulty, count):
    return asyncio.run(FallbackProvider().generate_questions(topic, topic, "", difficulty, count, {}))

async def llm_questions(topic, difficulty, count):
    return await get_provider().generate_questions(topic, topic, "", difficulty, count, {})
