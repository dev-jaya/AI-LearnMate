"""Optional live Gemini integration test.

The test is skipped unless GEMINI_API_KEY is configured. It performs one real,
minimal Gemini request against the configured model and is intended for local or
deployment verification, not every CI run.
"""
import asyncio
import pytest

from app.ai_providers import SdkGeminiProvider
from app.config import settings


def test_live_gemini_chat():
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY is not configured")

    provider = SdkGeminiProvider()
    reply = asyncio.run(provider.chat(
        "Reply with exactly: 5",
        {"learner_name": "integration-test", "conversation": []},
    ))
    assert reply
    assert "5" in reply
