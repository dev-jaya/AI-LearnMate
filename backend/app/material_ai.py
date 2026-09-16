from __future__ import annotations

from .ai_providers import validate_question
from .openai_responses import OpenAIResponsesProvider


async def generate_material_questions(
    material_text: str,
    subject: str,
    topic: str,
    difficulty: str,
    count: int,
    excluded_questions: list[str],
) -> list[dict] | None:
    """Generate MCQs grounded only in the learner's selected material."""
    provider = OpenAIResponsesProvider()
    if not provider.api_key:
        return None
    return await provider.generate_material_questions(
        material_text=material_text,
        subject=subject,
        topic=topic,
        difficulty=difficulty,
        count=count,
        excluded_questions=excluded_questions,
    )
