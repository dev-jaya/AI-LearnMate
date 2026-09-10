from __future__ import annotations

import json
import re

from .ai_providers import GeminiProvider, validate_question


SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "question": {"type": "STRING"},
                    "options": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "correct_answer": {"type": "STRING"},
                    "explanation": {"type": "STRING"},
                    "difficulty": {"type": "STRING"},
                    "question_type": {"type": "STRING"},
                    "subtopic": {"type": "STRING"},
                },
                "required": ["question", "options", "correct_answer", "explanation", "difficulty", "question_type", "subtopic"],
            },
        }
    },
    "required": ["questions"],
}


async def generate_material_questions(
    material_text: str,
    subject: str,
    topic: str,
    difficulty: str,
    count: int,
    excluded_questions: list[str],
) -> list[dict] | None:
    provider = GeminiProvider()
    if not provider.api_key:
        return None

    source = material_text[:12000]
    excluded = "\n".join(f"- {item}" for item in excluded_questions[-30:]) or "- none"
    prompt = f"""Generate exactly {count} new multiple-choice questions using ONLY the supplied learning material as the factual source.

Subject: {subject}
Topic: {topic}
Difficulty: {difficulty}

Rules:
- Every question must be answerable from the material.
- Do not invent facts that are not supported by the material.
- Use varied conceptual, application, reasoning, terminology, scenario and code/data interpretation questions when the material supports them.
- Exactly four options per question.
- correct_answer must exactly match one option.
- Do not repeat or paraphrase any excluded question.
- Include a short explanation grounded in the material.

Previously used questions:
{excluded}

SOURCE MATERIAL:
{source}

Return only the requested JSON object."""
    try:
        content = await provider._generate(prompt, SCHEMA)
        data = json.loads(re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.I).strip())
        raw = data.get("questions", []) if isinstance(data, dict) else []
        result = []
        for item in raw:
            validated = validate_question(item, subject, topic, "Material-based", difficulty)
            if validated:
                validated["provider"] = "gemini"
                result.append(validated)
        return result
    except Exception:
        return None
