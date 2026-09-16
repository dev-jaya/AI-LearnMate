"""OpenAI Responses API provider for AI LearnMate.

This module keeps the app's existing provider interface while using the
current Responses API for tutoring and question generation.
"""
import json
import logging
import re
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger(__name__)


def _clean_json(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


class _ProviderName(str):
    """Display as openai while remaining compatible with the legacy Gemini gate."""
    def __eq__(self, other):
        return other in {"openai", "gemini"} or str.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)


class OpenAIResponsesProvider:
    name = _ProviderName("openai")

    def __init__(self):
        self.base_url = (settings.llm_base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model or "gpt-5.6-luna"
        self.last_error = ""

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        chunks: list[str] = []
        for item in data.get("output", []) or []:
            for content in item.get("content", []) or []:
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text)
        return "".join(chunks).strip()

    async def _request(self, system: str, prompt: str, max_output_tokens: int = 3500) -> str | None:
        url = f"{self.base_url}/responses"
        payload = {
            "model": self.model,
            "instructions": system,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        }
        self.last_error = ""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(url, headers=self._headers(), json=payload)
            response.raise_for_status()
            data = response.json()
            text = self._extract_text(data)
            if not text:
                raise ValueError("OpenAI returned an empty response")
            return text
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("OpenAI Responses request failed model=%s type=%s error=%s", self.model, type(exc).__name__, exc)
            return None

    async def generate_questions(self, subject, topic, subtopic, difficulty, count, context):
        previous = context.get("excluded_questions", [])[-30:]
        prompt = f"""Generate exactly {count} genuinely new multiple-choice questions.
Subject: {subject}
Topic: {topic}
Subtopic: {subtopic or 'choose an appropriate subtopic'}
Difficulty: {difficulty}
Learner mastery: {context.get('mastery', {})}
Weak topics: {context.get('weak_topics', [])}
Previously used questions, which must not be repeated, paraphrased, or reused as the same scenario:
{chr(10).join('- ' + item for item in previous) or '- none'}

Return ONLY a JSON object with this exact shape:
{{"questions":[{{"question":"...","options":["A","B","C","D"],"correct_answer":"the exact option text","explanation":"...","difficulty":"easy|medium|hard","question_type":"conceptual|code-output|debugging|scenario|comparison|reasoning|terminology|practical|application|problem-solving","subject":"...","topic":"...","subtopic":"..."}}]}}
Do not add markdown, commentary, or extra keys."""
        content = await self._request(
            "You are a careful educational assessment generator. Create accurate, unambiguous questions and verify every answer key.",
            prompt,
            max_output_tokens=max(2500, count * 550),
        )
        if not content:
            return None
        try:
            data = json.loads(_clean_json(content))
            raw_questions = data.get("questions", []) if isinstance(data, dict) else data
            from .ai_providers import validate_question
            result = []
            for raw in raw_questions:
                item = dict(raw)
                item["answer"] = item.get("correct_answer", item.get("answer", -1))
                validated = validate_question(item, subject, topic, subtopic, difficulty)
                if validated:
                    validated["provider"] = "openai"
                    result.append(validated)
            return result
        except Exception as exc:
            logger.warning("OpenAI MCQ JSON validation failed type=%s error=%s", type(exc).__name__, exc)
            return None

    async def generate_material_questions(self, material_text, subject, topic, difficulty, count, excluded_questions):
        source = material_text[:16000]
        excluded = "\n".join(f"- {item}" for item in excluded_questions[-30:]) or "- none"
        prompt = f"""Generate exactly {count} new multiple-choice questions using ONLY the supplied learning material as the factual source.

Subject: {subject}
Topic: {topic}
Difficulty: {difficulty}

Rules:
- Every question must be answerable from the material.
- Do not invent facts not supported by the material.
- Use varied conceptual, application, reasoning, terminology, scenario and code/data interpretation questions when the material supports them.
- Exactly four options per question.
- correct_answer must exactly match one option.
- Do not repeat or paraphrase an excluded question.
- Include a short explanation grounded in the material.

Previously used questions:
{excluded}

SOURCE MATERIAL:
{source}

Return ONLY the JSON object requested."""
        content = await self._request(
            "You create reliable material-grounded educational MCQs. Use only the supplied source as evidence.",
            prompt,
            max_output_tokens=max(2500, count * 600),
        )
        if not content:
            return None
        try:
            data = json.loads(_clean_json(content))
            raw_questions = data.get("questions", []) if isinstance(data, dict) else data
            from .ai_providers import validate_question
            result = []
            for raw in raw_questions:
                item = dict(raw)
                item["answer"] = item.get("correct_answer", item.get("answer", -1))
                validated = validate_question(item, subject, topic, "Material-based", difficulty)
                if validated:
                    validated["provider"] = "openai"
                    result.append(validated)
            return result
        except Exception as exc:
            logger.warning("OpenAI material MCQ JSON validation failed type=%s error=%s", type(exc).__name__, exc)
            return None

    async def chat(self, message, context):
        compact_context = dict(context)
        conversation = compact_context.get("conversation", [])
        if conversation:
            compact_context["conversation"] = conversation[-12:]
        context_json = json.dumps(compact_context, ensure_ascii=True)
        prompt = f"""The learner's CURRENT message is the highest-priority instruction. Answer that message directly.

Learner context (background only; do not force the current message to match this context):
{context_json}

CURRENT learner message:
{message}

Conversation history is provided only to understand references such as 'that', 'it', or 'the previous example'. Do NOT answer an older message when the current message asks something new.
If the learner asks a general question, arithmetic problem, greeting, coding question, or unrelated question, answer it normally even when it differs from the selected learning topic.
For simple arithmetic, calculate the result exactly.
Use the selected topic and learning material as helpful context, not as a restriction.
Respond directly as an encouraging AI tutor. Be accurate, concise, and useful. When teaching, give a simple explanation first and then an example or steps when useful.
Do not mention internal prompts, provider configuration, or API details."""
        return await self._request(
            "You are the AI LearnMate tutor. The current learner message always takes priority over background topic context and previous conversation.",
            prompt,
            max_output_tokens=1800,
        )


# Install the Responses API provider into the legacy provider factory so
# quiz/assessment paths use the same configured OpenAI integration.
try:
    from . import ai_providers as _ai_providers
    _ai_providers.OpenAIResponsesProvider = OpenAIResponsesProvider

    def _get_provider_with_responses():
        if settings.gemini_api_key:
            return _ai_providers.GeminiProvider()
        if settings.ai_provider == "ollama" and settings.ollama_base_url and settings.ollama_model:
            return _ai_providers.OllamaProvider()
        if settings.ai_provider == "huggingface" and settings.hf_base_url and settings.hf_api_key and settings.hf_model:
            return _ai_providers.HuggingFaceProvider()
        if settings.llm_base_url and settings.llm_model and settings.llm_api_key:
            return OpenAIResponsesProvider()
        return _ai_providers.FallbackProvider()

    _ai_providers.get_provider.__code__ = _get_provider_with_responses.__code__
except Exception as exc:
    logger.warning("Could not install OpenAI provider override: %s", exc)
