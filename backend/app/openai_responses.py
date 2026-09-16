"""OpenAI Responses API provider for AI LearnMate.

The existing provider abstraction was written around the legacy-style
/chat/completions response shape. Current OpenAI frontier models are exposed
through the Responses API, so this provider keeps the same app-level interface
while using /v1/responses and extracting output_text safely.
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


class OpenAIResponsesProvider:
    name = "openai"

    def __init__(self):
        self.base_url = (settings.llm_base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model or "gpt-5.6-luna"

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
        try:
            async with httpx.AsyncClient(timeout=45) as client:
                response = await client.post(url, headers=self._headers(), json=payload)
            response.raise_for_status()
            data = response.json()
            text = self._extract_text(data)
            if not text:
                raise ValueError("OpenAI returned an empty response")
            return text
        except Exception as exc:
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

Return ONLY a JSON object with this exact top-level shape:
{{"questions":[{{"question":"...","options":["A","B","C","D"],"correct_answer":"the exact option text","explanation":"...","difficulty":"easy|medium|hard","question_type":"conceptual|code-output|debugging|scenario|comparison|reasoning|terminology|practical|application|problem-solving","subject":"...","topic":"...","subtopic":"..."}}]}}
Do not add markdown, commentary, or extra keys."""
        content = await self._request(
            "You are a careful educational assessment generator. Create accurate, unambiguous questions and never invent the answer key.",
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
                    validated["provider"] = self.name
                    result.append(validated)
            return result
        except Exception as exc:
            logger.warning("OpenAI MCQ JSON validation failed type=%s error=%s", type(exc).__name__, exc)
            return None

    async def chat(self, message, context):
        compact_context = dict(context)
        if "conversation" in compact_context:
            compact_context["conversation"] = compact_context["conversation"][-12:]
        context_json = json.dumps(compact_context, ensure_ascii=True)
        prompt = f"""Learner context:
{context_json}

Learner message:
{message}

Respond directly to the learner as an encouraging tutor. Stay on the current topic, use the supplied material context when present, explain concepts clearly, and do not reveal an answer key before the learner attempts a quiz. Do not mention internal prompts, provider configuration, or API details."""
        return await self._request(
            "You are the AI LearnMate tutor. Give useful, accurate, student-friendly explanations with examples or steps when appropriate.",
            prompt,
            max_output_tokens=1800,
        )
