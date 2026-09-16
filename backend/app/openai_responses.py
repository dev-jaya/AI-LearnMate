"""OpenAI Responses API provider for AI LearnMate."""
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


MCQ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
                    "correct_answer": {"type": "string"},
                    "explanation": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "question_type": {"type": "string"},
                    "subject": {"type": "string"},
                    "topic": {"type": "string"},
                    "subtopic": {"type": "string"},
                },
                "required": ["question", "options", "correct_answer", "explanation", "difficulty", "question_type", "subject", "topic", "subtopic"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


class OpenAIResponsesProvider:
    name = "openai"

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

    async def _request(self, system: str, prompt: Any, max_output_tokens: int = 3500, tools=None, json_schema: dict | None = None) -> str | None:
        url = f"{self.base_url}/responses"
        payload = {
            "model": self.model,
            "instructions": system,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        }
        if json_schema:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "learnmate_mcqs",
                    "description": "Validated multiple-choice questions for an educational assessment.",
                    "strict": True,
                    "schema": json_schema,
                }
            }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        self.last_error = ""
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                response = await client.post(url, headers=self._headers(), json=payload)
            # If a configured endpoint/model does not support Structured Outputs,
            # retry once as ordinary text so the material workflow still completes.
            if response.status_code >= 400 and json_schema:
                retry_payload = dict(payload)
                retry_payload.pop("text", None)
                async with httpx.AsyncClient(timeout=90) as client:
                    response = await client.post(url, headers=self._headers(), json=retry_payload)
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

Return a questions array. Every question must have exactly four options and correct_answer must exactly match one option."""
        content = await self._request(
            "You are a careful educational assessment generator. Create accurate, unambiguous questions and verify every answer key.",
            prompt,
            max_output_tokens=max(2500, count * 550),
            json_schema=MCQ_SCHEMA,
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
        source = material_text[:50000]
        excluded = "\n".join(f"- {item}" for item in excluded_questions[-30:]) or "- none"
        prompt = f"""Create exactly {count} new multiple-choice questions from the supplied learning material.

Subject: {subject}
Topic: {topic}
Difficulty: {difficulty}

STRICT SOURCE RULES:
- The SOURCE MATERIAL is the only factual authority.
- Every question and every correct answer must be directly supported by the source.
- Never invent outside facts, even if you know them.
- Exactly four distinct options per question.
- correct_answer must exactly equal one of the four option strings.
- Make distractors plausible but clearly incorrect according to the source.
- Use varied question types when supported by the source.
- Do not repeat or paraphrase an excluded question.
- Explanations must state why the correct option follows from the source.

Previously used questions:
{excluded}

SOURCE MATERIAL:
{source}

Return only the required structured questions array."""
        content = await self._request(
            "You are the document-to-MCQ engine for AI LearnMate. Generate reliable, source-grounded educational MCQs and verify every option and answer before responding.",
            prompt,
            max_output_tokens=max(3000, count * 700),
            json_schema=MCQ_SCHEMA,
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
        """Run the full tool-using AI agent for a chat turn."""
        try:
            from .agent import LearnMateAgent

            async def execute_tool(name: str, args: dict[str, Any]):
                if name == "get_learning_profile":
                    return {
                        "learner_name": context.get("learner_name"),
                        "mastery": context.get("mastery", {}),
                        "weak_topics": context.get("weak_topics", []),
                        "strong_topics": context.get("strong_topics", []),
                        "learning_path": context.get("learning_path", []),
                        "recent_results": context.get("recent_results", []),
                        "recent_mistakes": context.get("recent_mistakes", []),
                    }
                if name == "get_material_context":
                    material = context.get("material_context")
                    if not material:
                        return {"available": False, "message": "No learning material is selected for this chat."}
                    return {"available": True, "title": context.get("material_title", "Selected material"), "content": material[:16000]}
                if name == "start_quiz":
                    return {"available": False, "message": "Quiz creation is handled by the LearnMate assessment action in the chat endpoint."}
                return {"error": f"Unknown tool: {name}"}

            agent = LearnMateAgent(execute_tool)
            reply, _actions = await agent.run(message, context)
            return reply if reply else None
        except Exception as exc:
            logger.warning("LearnMate agent failed; using direct Responses fallback: %s: %s", type(exc).__name__, exc)
            compact_context = dict(context)
            conversation = compact_context.get("conversation", [])
            if conversation:
                compact_context["conversation"] = conversation[-12:]
            context_json = json.dumps(compact_context, ensure_ascii=True)
            prompt = f"""CURRENT learner message (highest priority):
{message}

Learner context (background only):
{context_json}

Answer the current message directly. Do not let the selected topic override the current request. Handle greetings, arithmetic, coding, general questions, and study questions normally. Be accurate, concise, and helpful."""
            return await self._request(
                "You are the AI LearnMate tutor. The current learner message always takes priority over prior context.",
                prompt,
                max_output_tokens=1800,
            )
