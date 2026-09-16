"""Fast deterministic smoke tests for the backend CI pipeline."""
import asyncio
import json

from app.agent import _safe_calculate
from app.materials import extract_text, material_chunks, validate_igot_url
from app.openai_responses import OpenAIResponsesProvider


class FakeOpenAIProvider(OpenAIResponsesProvider):
    async def _request(self, system, prompt, max_output_tokens=3500, tools=None):
        return json.dumps({
            "questions": [{
                "question": "What does the supplied material say binary search requires?",
                "options": ["Sorted data", "Unsorted data only", "A database", "A network"],
                "correct_answer": "Sorted data",
                "explanation": "The source states that binary search works on sorted data.",
                "difficulty": "easy",
                "question_type": "conceptual",
                "subject": "Algorithms",
                "topic": "Binary Search",
                "subtopic": "Searching",
            }]
        })


async def run():
    assert _safe_calculate("2 + 10")["result"] == 12
    assert _safe_calculate("3 * (4 + 2)")["result"] == 18

    text = "Binary search works on sorted data by repeatedly checking the middle element and discarding the impossible half. " * 3
    assert "Binary search" in extract_text("lesson.txt", text.encode(), "text/plain")
    assert len(material_chunks(text, size=100, overlap=20)) > 1
    assert validate_igot_url("https://igotkarmayogi.gov.in/course/123").startswith("https://igotkarmayogi.gov.in")

    provider = FakeOpenAIProvider()
    questions = await provider.generate_material_questions(
        text, "Algorithms", "Binary Search", "easy", 1, []
    )
    assert questions and len(questions) == 1
    assert questions[0]["provider"] == "openai"
    assert len(questions[0]["options"]) == 4
    assert questions[0]["answer"] == 0


if __name__ == "__main__":
    asyncio.run(run())
    print("AI LearnMate backend smoke tests: PASS")
