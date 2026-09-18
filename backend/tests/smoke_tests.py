"""Fast deterministic backend/frontend contract checks used by CI."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.ai_providers import (
    MCQ_SCHEMA,
    GeminiProvider,
    _legacy_schema,
    get_provider,
    validate_question,
)
from app.config import settings
from app.main import adaptive_difficulty_plan
from app.materials import extract_text, material_chunks, validate_igot_url


def run():
    text = (
        "Binary search works on sorted data by repeatedly checking the middle "
        "element and discarding the impossible half. "
    ) * 4
    assert "Binary search" in extract_text("lesson.txt", text.encode(), "text/plain")
    assert len(material_chunks(text, size=100, overlap=20)) > 1
    assert validate_igot_url(
        "https://igotkarmayogi.gov.in/course/123"
    ).startswith("https://igotkarmayogi.gov.in")

    for filename, message in [
        ("lesson.exe", "Unsupported file type"),
        ("empty.txt", "enough readable text"),
    ]:
        try:
            extract_text(filename, b"tiny", "application/octet-stream")
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("invalid document should be rejected")

    provider = GeminiProvider(api_key="test-key", model="test-model")
    assert provider.name == "gemini"
    assert get_provider().name == "gemini"
    assert settings.gemini_model == "gemini-3.8-flash"
    assert GeminiProvider(api_key="test-key", model="gemini-3.8-flash").model == "gemini-3.8-flash"
    assert MCQ_SCHEMA["type"] == "object"
    assert MCQ_SCHEMA["properties"]["questions"]["type"] == "array"
    assert MCQ_SCHEMA["properties"]["questions"]["minItems"] == 1
    assert MCQ_SCHEMA["properties"]["questions"]["maxItems"] == 20
    assert _legacy_schema(MCQ_SCHEMA)["type"] == "OBJECT"
    assert _legacy_schema(MCQ_SCHEMA)["properties"]["questions"]["type"] == "ARRAY"

    question = validate_question(
        {
            "question": "Which condition does binary search require?",
            "options": ["Sorted data", "Random data only", "An image", "A server"],
            "correct_answer": "a",
            "explanation": "Binary search relies on ordered data.",
            "difficulty": "easy",
        },
        "Algorithms",
        "Binary Search",
        "Searching",
        "easy",
    )
    assert question and question["answer"] == 0 and question["provider"] == "gemini"

    invalid = validate_question(
        {
            "question": "Duplicate options",
            "options": ["Same", "same", "A", "B"],
            "correct_answer": "A",
            "explanation": "bad question",
        },
        "Algorithms",
        "Binary Search",
        "",
        "easy",
    )
    assert invalid is None

    assert adaptive_difficulty_plan(None, 5) == [
        "easy",
        "easy",
        "medium",
        "medium",
        "hard",
    ]
    assert adaptive_difficulty_plan(40, 5) == [
        "easy",
        "easy",
        "easy",
        "medium",
        "medium",
    ]
    assert adaptive_difficulty_plan(65, 5) == [
        "easy",
        "medium",
        "medium",
        "hard",
        "hard",
    ]
    assert adaptive_difficulty_plan(90, 5) == [
        "medium",
        "medium",
        "hard",
        "hard",
        "hard",
    ]

    async def provider_check():
        async def fake_generate(
            contents, system, response_schema=None, json_mode=False
        ):
            assert response_schema is not None and json_mode
            assert response_schema["type"] == "object"
            return json.dumps(
                {
                    "questions": [
                        {
                            "question": "Which condition does binary search require?",
                            "options": [
                                "Sorted data",
                                "Random data only",
                                "An image",
                                "A database server",
                            ],
                            "correct_answer": "Sorted data",
                            "explanation": "Binary search relies on ordered data.",
                            "difficulty": "easy",
                            "question_type": "conceptual",
                            "subject": "Algorithms",
                            "topic": "Binary Search",
                            "subtopic": "Searching",
                            "source_evidence": "Binary search works on sorted data by repeatedly checking the middle element",
                        }
                    ]
                }
            )

        provider._generate = fake_generate
        questions = await provider.generate_material_questions(
            text, "Algorithms", "Binary Search", "easy", 1, []
        )
        assert (
            len(questions) == 1
            and questions[0]["provider"] == "gemini"
            and questions[0]["answer"] == 0
        )

    asyncio.run(provider_check())

    backend_source = (ROOT / "backend/app/main.py").read_text(encoding="utf-8")
    provider_source = (
        ROOT / "backend/app/ai_providers.py"
    ).read_text(encoding="utf-8")
    config_source = (ROOT / "backend/app/config.py").read_text(encoding="utf-8")
    frontend_source = (
        ROOT / "frontend/src/main.jsx"
    ).read_text(encoding="utf-8")
    frontend_index = (
        ROOT / "frontend/index.html"
    ).read_text(encoding="utf-8")
    difficulty_source = (
        ROOT / "frontend/src/practice-difficulty.js"
    ).read_text(encoding="utf-8")

    assert "provider.generate_material_questions" in backend_source
    assert '@app.get("/api/conversations/{learner_id}")' in backend_source
    assert '@app.get("/api/conversations/{conversation_id}/messages")' in backend_source
    assert '"provider":"gemini"' in backend_source.replace(" ", "")
    assert "adaptive_difficulty_plan" in backend_source
    assert "shuffle_question_options" in backend_source
    assert "infer_subject" in backend_source
    assert "safe_gemini_failure" in backend_source
    assert "frontend_origins" in config_source
    assert "gemini_api_key" in config_source and "gemini_model" in config_source
    assert "GeminiProvider" in provider_source and "get_provider" in provider_source
    assert "response_format" in provider_source and "x-goog-api-key" in provider_source
    assert "/health/ai/probe" in backend_source
    assert "Api-Revision" in provider_source
    assert "gemini-3.8-flash" in config_source
    assert "google import genai" in provider_source

    for contract in [
        "loadConversationHistory",
        "openConversation",
        "Conversation history",
        "/conversations/",
        "chatMessagesRef",
        "scrollTo",
        "New conversation",
        "Thinking…",
        "Shift+Enter",
        "code-block",
        "copy-code",
        "AI LearnMate Assistant",
        "currentQuestion",
        "Next Question",
        "Restart Quiz",
    ]:
        assert contract in frontend_source, contract

    assert (
        "const activeMaterialId=forcedMaterialId??materialId" in frontend_source
        and "material_id:activeMaterialId" in frontend_source
    )

    assert "/src/practice-difficulty.js" in frontend_index
    assert frontend_index.index("practice-difficulty.js") < frontend_index.index("main.jsx")
    for contract in [
        "assessment/start",
        "practice-difficulty-wrap",
        "Practice difficulty",
        "['easy', 'Easy']",
        "['medium', 'Medium']",
        "['hard', 'Difficult']",
        "/api/chat",
    ]:
        assert contract in difficulty_source, contract

    print("AI LearnMate Gemini assistant + MCQ smoke tests: PASS")


if __name__ == "__main__":
    run()
