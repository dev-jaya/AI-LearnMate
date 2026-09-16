"""Fast deterministic backend/frontend contract checks used by CI."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.ai_providers import GeminiProvider
from app.materials import extract_text, material_chunks, validate_igot_url


def run():
    text = "Binary search works on sorted data by repeatedly checking the middle element and discarding the impossible half. " * 3
    assert "Binary search" in extract_text("lesson.txt", text.encode(), "text/plain")
    assert len(material_chunks(text, size=100, overlap=20)) > 1
    assert validate_igot_url("https://igotkarmayogi.gov.in/course/123").startswith("https://igotkarmayogi.gov.in")
    for filename, message in [("lesson.exe", "Unsupported file type"), ("empty.txt", "enough readable text")]:
        try: extract_text(filename, b"tiny", "application/octet-stream")
        except ValueError as exc: assert message in str(exc)
        else: raise AssertionError("invalid document should be rejected")

    provider = GeminiProvider(api_key="test-key", model="test-model")
    assert provider.name == "gemini"

    async def provider_check():
        async def fake_generate(contents, system, response_schema=None, json_mode=False):
            assert response_schema is not None and json_mode
            return json.dumps({"questions": [{"question": "Which condition does binary search require?", "options": ["Sorted data", "Random data only", "An image", "A database server"], "correct_answer": "Sorted data", "explanation": "Binary search relies on ordered data.", "difficulty": "easy", "question_type": "conceptual", "subject": "Algorithms", "topic": "Binary Search", "subtopic": "Searching"}]})
        provider._generate = fake_generate
        questions = await provider.generate_material_questions(text, "Algorithms", "Binary Search", "easy", 1, [])
        assert len(questions) == 1 and questions[0]["provider"] == "gemini" and questions[0]["answer"] == 0
    asyncio.run(provider_check())

    backend_source = (ROOT / "backend/app/main.py").read_text(encoding="utf-8")
    provider_source = (ROOT / "backend/app/ai_providers.py").read_text(encoding="utf-8")
    frontend_source = (ROOT / "frontend/src/main.jsx").read_text(encoding="utf-8")
    assert "provider.generate_material_questions" in backend_source
    assert '@app.get("/api/conversations/{learner_id}")' in backend_source
    assert '@app.get("/api/conversations/{conversation_id}/messages")' in backend_source
    assert '"provider":"gemini"' in backend_source.replace(" ", "")
    for forbidden in ("ollama", "chatgpt", "api.openai.com"):
        assert forbidden not in backend_source.lower()
        assert forbidden not in provider_source.lower()
    for contract in ["loadConversationHistory", "openConversation", "Conversation history", "/conversations/", "chatMessagesRef", "scrollTo", "New conversation", "Thinking…", "Shift+Enter", "code-block", "copy-code", "AI LearnMate Assistant", "currentQuestion", "Next Question", "Restart Quiz"]:
        assert contract in frontend_source, contract
    assert "const activeMaterialId=forcedMaterialId??materialId" in frontend_source and "material_id:activeMaterialId" in frontend_source
    print("AI LearnMate Gemini assistant smoke tests: PASS")


if __name__ == "__main__": run()
