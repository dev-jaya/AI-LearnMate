"""Fast deterministic backend and frontend contract checks used by CI."""
import asyncio
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
from app.agent import LearnMateAgent, _safe_calculate
from app.materials import extract_text, material_chunks, validate_igot_url
from app.openai_responses import OpenAIResponsesProvider

def run():
    assert _safe_calculate("2 + 10")["result"] == 12
    assert _safe_calculate("3 * (4 + 2)")["result"] == 18
    async def agent_check():
        agent = LearnMateAgent(lambda name, args: asyncio.sleep(0, result={"ok": True}))
        responses = [{"output": [{"type": "function_call", "name": "calculate", "arguments": '{"expression":"2+3"}', "call_id": "smoke-1"}]}, {"output_text": "2 + 3 = 5."}]
        async def fake_request(instructions, input_data): return responses.pop(0)
        agent._request = fake_request
        reply, actions = await agent.run("2+3", {"topic": "Binary Search", "conversation": []})
        assert reply == "2 + 3 = 5." and actions == []
    asyncio.run(agent_check())
    text = "Binary search works on sorted data by repeatedly checking the middle element and discarding the impossible half. " * 3
    assert "Binary search" in extract_text("lesson.txt", text.encode(), "text/plain")
    assert len(material_chunks(text, size=100, overlap=20)) > 1
    assert validate_igot_url("https://igotkarmayogi.gov.in/course/123").startswith("https://igotkarmayogi.gov.in")
    for filename, message in [("lesson.exe", "Unsupported file type"), ("empty.txt", "enough readable text")]:
        try: extract_text(filename, b"tiny", "application/octet-stream")
        except ValueError as exc: assert message in str(exc)
        else: raise AssertionError("invalid document should be rejected")
    provider = OpenAIResponsesProvider()
    assert provider.name == "openai"
    async def provider_check():
        async def fake_request(system, prompt, max_output_tokens=3500, tools=None, json_schema=None):
            assert json_schema is not None
            return json.dumps({"questions": [{"question": "Which condition does binary search require?", "options": ["Sorted data", "Random data only", "An image", "A database server"], "correct_answer": "Sorted data", "explanation": "Binary search relies on ordered data.", "difficulty": "easy", "question_type": "conceptual", "subject": "Algorithms", "topic": "Binary Search", "subtopic": "Searching"}]})
        provider._request = fake_request
        questions = await provider.generate_material_questions(text, "Algorithms", "Binary Search", "easy", 1, [])
        assert len(questions) == 1 and questions[0]["provider"] == "openai" and questions[0]["answer"] == 0
    asyncio.run(provider_check())
    backend_source = (ROOT / "backend/app/main.py").read_text(encoding="utf-8")
    frontend_source = (ROOT / "frontend/src/main.jsx").read_text(encoding="utf-8")
    assert "async def _generate_material_quiz" in backend_source and "provider.generate_material_questions" in backend_source
    assert "requires the Gemini provider" not in backend_source
    assert '@app.get("/api/conversations/{learner_id}")' in backend_source and '@app.get("/api/conversations/{conversation_id}/messages")' in backend_source
    for contract in ["loadConversationHistory", "openConversation", "Conversation history", "/conversations/", "chatMessagesRef", "scrollTo", "New conversation", "Thinking…", "Shift+Enter", "code-block", "copy-code", "AI LearnMate Assistant", "currentQuestion", "Next Question", "Restart Quiz"]:
        assert contract in frontend_source, contract
    assert "const activeMaterialId=forcedMaterialId??materialId" in frontend_source and "material_id:activeMaterialId" in frontend_source
    assert "correct_answer" in (ROOT / "backend/app/openai_responses.py").read_text(encoding="utf-8")
    print("AI LearnMate conversational assistant smoke tests: PASS")

if __name__ == "__main__": run()
