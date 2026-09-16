"""Fast deterministic backend checks used by CI; no network or API key required."""
import asyncio
import json
from uuid import uuid4

from sqlalchemy.orm import Session

from app import main
from app.agent import LearnMateAgent, _safe_calculate
from app.materials import extract_text, material_chunks, validate_igot_url
from app.models import Learner
from app.openai_responses import OpenAIResponsesProvider
from app.schemas import ChatRequest, QuizRequest


class FakeProvider:
    name = "openai"
    model = "smoke-model"
    api_key = "smoke-key"
    last_error = ""

    async def chat(self, message, context):
        return f"Echo: {message}"

    async def generate_material_questions(self, material_text, subject, topic, difficulty, count, excluded_questions):
        return [{
            "id": f"smoke-{index}",
            "subject": subject,
            "topic": topic,
            "subtopic": "Material-based",
            "difficulty": difficulty,
            "question": f"What concept is supported by the material, item {index}?",
            "options": ["Binary search", "Unrelated topic", "Another unrelated topic", "No concept"],
            "answer": 0,
            "correct_answer": 0,
            "explanation": "The supplied material supports binary search.",
            "fingerprint": f"smoke-{index}-{uuid4().hex}",
            "question_type": "conceptual",
            "provider": "openai",
        } for index in range(count)]


def run():
    # Calculator and agent orchestration.
    assert _safe_calculate("2 + 10")["result"] == 12
    assert _safe_calculate("3 * (4 + 2)")["result"] == 18

    async def agent_check():
        agent = LearnMateAgent(lambda name, args: asyncio.sleep(0, result={"ok": True}))
        responses = [
            {"output": [{"type": "function_call", "name": "calculate", "arguments": '{"expression":"2+3"}', "call_id": "smoke-1"}]},
            {"output_text": "2 + 3 = 5."},
        ]

        async def fake_request(instructions, input_data):
            return responses.pop(0)

        agent._request = fake_request
        reply, actions = await agent.run("2+3", {"topic": "Binary Search", "conversation": []})
        assert reply == "2 + 3 = 5."
        assert actions == []

    asyncio.run(agent_check())

    # Material extraction and official-domain validation.
    text = "Binary search works on sorted data by repeatedly checking the middle element and discarding the impossible half. " * 3
    assert "Binary search" in extract_text("lesson.txt", text.encode(), "text/plain")
    assert len(material_chunks(text, size=100, overlap=20)) > 1
    assert validate_igot_url("https://igotkarmayogi.gov.in/course/123").startswith("https://igotkarmayogi.gov.in")

    provider = OpenAIResponsesProvider()
    assert provider.name == "openai"
    assert callable(provider.generate_material_questions)

    # Validate OpenAI material-MCQ parsing without calling the network.
    async def provider_check():
        async def fake_request(system, prompt, max_output_tokens=3500, tools=None):
            return json.dumps({"questions": [{
                "question": "Which condition does binary search require?",
                "options": ["Sorted data", "Random data only", "An image", "A database server"],
                "correct_answer": "Sorted data",
                "explanation": "Binary search relies on ordered data.",
                "difficulty": "easy",
                "question_type": "conceptual",
                "subject": "Algorithms",
                "topic": "Binary Search",
                "subtopic": "Searching",
            }]})

        provider._request = fake_request
        questions = await provider.generate_material_questions(text, "Algorithms", "Binary Search", "easy", 1, [])
        assert len(questions) == 1
        assert questions[0]["provider"] == "openai"
        assert questions[0]["answer"] == 0

    asyncio.run(provider_check())

    # Conversation persistence and provider-neutral material quiz routing.
    original_configured = main._openai_configured
    original_provider = main.OpenAIResponsesProvider
    main._openai_configured = lambda: True
    main.OpenAIResponsesProvider = FakeProvider
    try:
        with Session(bind=main.engine) as db:
            learner = Learner(name=f"CI-Smoke-{uuid4().hex[:8]}")
            db.add(learner)
            db.commit()
            db.refresh(learner)

            chat_result = asyncio.run(main.chat(ChatRequest(
                learner_id=learner.id,
                message="Hello smoke test",
                topic="Java",
            ), db))
            conversation_id = chat_result["conversation_id"]
            history = main.list_conversations(learner.id, db)["conversations"]
            assert any(item["id"] == conversation_id for item in history)
            messages = main.get_conversation_messages(conversation_id, learner.id, db)["messages"]
            assert any(item["content"] == "Hello smoke test" for item in messages)
            assert any(item["content"] == "Echo: Hello smoke test" for item in messages)

            from app.materials import Material
            material = Material(
                learner_id=learner.id,
                title="Binary Search Lesson",
                filename="lesson.txt",
                mime_type="text/plain",
                source_type="upload",
                extracted_text=text,
            )
            db.add(material)
            db.commit()
            db.refresh(material)
            quiz = asyncio.run(main._generate_material_quiz(material, QuizRequest(
                learner_id=learner.id,
                topic="Algorithms",
                subject="Algorithms",
                difficulty="easy",
                count=3,
            ), db))
            assert len(quiz["questions"]) == 3
            assert all(question["provider"] == "openai" for question in quiz["questions"])
    finally:
        main._openai_configured = original_configured
        main.OpenAIResponsesProvider = original_provider

    print("AI LearnMate backend smoke tests: PASS")


if __name__ == "__main__":
    run()
