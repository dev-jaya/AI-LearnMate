import asyncio
from uuid import uuid4

from sqlalchemy.orm import Session

from app import main
from app.models import Learner
from app.materials import Material
from app.schemas import ChatRequest, QuizRequest


class FakeProvider:
    name = "openai"
    model = "test-model"
    api_key = "test-key"
    last_error = ""

    async def chat(self, message, context):
        return f"Echo: {message}"

    async def generate_material_questions(self, material_text, subject, topic, difficulty, count, excluded_questions):
        return [{
            "id": f"test-{index}",
            "subject": subject,
            "topic": topic,
            "subtopic": "Material-based",
            "difficulty": difficulty,
            "question": f"What does the material explain in question {index}?",
            "options": ["The first concept", "The second concept", "The third concept", "The fourth concept"],
            "answer": 0,
            "explanation": "The answer is stated in the supplied material.",
            "fingerprint": f"test-fingerprint-{index}-{uuid4().hex}",
            "question_type": "conceptual",
            "provider": "openai",
        } for index in range(count)]


def _new_learner(db: Session, prefix: str) -> Learner:
    learner = Learner(name=f"{prefix}-{uuid4().hex[:8]}")
    db.add(learner)
    db.commit()
    db.refresh(learner)
    return learner


def test_chat_history_is_saved_and_reloadable(monkeypatch):
    monkeypatch.setattr(main, "_openai_configured", lambda: True)
    monkeypatch.setattr(main, "OpenAIResponsesProvider", FakeProvider)
    with Session(bind=main.engine) as db:
        learner = _new_learner(db, "CI-Chat")
        response = asyncio.run(main.chat(ChatRequest(
            learner_id=learner.id,
            message="Hello from automated tests",
            topic="Java",
        ), db))
        conversation_id = response["conversation_id"]
        history = main.list_conversations(learner.id, db)["conversations"]
        assert any(item["id"] == conversation_id for item in history)
        messages = main.get_conversation_messages(conversation_id, learner.id, db)["messages"]
        contents = [item["content"] for item in messages]
        assert "Hello from automated tests" in contents
        assert "Echo: Hello from automated tests" in contents


def test_material_quiz_uses_configured_provider_without_gemini(monkeypatch):
    monkeypatch.setattr(main, "_openai_configured", lambda: True)
    monkeypatch.setattr(main, "OpenAIResponsesProvider", FakeProvider)
    with Session(bind=main.engine) as db:
        learner = _new_learner(db, "CI-Material")
        material = Material(
            learner_id=learner.id,
            title="Binary Search Lesson",
            filename="lesson.txt",
            mime_type="text/plain",
            source_type="upload",
            extracted_text=("Binary search works on sorted data by repeatedly checking the middle element "
                            "and discarding the impossible half. " * 3),
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
        assert all(question["answer"] in range(4) for question in quiz["questions"])
