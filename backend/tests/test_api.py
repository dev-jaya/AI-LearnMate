import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
from uuid import uuid4
from sqlalchemy.orm import Session
from app import main
from app.models import Learner
from app.materials import Material
from app.schemas import ChatRequest, QuizRequest
from app.main import safe_gemini_failure


class FakeProvider:
    name="gemini"; model="test-model"; api_key="test-key"; last_error=""; last_error_category=""; last_status_code=200; last_interaction_id=None
    async def chat(self,message,context): return f"Echo: {message}"
    async def generate_questions(self,subject,topic,subtopic,difficulty,count,context): return self._questions(subject,topic,difficulty,count)
    async def generate_material_questions(self,material_text,subject,topic,difficulty,count,excluded_questions,source_reference="document"): return self._questions(subject,topic,difficulty,count)
    def _questions(self,subject,topic,difficulty,count):
        return [{"id":f"test-{i}","subject":subject,"topic":topic,"subtopic":"Material-based","difficulty":difficulty,"question":f"What does the material explain in question {i}?","options":["The first concept","The second concept","The third concept","The fourth concept"],"answer":0,"correct_answer":0,"explanation":"The answer is stated in the supplied material.","source_evidence":"Binary search works on sorted data by repeatedly checking the middle element","fingerprint":f"test-fingerprint-{i}-{uuid4().hex}","question_type":"conceptual","provider":"gemini"} for i in range(count)]


def _new_learner(db,prefix):
    learner=Learner(name=f"{prefix}-{uuid4().hex[:8]}"); db.add(learner); db.commit(); db.refresh(learner); return learner


def test_chat_history_is_saved_and_reloadable(monkeypatch):
    monkeypatch.setattr(main,"get_provider",lambda:FakeProvider())
    with Session(bind=main.engine) as db:
        learner=_new_learner(db,"CI-Chat"); response=asyncio.run(main.chat(ChatRequest(learner_id=learner.id,message="Hello from automated tests",topic="Java"),db)); cid=response["conversation_id"]
        assert response["provider"]=="gemini"
        assert any(item["id"]==cid for item in main.list_conversations(learner.id,db)["conversations"])
        contents=[item["content"] for item in main.get_conversation_messages(cid,learner.id,db)["messages"]]
        assert "Hello from automated tests" in contents and "Echo: Hello from automated tests" in contents


def test_material_quiz_uses_gemini_provider(monkeypatch):
    monkeypatch.setattr(main,"get_provider",lambda:FakeProvider())
    with Session(bind=main.engine) as db:
        learner=_new_learner(db,"CI-Material"); material=Material(learner_id=learner.id,title="Binary Search Lesson",filename="lesson.txt",mime_type="text/plain",source_type="upload",extracted_text=("Binary search works on sorted data by repeatedly checking the middle element and discarding the impossible half. "*3)); db.add(material); db.commit(); db.refresh(material)
        quiz=asyncio.run(main._generate_material_quiz(material,QuizRequest(learner_id=learner.id,topic="Algorithms",subject="Algorithms",difficulty="easy",count=3),db))
        assert len(quiz["questions"])==3 and all(q["provider"]=="gemini" for q in quiz["questions"])


def test_adaptive_quiz_submit_updates_dashboard(monkeypatch):
    monkeypatch.setattr(main,"get_provider",lambda:FakeProvider())
    with Session(bind=main.engine) as db:
        learner=_new_learner(db,"CI-Quiz")
        req=QuizRequest(learner_id=learner.id,topic="Algorithms",subject="Algorithms",difficulty="easy",count=3)
        quiz=asyncio.run(main.generate_quiz(req,db))
        assert len(quiz["questions"])==3
        assert all(len(q["options"])==4 and q["answer"] in range(4) for q in quiz["questions"])
        result=main.submit_assessment(__import__("app.schemas",fromlist=["SubmitRequest"]).SubmitRequest(quiz_id=quiz["id"],learner_id=learner.id,answers=[q["answer"] for q in quiz["questions"]]),db)
        assert result["score"]==100.0 and result["total"]==3
        data=main.dashboard(learner.id,db)
        assert data["attempts"]==1 and data["average_score"]==100.0
        assert data["mastery"]["Algorithms"]==100.0


def test_chat_quiz_intent_returns_persisted_quiz(monkeypatch):
    monkeypatch.setattr(main,"get_provider",lambda:FakeProvider())
    with Session(bind=main.engine) as db:
        learner=_new_learner(db,"CI-ChatQuiz")
        response=asyncio.run(main.chat(ChatRequest(learner_id=learner.id,message="Test me with 3 questions on Python",topic="Python"),db))
        assert response["quiz"] is not None
        assert len(response["quiz"]["questions"])==3
        messages=main.get_conversation_messages(response["conversation_id"],learner.id,db)["messages"]
        assert messages[-1]["role"]=="assistant"


def test_material_upload_and_grounded_chat_context(monkeypatch):
    captured={}
    provider=FakeProvider()
    async def chat(message,context):
        captured.update(context)
        return "Grounded response"
    provider.chat=chat
    monkeypatch.setattr(main,"get_provider",lambda:provider)
    with Session(bind=main.engine) as db:
        learner=_new_learner(db,"CI-Grounded")
        material=Material(learner_id=learner.id,title="Java Basics",filename="java.txt",mime_type="text/plain",source_type="upload",extracted_text=("Java classes define state and behavior. Objects are created from classes. "*4))
        db.add(material); db.commit(); db.refresh(material)
        response=asyncio.run(main.chat(ChatRequest(learner_id=learner.id,message="Explain classes using my material",topic="Java",material_id=material.id),db))
        assert response["message"]["content"]=="Grounded response"
        assert "material_context" in captured and "Java classes" in captured["material_context"]


def test_no_repeat_uses_learner_history(monkeypatch):
    monkeypatch.setattr(main, "get_provider", lambda: FakeProvider())
    with Session(bind=main.engine) as db:
        learner = _new_learner(db, "CI-NoRepeat")
        first = asyncio.run(main.generate_quiz(
            QuizRequest(learner_id=learner.id, topic="Java", subject="Java", difficulty="easy", count=3),
            db,
        ))
        second = asyncio.run(main.generate_quiz(
            QuizRequest(learner_id=learner.id, topic="Python", subject="Python", difficulty="easy", count=3),
            db,
        ))
        first_questions = {q["question"] for q in first["questions"]}
        second_questions = {q["question"] for q in second["questions"]}
        assert first_questions.isdisjoint(second_questions)


def test_chat_does_not_require_external_gemini_session_state(monkeypatch):
    provider = FakeProvider()
    captured = {}
    async def chat(message, context):
        captured["conversation"] = list(context.get("conversation", []))
        return "ok"
    provider.chat = chat
    monkeypatch.setattr(main, "get_provider", lambda: provider)
    with Session(bind=main.engine) as db:
        learner = _new_learner(db, "CI-State")
        first = asyncio.run(main.chat(ChatRequest(
            learner_id=learner.id, message="My name is Kumar", topic="Java"
        ), db))
        asyncio.run(main.chat(ChatRequest(
            learner_id=learner.id,
            message="What is my name?",
            topic="Java",
            conversation_id=first["conversation_id"],
        ), db))
        texts = [item["content"] for item in captured["conversation"]]
        assert texts.count("What is my name?") == 1
        assert "My name is Kumar" in texts



def test_gemini_error_categories_are_safe(monkeypatch):
    provider = FakeProvider()
    for status, category, expected_status in [
        (401, "authentication", 503),
        (403, "authentication", 503),
        (404, "model_not_found", 502),
        (429, "quota", 429),
        (500, "service_unavailable", 503),
    ]:
        provider.last_status_code = status
        provider.last_error_category = category
        provider.last_error = "secret-safe-error"
        exc = safe_gemini_failure(provider, "default")
        assert exc.status_code == expected_status
        assert "test-key" not in str(exc.detail)


def test_material_context_uses_user_question(monkeypatch):
    captured = {}
    provider = FakeProvider()
    async def chat(message, context):
        captured["material_context"] = context.get("material_context", "")
        return "grounded"
    provider.chat = chat
    monkeypatch.setattr(main, "get_provider", lambda: provider)
    with Session(bind=main.engine) as db:
        learner = _new_learner(db, "CI-MaterialQuery")
        material = Material(
            learner_id=learner.id,
            title="Topic Map",
            filename="topic.txt",
            mime_type="text/plain",
            source_type="upload",
            extracted_text=(
                "ALPHA fact about arrays. "
                "BETA fact about recursion. "
                "GAMMA fact about JVM internals. "
            ) * 8,
        )
        db.add(material)
        db.commit()
        db.refresh(material)
        asyncio.run(main.chat(ChatRequest(
            learner_id=learner.id,
            message="What does the material say about recursion?",
            topic="Java",
            material_id=material.id,
        ), db))
        assert "recursion" in captured["material_context"].lower()
