import asyncio
from uuid import uuid4
from sqlalchemy.orm import Session
from app import main
from app.models import Learner
from app.materials import Material
from app.schemas import ChatRequest, QuizRequest


class FakeProvider:
    name="gemini"; model="test-model"; api_key="test-key"; last_error=""
    async def chat(self,message,context): return f"Echo: {message}"
    async def generate_questions(self,subject,topic,subtopic,difficulty,count,context): return self._questions(subject,topic,difficulty,count)
    async def generate_material_questions(self,material_text,subject,topic,difficulty,count,excluded_questions): return self._questions(subject,topic,difficulty,count)
    def _questions(self,subject,topic,difficulty,count):
        return [{"id":f"test-{i}","subject":subject,"topic":topic,"subtopic":"Material-based","difficulty":difficulty,"question":f"What does the material explain in question {i}?","options":["The first concept","The second concept","The third concept","The fourth concept"],"answer":0,"correct_answer":0,"explanation":"The answer is stated in the supplied material.","fingerprint":f"test-fingerprint-{i}-{uuid4().hex}","question_type":"conceptual","provider":"gemini"} for i in range(count)]


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
