from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
from datetime import datetime
import json
import re
import uuid

from .db import Base, engine, get_db
from .models import Learner, Attempt, Quiz, Conversation, Message, GeneratedQuestion, QuestionFingerprint, LearnerQuestionHistory, Subject, Topic, QuizQuestion
from .materials import Material, extract_text, retrieve_material_context, import_igot_resource
from .material_ai import generate_material_questions
from .schemas import *
from .ai_providers import SUBJECTS, FallbackProvider, get_provider, question_similarity

Base.metadata.create_all(bind=engine)
app = FastAPI(title="AI LearnMate API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:5500", "http://127.0.0.1:5173", "http://127.0.0.1:5500", "https://ai-learnmate-frontend.onrender.com"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def migrate_sqlite_columns():
    if not str(engine.url).startswith("sqlite"):
        return
    columns = {item["name"] for item in inspect(engine).get_columns("generated_questions")}
    additions = {"normalized_question": "TEXT NOT NULL DEFAULT ''", "question_type": "VARCHAR(40) NOT NULL DEFAULT 'conceptual'"}
    with engine.begin() as connection:
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(text(f"ALTER TABLE generated_questions ADD COLUMN {name} {definition}"))


migrate_sqlite_columns()


def seed_subject_catalog():
    from .ai_providers import CONCEPTS
    with Session(bind=engine) as db:
        for subject_name in SUBJECTS:
            subject = db.query(Subject).filter(Subject.name == subject_name).first()
            if not subject:
                subject = Subject(name=subject_name)
                db.add(subject)
                db.flush()
            existing = {item.name for item in subject.topics}
            names = {concept[4] for concept in CONCEPTS.get(subject_name, [])}
            for topic_name in names:
                if topic_name not in existing:
                    db.add(Topic(subject_id=subject.id, name=topic_name))
        db.commit()


seed_subject_catalog()


@app.get("/")
def root():
    return {"name": "AI LearnMate", "status": "running", "version": "1.1.0"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-learnmate-api"}


@app.get("/health/ai")
def ai_health():
    provider = get_provider()
    return {"provider": provider.name, "model": getattr(provider, "model", None), "key_loaded": bool(getattr(provider, "api_key", ""))}


@app.get("/api/igot")
def igot_info():
    return {
        "platform": "iGOT Karmayogi",
        "official_url": "https://igotkarmayogi.gov.in/",
        "public_content_url": "https://portal.igotkarmayogi.gov.in/",
        "integration_mode": "official public-resource import",
        "note": "Paste a public iGOT Karmayogi resource URL or upload the official learning material. Authenticated government APIs require official credentials and access not available to this demo."
    }


@app.post("/api/learners", response_model=LearnerOut)
def create_learner(data: LearnerCreate, db: Session = Depends(get_db)):
    name = data.name.strip()
    if not name:
        raise HTTPException(400, "Name cannot be empty")
    learner = Learner(name=name)
    db.add(learner)
    db.commit()
    db.refresh(learner)
    return learner


@app.get("/api/learners/{learner_id}", response_model=LearnerOut)
def get_learner(learner_id: int, db: Session = Depends(get_db)):
    learner = db.get(Learner, learner_id)
    if not learner:
        raise HTTPException(404, "Learner not found")
    return learner


@app.get("/api/topics")
def topics(db: Session = Depends(get_db)):
    records = db.query(Subject).filter(Subject.active.is_(True)).order_by(Subject.name).all()
    return {"topics": [record.name for record in records], "subjects": [{"name": record.name, "topics": [item.name for item in record.topics if item.active]} for record in records]}


def learner_mastery(learner_id: int, db: Session):
    attempts = db.query(Attempt).filter(Attempt.learner_id == learner_id).order_by(Attempt.created_at.desc()).all()
    topic_scores = {}
    for attempt in attempts:
        topic_scores.setdefault(attempt.topic, []).append(attempt.score)
    return {topic: round(sum(scores) / len(scores), 1) for topic, scores in topic_scores.items()}


def store_questions(db: Session, learner_id: int, questions: list[dict], topic: str, difficulty: str) -> Quiz:
    quiz = Quiz(learner_id=learner_id, topic=topic, difficulty=difficulty, questions_json=json.dumps(questions))
    db.add(quiz)
    db.flush()
    for position, question in enumerate(questions):
        stored = GeneratedQuestion(
            public_id=question["id"], learner_id=learner_id, subject=question["subject"], topic=question["topic"],
            subtopic=question["subtopic"], difficulty=question["difficulty"], question=question["question"],
            normalized_question=re.sub(r"[^a-z0-9]+", " ", question["question"].lower()).strip(),
            options_json=json.dumps(question["options"]), answer=question["answer"], explanation=question["explanation"],
            fingerprint=question["fingerprint"], provider=question.get("provider", "fallback"),
            question_type=question.get("question_type", "conceptual")
        )
        db.add(stored)
        db.flush()
        db.add(QuestionFingerprint(fingerprint=question["fingerprint"], question_id=stored.id))
        db.add(LearnerQuestionHistory(learner_id=learner_id, question_id=stored.id, fingerprint=question["fingerprint"]))
        db.add(QuizQuestion(quiz_id=quiz.id, question_id=stored.id, position=position))
    db.commit()
    db.refresh(quiz)
    return quiz


async def generate_quiz(req: QuizRequest, db: Session):
    learner = db.get(Learner, req.learner_id)
    if not learner:
        raise HTTPException(404, "Learner not found")
    subject = req.subject or req.topic
    mastery = learner_mastery(req.learner_id, db)
    difficulty = req.difficulty
    if difficulty == "adaptive":
        score = mastery.get(req.topic)
        difficulty = "hard" if score is not None and score >= 80 else "medium" if score is not None and score >= 50 else "easy"
    history_rows = db.query(LearnerQuestionHistory).filter(LearnerQuestionHistory.learner_id == req.learner_id).all()
    excluded = {row.fingerprint for row in history_rows}
    question_ids = [row.question_id for row in history_rows]
    excluded_questions = [row.question for row in db.query(GeneratedQuestion).filter(GeneratedQuestion.id.in_(question_ids), GeneratedQuestion.topic == req.topic).order_by(GeneratedQuestion.created_at.desc()).limit(30).all()] if question_ids else []
    provider = get_provider()
    provider_failures = 0
    context = {"mastery": mastery, "weak_topics": [name for name, score in mastery.items() if score < 70], "strong_topics": [name for name, score in mastery.items() if score >= 80], "excluded_fingerprints": list(excluded), "excluded_questions": excluded_questions}
    questions = []
    for _ in range(3):
        remaining = req.count - len(questions)
        if remaining <= 0:
            break
        candidates = await provider.generate_questions(subject, req.topic, req.subtopic, difficulty, max(remaining * 2, 6), context)
        if candidates is None and provider.name != "fallback":
            provider_failures += 1
            if provider_failures >= 2:
                provider = FallbackProvider()
                candidates = await provider.generate_questions(subject, req.topic, req.subtopic, difficulty, max(remaining * 2, 6), context)
            else:
                candidates = []
        elif candidates:
            provider_failures = 0
        for question in candidates or []:
            if question["fingerprint"] in excluded or question["fingerprint"] in {item["fingerprint"] for item in questions}:
                continue
            if any(question_similarity(question["question"], previous) >= 0.9 for previous in excluded_questions + [item["question"] for item in questions]):
                continue
            question["id"] = f"{provider.name}-{uuid.uuid4().hex[:16]}"
            question["provider"] = provider.name
            questions.append(question)
            if len(questions) == req.count:
                break
        context["excluded_fingerprints"] = list(excluded | {item["fingerprint"] for item in questions})
        context["excluded_questions"] = excluded_questions + [item["question"] for item in questions]
        if len(questions) < req.count and provider.name != "fallback":
            provider = get_provider()
    if len(questions) < req.count:
        raise HTTPException(409, f"Only {len(questions)} new validated questions are available for this topic. Try another topic or reset practice history.")
    quiz = store_questions(db, req.learner_id, questions, req.topic, difficulty)
    return {"id": quiz.id, "topic": req.topic, "difficulty": difficulty, "questions": questions}


@app.post("/api/quiz", response_model=QuizOut)
async def create_quiz(req: QuizRequest, db: Session = Depends(get_db)):
    return await generate_quiz(req, db)


@app.post("/api/assessment/start", response_model=QuizOut)
async def start_assessment(req: QuizRequest, db: Session = Depends(get_db)):
    return await generate_quiz(req, db)


@app.post("/api/attempts", response_model=AttemptOut)
def submit(req: SubmitRequest, db: Session = Depends(get_db)):
    if not db.get(Learner, req.learner_id):
        raise HTTPException(404, "Learner not found")
    quiz = db.get(Quiz, req.quiz_id) if req.quiz_id else None
    if not quiz or quiz.learner_id != req.learner_id:
        raise HTTPException(404, "Quiz not found")
    if quiz.submitted:
        raise HTTPException(409, "Quiz already submitted")
    questions = json.loads(quiz.questions_json)
    if len(req.answers) != len(questions):
        raise HTTPException(400, "Submit one answer for every question")
    if any(answer not in range(4) for answer in req.answers):
        raise HTTPException(400, "Answers must use option indexes 0 to 3")
    correct = sum(answer == question["answer"] for answer, question in zip(req.answers, questions))
    total = len(questions)
    quiz.submitted = True
    for answer, question in zip(req.answers, questions):
        stored = db.query(GeneratedQuestion).filter(GeneratedQuestion.public_id == question["id"]).first()
        if stored:
            history = db.query(LearnerQuestionHistory).filter(LearnerQuestionHistory.learner_id == req.learner_id, LearnerQuestionHistory.question_id == stored.id).order_by(LearnerQuestionHistory.seen_at.desc()).first()
            if history:
                history.answered = True
                history.correct = answer == question["answer"]
    score = round(correct * 100 / total, 2)
    attempt = Attempt(learner_id=req.learner_id, topic=quiz.topic, difficulty=quiz.difficulty, score=score, total=total, answers_json=json.dumps(req.answers))
    db.add(attempt)
    db.commit()
    db.refresh(attempt)
    return {"id": attempt.id, "topic": attempt.topic, "score": attempt.score, "total": attempt.total, "difficulty": attempt.difficulty, "created_at": attempt.created_at.isoformat()}


@app.post("/api/assessment/submit", response_model=AttemptOut)
def submit_assessment(req: SubmitRequest, db: Session = Depends(get_db)):
    return submit(req, db)


@app.get("/api/dashboard/{learner_id}")
def dashboard(learner_id: int, db: Session = Depends(get_db)):
    learner = db.get(Learner, learner_id)
    if not learner:
        raise HTTPException(404, "Learner not found")
    attempts = db.query(Attempt).filter(Attempt.learner_id == learner_id).order_by(Attempt.created_at.desc()).all()
    topic_scores = {}
    for attempt in attempts:
        topic_scores.setdefault(attempt.topic, []).append(attempt.score)
    mastery = {topic: round(sum(scores) / len(scores), 1) for topic, scores in topic_scores.items()}
    weak = sorted(mastery.items(), key=lambda item: item[1])[:3]
    recommendations = [{"topic": topic, "reason": f"Mastery is {score}%. Practice this topic next.", "priority": "high"} for topic, score in weak if score < 70]
    if not recommendations:
        recommendations = [{"topic": "Python", "reason": "Start a diagnostic assessment to build your learning profile.", "priority": "medium"}]
    topics_list = [item.name for item in db.query(Subject).filter(Subject.active.is_(True)).order_by(Subject.name).all()]
    learning_path = [{"topic": topic, "mastery": mastery.get(topic), "action": "Advanced practice" if mastery.get(topic, 0) >= 80 else "Revision and guided practice" if mastery.get(topic, 0) < 50 else "Focused practice"} for topic in topics_list]
    weak_topics = [topic for topic, score in mastery.items() if score < 70]
    strong_topics = [topic for topic, score in mastery.items() if score >= 80]
    return {"learner": {"id": learner.id, "name": learner.name}, "attempts": len(attempts), "average_score": round(sum(a.score for a in attempts) / len(attempts), 1) if attempts else 0, "mastery": mastery, "weak_topics": weak_topics, "strong_topics": strong_topics, "learning_path": learning_path, "recommendations": recommendations, "recent": [{"topic": a.topic, "score": a.score, "difficulty": a.difficulty, "date": a.created_at.isoformat()} for a in attempts[:8]]}


@app.get("/api/learners/{learner_id}/progress")
def progress(learner_id: int, db: Session = Depends(get_db)):
    data = dashboard(learner_id, db)
    return {"learner": data["learner"], "mastery": data["mastery"], "recent": data["recent"]}


@app.get("/api/learners/{learner_id}/analytics")
def analytics(learner_id: int, db: Session = Depends(get_db)):
    data = dashboard(learner_id, db)
    return {"attempts": data["attempts"], "average_score": data["average_score"], "mastery": data["mastery"]}


@app.get("/api/learners/{learner_id}/recommendations")
def recommendations(learner_id: int, db: Session = Depends(get_db)):
    return {"recommendations": dashboard(learner_id, db)["recommendations"]}


@app.post("/api/materials/upload")
async def upload_material(learner_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not db.get(Learner, learner_id):
        raise HTTPException(404, "Learner not found")
    raw = await file.read()
    try:
        text_content = extract_text(file.filename or "material.txt", raw, file.content_type or "")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    material = Material(learner_id=learner_id, title=(file.filename or "Learning material")[:200], filename=file.filename or "material", mime_type=file.content_type or "application/octet-stream", source_type="upload", extracted_text=text_content)
    db.add(material)
    db.commit()
    db.refresh(material)
    return {"id": material.id, "title": material.title, "filename": material.filename, "source_type": material.source_type, "characters": len(text_content)}


@app.post("/api/materials/igot")
async def import_igot(learner_id: int, url: str, db: Session = Depends(get_db)):
    if not db.get(Learner, learner_id):
        raise HTTPException(404, "Learner not found")
    try:
        title, text_content, content_type = await import_igot_resource(url)
    except Exception as exc:
        raise HTTPException(400, f"Could not import the public iGOT resource: {exc}")
    material = Material(learner_id=learner_id, title=title, filename=title, mime_type=content_type, source_type="igot", source_url=url, extracted_text=text_content)
    db.add(material)
    db.commit()
    db.refresh(material)
    return {"id": material.id, "title": material.title, "source_type": "igot", "source_url": material.source_url, "characters": len(text_content)}


@app.get("/api/materials/{learner_id}")
def list_materials(learner_id: int, db: Session = Depends(get_db)):
    if not db.get(Learner, learner_id):
        raise HTTPException(404, "Learner not found")
    materials = db.query(Material).filter(Material.learner_id == learner_id).order_by(Material.created_at.desc()).all()
    return {"materials": [{"id": item.id, "title": item.title, "filename": item.filename, "source_type": item.source_type, "source_url": item.source_url, "characters": len(item.extracted_text), "created_at": item.created_at.isoformat()} for item in materials]}


@app.post("/api/materials/{material_id}/quiz", response_model=QuizOut)
async def material_quiz(material_id: int, req: QuizRequest, db: Session = Depends(get_db)):
    material = db.get(Material, material_id)
    if not material or material.learner_id != req.learner_id:
        raise HTTPException(404, "Learning material not found")
    provider = get_provider()
    if provider.name != "gemini":
        raise HTTPException(503, "Material-based MCQ generation requires the Gemini provider. Configure GEMINI_API_KEY and try again.")
    mastery = learner_mastery(req.learner_id, db)
    difficulty = req.difficulty
    if difficulty == "adaptive":
        score = mastery.get(req.topic)
        difficulty = "hard" if score is not None and score >= 80 else "medium" if score is not None and score >= 50 else "easy"
    previous = [row.question for row in db.query(GeneratedQuestion).filter(GeneratedQuestion.learner_id == req.learner_id, GeneratedQuestion.topic == req.topic).order_by(GeneratedQuestion.created_at.desc()).limit(30).all()]
    source = retrieve_material_context(material.extracted_text, req.topic or material.title)
    if not source:
        raise HTTPException(400, "The material has no readable content.")
    questions = []
    excluded = previous[:]
    for _ in range(3):
        remaining = req.count - len(questions)
        candidates = await generate_material_questions(source, req.subject or req.topic or material.title, req.topic or material.title, difficulty, max(remaining * 2, 6), excluded)
        for question in candidates or []:
            if any(question_similarity(question["question"], old) >= 0.9 for old in excluded + [item["question"] for item in questions]):
                continue
            question["id"] = f"gemini-{uuid.uuid4().hex[:16]}"
            questions.append(question)
            if len(questions) == req.count:
                break
        excluded.extend([item["question"] for item in questions])
        if len(questions) == req.count:
            break
    if len(questions) < req.count:
        raise HTTPException(502, "Gemini could not produce enough new material-grounded questions. Try a shorter or clearer source document.")
    quiz = store_questions(db, req.learner_id, questions, req.topic or material.title, difficulty)
    return {"id": quiz.id, "topic": req.topic or material.title, "difficulty": difficulty, "questions": questions}


def tutor_context(learner_id: int, topic: str, db: Session, material_id: int | None = None):
    learner = db.get(Learner, learner_id)
    if not learner:
        raise HTTPException(404, "Learner not found")
    mastery = learner_mastery(learner_id, db)
    weak = [name for name, score in mastery.items() if score < 70]
    strong = [name for name, score in mastery.items() if score >= 80]
    recent = db.query(Attempt).filter(Attempt.learner_id == learner_id).order_by(Attempt.created_at.desc()).limit(5).all()
    recent_questions = db.query(GeneratedQuestion).filter(GeneratedQuestion.learner_id == learner_id).order_by(GeneratedQuestion.created_at.desc()).limit(5).all()
    recent_quiz = db.query(Quiz).filter(Quiz.learner_id == learner_id, Quiz.topic == topic).order_by(Quiz.created_at.desc()).first()
    mistakes = db.query(LearnerQuestionHistory).filter(LearnerQuestionHistory.learner_id == learner_id, LearnerQuestionHistory.correct.is_(False)).order_by(LearnerQuestionHistory.seen_at.desc()).limit(5).all()
    mistake_questions = [db.get(GeneratedQuestion, item.question_id) for item in mistakes]
    context = {"learner_name": learner.name, "topic": topic, "mastery": mastery, "weak_topics": weak, "strong_topics": strong, "learning_path": [{"topic": name, "mastery": mastery.get(name), "action": "revision" if mastery.get(name, 0) < 50 else "focused practice" if mastery.get(name, 0) < 80 else "advanced practice"} for name in weak + strong], "recent_results": [{"topic": a.topic, "score": a.score, "difficulty": a.difficulty} for a in recent], "recent_mistakes": [item.subtopic for item in mistake_questions if item], "recent_questions": [q.question for q in recent_questions], "recent_question_details": [{"question": q.question, "explanation": q.explanation, "subtopic": q.subtopic} for q in recent_questions], "recent_question_types": [q.question_type for q in recent_questions], "recent_quiz_difficulty": recent_quiz.difficulty if recent_quiz else None}
    if material_id:
        material = db.get(Material, material_id)
        if not material or material.learner_id != learner_id:
            raise HTTPException(404, "Learning material not found")
        context["material_title"] = material.title
        context["material_context"] = retrieve_material_context(material.extracted_text, topic or material.title)
    return context


@app.post("/api/chat", response_model=ChatOut)
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    learner = db.get(Learner, req.learner_id)
    if not learner:
        raise HTTPException(404, "Learner not found")
    conversation = db.get(Conversation, req.conversation_id) if req.conversation_id else None
    if conversation and conversation.learner_id != req.learner_id:
        raise HTTPException(403, "Conversation does not belong to learner")
    if not conversation:
        conversation = Conversation(learner_id=req.learner_id, topic=req.topic)
        db.add(conversation)
        db.flush()
    topic = req.topic or conversation.topic or "Python"
    conversation.topic = topic
    db.add(Message(conversation_id=conversation.id, role="user", content=req.message))
    context = tutor_context(req.learner_id, topic, db, req.material_id)
    conversation_messages = db.query(Message).filter(Message.conversation_id == conversation.id).order_by(Message.created_at.desc()).limit(12).all()
    context["conversation"] = [{"role": item.role, "content": item.content} for item in reversed(conversation_messages)]
    lower = req.message.lower()
    quiz = None
    response_provider = "fallback"
    quiz_intent = any(phrase in lower for phrase in ("test me", "quiz me", "mcq", "ask me", "give me questions", "create a quiz", "generate questions", "make them harder", "make them easier", "make them intermediate")) or ("another" in lower and bool(re.search(r"\b\d+\b", lower)))
    if quiz_intent:
        count_match = re.search(r"\b(\d+)\b", lower)
        count = min(max(int(count_match.group(1)) if count_match else 5, 3), 20)
        selected_topic = next((candidate for candidate in SUBJECTS if candidate.lower() in lower), topic)
        if "weak" in lower and context["weak_topics"]:
            selected_topic = context["weak_topics"][0]
        difficulty = "hard" if any(term in lower for term in ("hard", "difficult")) else "easy" if any(term in lower for term in ("easy", "easier", "beginner")) else "medium" if any(term in lower for term in ("medium", "intermediate")) else (context.get("recent_quiz_difficulty") if "another" in lower else "adaptive") or "adaptive"
        if req.material_id:
            material = db.get(Material, req.material_id)
            if not material or material.learner_id != req.learner_id:
                raise HTTPException(404, "Learning material not found")
            material_req = QuizRequest(learner_id=req.learner_id, topic=selected_topic, subject=selected_topic, difficulty=difficulty, count=count)
            quiz = await material_quiz(req.material_id, material_req, db)
        else:
            quiz = await generate_quiz(QuizRequest(learner_id=req.learner_id, topic=selected_topic, subject=selected_topic, difficulty=difficulty, count=count), db)
        reply = f"I prepared {count} new {selected_topic} questions at {quiz['difficulty']} difficulty." + (" They are grounded in your selected learning material." if req.material_id else " They avoid previously shown questions.")
        response_provider = quiz["questions"][0].get("provider", "fallback") if quiz.get("questions") else "fallback"
    else:
        provider = get_provider()
        provider_reply = await provider.chat(req.message, context)
        if provider_reply:
            reply = provider_reply
            response_provider = provider.name
        else:
            fallback = FallbackProvider()
            reply = await fallback.chat(req.message, context) or "I could not reach the configured AI provider."
            response_provider = fallback.name
    assistant = Message(conversation_id=conversation.id, role="assistant", content=reply)
    db.add(assistant)
    db.commit()
    db.refresh(assistant)
    return {"conversation_id": conversation.id, "message": {"id": assistant.id, "role": assistant.role, "content": assistant.content, "created_at": assistant.created_at.isoformat()}, "quiz": quiz, "provider": response_provider}
