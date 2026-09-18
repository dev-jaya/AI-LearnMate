from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
import json
import logging
import math
import re
import uuid
import random
import time

from .db import Base, engine, get_db
from .config import settings
from .models import (
    Learner,
    Attempt,
    Quiz,
    Conversation,
    Message,
    GeneratedQuestion,
    QuestionFingerprint,
    LearnerQuestionHistory,
    Subject,
    Topic,
    QuizQuestion,
    MaterialChunk,
)
from .materials import (
    Material,
    extract_segments,
    material_chunk_records,
    extract_text,
    retrieve_material_context,
    import_igot_resource,
)
from .schemas import *
from .ai_providers import SUBJECTS, get_provider, question_similarity

logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)
app = FastAPI(title="AI LearnMate API", version="1.4.0")


def configured_cors_origins() -> list[str]:
    defaults = [
        "http://localhost:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5500",
        "https://ai-learnmate-frontend.onrender.com",
    ]
    raw = settings.frontend_origins
    return sorted({origin.strip().rstrip("/") for origin in (raw.split(",") + defaults) if origin.strip()})


app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def migrate_database():
    tables = set(inspect(engine).get_table_names())
    additions = {
        "generated_questions": {
            "normalized_question": "TEXT NOT NULL DEFAULT ''",
            "question_type": "VARCHAR(40) NOT NULL DEFAULT 'conceptual'",
            "material_id": "INTEGER",
        },
        "quizzes": {"material_id": "INTEGER"},
        "conversations": {"gemini_interaction_id": "VARCHAR(160) DEFAULT ''"},
    }
    with engine.begin() as connection:
        for table, columns in additions.items():
            if table not in tables:
                continue
            existing = {item["name"] for item in inspect(engine).get_columns(table)}
            for name, definition in columns.items():
                if name not in existing:
                    connection.execute(
                        text(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {definition}')
                    )

    with Session(bind=engine) as db:
        duplicate_groups = (
            db.query(
                LearnerQuestionHistory.learner_id,
                LearnerQuestionHistory.fingerprint,
            )
            .group_by(
                LearnerQuestionHistory.learner_id,
                LearnerQuestionHistory.fingerprint,
            )
            .having(text("COUNT(*) > 1"))
            .all()
        )
        for learner_id, fingerprint in duplicate_groups:
            rows = (
                db.query(LearnerQuestionHistory)
                .filter(
                    LearnerQuestionHistory.learner_id == learner_id,
                    LearnerQuestionHistory.fingerprint == fingerprint,
                )
                .order_by(LearnerQuestionHistory.id.asc())
                .all()
            )
            for duplicate in rows[1:]:
                db.delete(duplicate)
        db.commit()

    with engine.begin() as connection:
        connection.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_question_history_learner_fingerprint "
            "ON learner_question_history (learner_id, fingerprint)"
        ))


migrate_database()


def seed_subject_catalog():
    with Session(bind=engine) as db:
        for subject_name in SUBJECTS:
            if not db.query(Subject).filter(Subject.name == subject_name).first():
                db.add(Subject(name=subject_name))
        db.commit()


seed_subject_catalog()


@app.get("/")
def root():
    return {"name": "AI LearnMate", "status": "running", "version": "1.4.0"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-learnmate-api", "version": "1.4.0"}


@app.get("/health/ai")
def ai_health():
    provider = get_provider()
    configured = bool(provider.api_key)
    return {
        "provider": provider.name,
        "model": provider.model,
        "configured": configured,
        "status": "configured" if configured else "missing-key",
        "note": "Configuration only; no Gemini request is made.",
    }


@app.get("/health/ai/probe")
async def ai_probe():
    provider = get_provider()
    if not provider.api_key:
        raise HTTPException(503, "Gemini is not configured on the server.")
    started = time.perf_counter()
    reply = await provider.chat(
        "Reply with exactly OK.",
        {"learner_name": "health probe", "conversation": []},
    )
    duration_ms = round((time.perf_counter() - started) * 1000)
    if not reply:
        raise safe_gemini_failure(provider, "Gemini health probe failed.")
    return {
        "status": "ok",
        "provider": provider.name,
        "model": provider.model,
        "response": reply,
        "duration_ms": duration_ms,
    }


@app.get("/api/igot")
def igot_info():
    return {
        "platform": "iGOT Karmayogi",
        "official_url": "https://igotkarmayogi.gov.in/",
        "public_content_url": "https://portal.igotkarmayogi.gov.in/",
        "integration_mode": "official public-resource import",
        "note": "Paste a public iGOT Karmayogi resource URL or upload official learning material.",
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
    records = (
        db.query(Subject)
        .filter(Subject.active.is_(True))
        .order_by(Subject.name)
        .all()
    )
    return {
        "topics": [record.name for record in records],
        "subjects": [
            {
                "name": record.name,
                "topics": [topic.name for topic in record.topics if topic.active],
            }
            for record in records
        ],
    }


def learner_mastery(learner_id: int, db: Session):
    attempts = (
        db.query(Attempt)
        .filter(Attempt.learner_id == learner_id)
        .order_by(Attempt.created_at.desc())
        .all()
    )
    scores: dict[str, list[float]] = {}
    for attempt in attempts:
        scores.setdefault(attempt.topic, []).append(attempt.score)
    return {
        topic: round(sum(values) / len(values), 1)
        for topic, values in scores.items()
    }


def adaptive_difficulty_plan(score: float | None, count: int) -> list[str]:
    """Return an intentionally simple progression plan based on observed mastery."""
    count = max(0, int(count))
    if count == 0:
        return []

    if score is None:
        easy = max(1, math.ceil(count * 0.4))
        medium = max(1, math.ceil(count * 0.4))
        hard = max(0, count - easy - medium)
    elif score < 50:
        easy = max(1, math.ceil(count * 0.6))
        medium = max(0, count - easy)
        hard = 0
    elif score < 80:
        easy = max(1, math.ceil(count * 0.2))
        medium = max(1, math.ceil(count * 0.4))
        hard = max(0, count - easy - medium)
    else:
        easy = 0
        medium = max(1, math.ceil(count * 0.35))
        hard = max(0, count - medium)

    plan = ["easy"] * easy + ["medium"] * medium + ["hard"] * hard
    return plan[:count]


def infer_subject(message: str, default: str) -> str:
    """Match an explicitly named subject without C matching arbitrary English words."""
    lower = message.casefold()
    for subject in sorted(SUBJECTS, key=len, reverse=True):
        if subject == "C":
            pattern = r"\bc\b"
        elif subject == "C++":
            pattern = r"(?<!\w)c\+\+(?!\w)"
        else:
            pattern = rf"(?<!\w){re.escape(subject.casefold())}(?!\w)"
        if re.search(pattern, lower):
            return subject
    return default


def safe_gemini_failure(provider, default_message: str) -> HTTPException:
    status = provider.last_status_code
    detail = (provider.last_error or "").strip()
    category = getattr(provider, "last_error_category", "")

    if category == "configuration" or not provider.api_key:
        return HTTPException(
            503,
            "Gemini is not configured on the server. Set GEMINI_API_KEY in Render/server environment settings.",
        )
    if category == "authentication" or status in {401, 403}:
        return HTTPException(
            503,
            "Gemini authentication failed. Check the server-side GEMINI_API_KEY and Google API authorization.",
        )
    if category == "quota" or status == 429:
        return HTTPException(
            429,
            "Gemini rate/quota limit reached. Wait and retry later, or check the Gemini project quota.",
        )
    if category == "model_not_found" or status == 404:
        return HTTPException(
            502,
            f"Gemini model '{provider.model}' was rejected by the API.",
        )
    if category == "blocked_response":
        return HTTPException(502, "Gemini blocked the response for safety reasons.")
    if category == "bad_request" or status in {400, 422}:
        safe_detail = detail.replace(provider.api_key or "", "[REDACTED]")
        return HTTPException(502, f"Gemini rejected the request. {safe_detail[:300]}")
    if category == "timeout" or status == 504:
        return HTTPException(504, "Gemini took too long to respond. Please try again.")
    if category == "network":
        return HTTPException(503, "The AI service could not be reached. Please try again.")
    if category == "service_unavailable" or status in {500, 502, 503}:
        return HTTPException(503, "Gemini is temporarily unavailable. Please try again.")
    if detail:
        return HTTPException(502, f"{default_message} Details: {detail[:300]}")
    return HTTPException(502, default_message)


def shuffle_question_options(question: dict) -> dict:
    """Randomize answer positions without changing the correct answer."""
    options = list(question["options"])
    correct_index = int(question["answer"])
    indexed = list(enumerate(options))
    random.SystemRandom().shuffle(indexed)

    question["options"] = [value for _, value in indexed]
    question["answer"] = next(
        index for index, (old_index, _) in enumerate(indexed)
        if old_index == correct_index
    )
    question["correct_answer"] = question["answer"]
    return question


def store_questions(
    db: Session,
    learner_id: int,
    questions: list[dict],
    topic: str,
    difficulty: str,
    material_id: int | None = None,
) -> Quiz:
    quiz = Quiz(
        learner_id=learner_id,
        material_id=material_id,
        topic=topic,
        difficulty=difficulty,
        questions_json=json.dumps(questions),
    )
    db.add(quiz)
    db.flush()

    for position, question in enumerate(questions):
        duplicate = (
            db.query(LearnerQuestionHistory)
            .filter(
                LearnerQuestionHistory.learner_id == learner_id,
                LearnerQuestionHistory.fingerprint == question["fingerprint"],
            )
            .first()
        )
        if duplicate:
            db.rollback()
            raise HTTPException(
                409,
                "A duplicate question was detected while saving the quiz. Please regenerate.",
            )

        stored = GeneratedQuestion(
            public_id=question["id"],
            learner_id=learner_id,
            material_id=material_id,
            subject=question["subject"],
            topic=question["topic"],
            subtopic=question["subtopic"],
            difficulty=question["difficulty"],
            question=question["question"],
            normalized_question=re.sub(
                r"[^a-z0-9]+", " ", question["question"].lower()
            ).strip(),
            options_json=json.dumps(question["options"]),
            answer=question["answer"],
            explanation=question["explanation"],
            fingerprint=question["fingerprint"],
            provider="gemini",
            question_type=question.get("question_type", "conceptual"),
        )
        db.add(stored)
        db.flush()
        db.add(
            QuestionFingerprint(
                fingerprint=question["fingerprint"],
                question_id=stored.id,
            )
        )
        db.add(
            LearnerQuestionHistory(
                learner_id=learner_id,
                question_id=stored.id,
                fingerprint=question["fingerprint"],
            )
        )
        db.add(
            QuizQuestion(
                quiz_id=quiz.id,
                question_id=stored.id,
                position=position,
            )
        )

    db.commit()
    db.refresh(quiz)
    return quiz


def _previous_questions(
    learner_id: int,
    topic: str,
    db: Session,
    limit: int = 500,
) -> list[str]:
    rows = (
        db.query(GeneratedQuestion)
        .filter(GeneratedQuestion.learner_id == learner_id)
        .order_by(GeneratedQuestion.created_at.desc())
        .limit(limit)
        .all()
    )
    return [row.question for row in rows]


def _unique_candidates(
    candidates: list[dict] | None,
    previous: list[str],
    chosen: list[dict],
    desired_difficulty: str | None = None,
) -> list[dict]:
    accepted = []
    blocked = previous + [item["question"] for item in chosen]

    for candidate in candidates or []:
        if desired_difficulty and candidate.get("difficulty") != desired_difficulty:
            continue
        if any(
            question_similarity(candidate["question"], old) >= 0.9
            for old in blocked
        ):
            continue
        accepted.append(candidate)
        blocked.append(candidate["question"])
    return accepted


async def generate_quiz(req: QuizRequest, db: Session):
    if not db.get(Learner, req.learner_id):
        raise HTTPException(404, "Learner not found")

    mastery = learner_mastery(req.learner_id, db)
    score = mastery.get(req.topic)
    requested_difficulty = req.difficulty

    if requested_difficulty == "adaptive":
        difficulty_plan = adaptive_difficulty_plan(score, req.count)
        generation_difficulty = difficulty_plan[0] if difficulty_plan else "easy"
    else:
        difficulty_plan = [requested_difficulty] * req.count
        generation_difficulty = requested_difficulty

    previous = _previous_questions(req.learner_id, req.topic, db)
    provider = get_provider()

    if not provider.api_key:
        raise HTTPException(
            503,
            "Gemini is not configured on the server. Set GEMINI_API_KEY in Render/server environment settings.",
        )

    context = {
        "mastery": mastery,
        "weak_topics": [name for name, value in mastery.items() if value < 70],
        "strong_topics": [name for name, value in mastery.items() if value >= 80],
        "excluded_questions": previous,
        "difficulty_plan": difficulty_plan,
        "generation_variant": uuid.uuid4().hex[:10],
    }

    questions: list[dict] = []
    for _ in range(6):
        if len(questions) >= req.count:
            break

        remaining = req.count - len(questions)
        missing_plan = difficulty_plan[len(questions):]
        request_count = min(max(remaining * 3, 10), 20)

        candidates = await provider.generate_questions(
            req.subject or req.topic,
            req.topic,
            req.subtopic,
            generation_difficulty,
            request_count,
            {**context, "difficulty_plan": missing_plan},
        )

        for desired in missing_plan:
            selected = _unique_candidates(
                candidates,
                previous,
                questions,
                desired_difficulty=desired,
            )
            if selected:
                question = selected[0]
                question["id"] = f"gemini-{uuid.uuid4().hex[:16]}"
                question["provider"] = "gemini"
                questions.append(shuffle_question_options(question))
                if len(questions) == req.count:
                    break

        if len(questions) < req.count:
            extras = _unique_candidates(
                candidates,
                previous,
                questions,
                desired_difficulty=None if requested_difficulty == "adaptive" else generation_difficulty,
            )
            for question in extras:
                question["id"] = f"gemini-{uuid.uuid4().hex[:16]}"
                question["provider"] = "gemini"
                questions.append(shuffle_question_options(question))
                if len(questions) == req.count:
                    break

        context["excluded_questions"] = previous + [
            q["question"] for q in questions
        ]
        generation_difficulty = (
            difficulty_plan[len(questions)]
            if len(questions) < len(difficulty_plan)
            else generation_difficulty
        )

    if len(questions) < req.count:
        raise safe_gemini_failure(
            provider,
            f"Gemini produced only {len(questions)} validated new questions.",
        )

    quiz = store_questions(
        db,
        req.learner_id,
        questions,
        req.topic,
        "adaptive" if requested_difficulty == "adaptive" else generation_difficulty,
    )
    return {
        "id": quiz.id,
        "topic": req.topic,
        "difficulty": "adaptive" if requested_difficulty == "adaptive" else generation_difficulty,
        "questions": questions,
    }


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

    correct = sum(
        answer == question["answer"]
        for answer, question in zip(req.answers, questions)
    )
    quiz.submitted = True

    for answer, question in zip(req.answers, questions):
        stored = (
            db.query(GeneratedQuestion)
            .filter(GeneratedQuestion.public_id == question["id"])
            .first()
        )
        if stored:
            row = (
                db.query(LearnerQuestionHistory)
                .filter(
                    LearnerQuestionHistory.learner_id == req.learner_id,
                    LearnerQuestionHistory.question_id == stored.id,
                )
                .order_by(LearnerQuestionHistory.seen_at.desc())
                .first()
            )
            if row:
                row.answered = True
                row.correct = answer == question["answer"]

    score = round(correct * 100 / len(questions), 2)
    attempt = Attempt(
        learner_id=req.learner_id,
        topic=quiz.topic,
        difficulty=quiz.difficulty,
        score=score,
        total=len(questions),
        answers_json=json.dumps(req.answers),
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)

    return {
        "id": attempt.id,
        "topic": attempt.topic,
        "score": attempt.score,
        "total": attempt.total,
        "difficulty": attempt.difficulty,
        "created_at": attempt.created_at.isoformat(),
    }


@app.post("/api/assessment/submit", response_model=AttemptOut)
def submit_assessment(req: SubmitRequest, db: Session = Depends(get_db)):
    return submit(req, db)


@app.get("/api/dashboard/{learner_id}")
def dashboard(learner_id: int, db: Session = Depends(get_db)):
    learner = db.get(Learner, learner_id)
    if not learner:
        raise HTTPException(404, "Learner not found")

    attempts = (
        db.query(Attempt)
        .filter(Attempt.learner_id == learner_id)
        .order_by(Attempt.created_at.desc())
        .all()
    )
    mastery = learner_mastery(learner_id, db)
    weak = sorted(mastery.items(), key=lambda item: item[1])[:3]
    recommendations = [
        {
            "topic": topic,
            "reason": f"Mastery is {score}%. Practice this topic next.",
            "priority": "high",
        }
        for topic, score in weak
        if score < 70
    ] or [
        {
            "topic": "Python",
            "reason": "Start a diagnostic assessment to build your learning profile.",
            "priority": "medium",
        }
    ]

    names = [
        item.name
        for item in db.query(Subject)
        .filter(Subject.active.is_(True))
        .order_by(Subject.name)
        .all()
    ]

    return {
        "learner": {"id": learner.id, "name": learner.name},
        "attempts": len(attempts),
        "average_score": round(
            sum(attempt.score for attempt in attempts) / len(attempts), 1
        )
        if attempts
        else 0,
        "mastery": mastery,
        "weak_topics": [
            topic for topic, score in mastery.items() if score < 70
        ],
        "strong_topics": [
            topic for topic, score in mastery.items() if score >= 80
        ],
        "learning_path": [
            {
                "topic": topic,
                "mastery": mastery.get(topic),
                "action": (
                    "Advanced practice"
                    if mastery.get(topic, 0) >= 80
                    else "Revision and guided practice"
                    if mastery.get(topic, 0) < 50
                    else "Focused practice"
                ),
            }
            for topic in names
        ],
        "recommendations": recommendations,
        "recent": [
            {
                "topic": attempt.topic,
                "score": attempt.score,
                "difficulty": attempt.difficulty,
                "date": attempt.created_at.isoformat(),
            }
            for attempt in attempts[:8]
        ],
    }


@app.get("/api/learners/{learner_id}/progress")
def progress(learner_id: int, db: Session = Depends(get_db)):
    data = dashboard(learner_id, db)
    return {
        "learner": data["learner"],
        "mastery": data["mastery"],
        "recent": data["recent"],
    }


@app.get("/api/learners/{learner_id}/analytics")
def analytics(learner_id: int, db: Session = Depends(get_db)):
    data = dashboard(learner_id, db)
    return {
        "attempts": data["attempts"],
        "average_score": data["average_score"],
        "mastery": data["mastery"],
    }


@app.get("/api/learners/{learner_id}/recommendations")
def recommendations(learner_id: int, db: Session = Depends(get_db)):
    return {"recommendations": dashboard(learner_id, db)["recommendations"]}


@app.post("/api/materials/upload")
async def upload_material(
    learner_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not db.get(Learner, learner_id):
        raise HTTPException(404, "Learner not found")

    raw = await file.read()
    try:
        segments = extract_segments(
            file.filename or "material.txt",
            raw,
            file.content_type or "",
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    content = "\n".join(text for _ref, text in segments)
    material = Material(
        learner_id=learner_id,
        title=(file.filename or "Learning material")[:200],
        filename=file.filename or "material",
        mime_type=file.content_type or "application/octet-stream",
        source_type="upload",
        extracted_text=content,
    )
    db.add(material)
    db.flush()
    chunk_rows = [
        MaterialChunk(
            material_id=material.id,
            chunk_index=item["chunk_index"],
            source_ref=item["source_ref"],
            content=item["content"],
        )
        for item in material_chunk_records(segments)
    ]
    db.add_all(chunk_rows)
    db.commit()
    db.refresh(material)
    return {
        "id": material.id,
        "title": material.title,
        "filename": material.filename,
        "source_type": material.source_type,
        "characters": len(content),
        "chunks": len(chunk_rows),
    }


@app.post("/api/materials/igot")
async def import_igot(
    learner_id: int,
    url: str,
    db: Session = Depends(get_db),
):
    if not db.get(Learner, learner_id):
        raise HTTPException(404, "Learner not found")
    try:
        title, content, ctype = await import_igot_resource(url)
    except Exception as exc:
        raise HTTPException(
            400,
            f"Could not import the public iGOT resource: {type(exc).__name__}.",
        ) from exc

    material = Material(
        learner_id=learner_id,
        title=title,
        filename=title,
        mime_type=ctype,
        source_type="igot",
        source_url=url,
        extracted_text=content,
    )
    db.add(material)
    db.flush()
    chunk_rows = [
        MaterialChunk(
            material_id=material.id,
            chunk_index=item["chunk_index"],
            source_ref=item["source_ref"],
            content=item["content"],
        )
        for item in material_chunk_records([("iGOT resource", content)])
    ]
    db.add_all(chunk_rows)
    db.commit()
    db.refresh(material)
    return {
        "id": material.id,
        "title": material.title,
        "source_type": "igot",
        "source_url": material.source_url,
        "characters": len(content),
        "chunks": len(chunk_rows),
    }


@app.get("/api/materials/{learner_id}")
def list_materials(learner_id: int, db: Session = Depends(get_db)):
    if not db.get(Learner, learner_id):
        raise HTTPException(404, "Learner not found")

    items = (
        db.query(Material)
        .filter(Material.learner_id == learner_id)
        .order_by(Material.created_at.desc())
        .all()
    )
    return {
        "materials": [
            {
                "id": item.id,
                "title": item.title,
                "filename": item.filename,
                "source_type": item.source_type,
                "source_url": item.source_url,
                "characters": len(item.extracted_text),
                "created_at": item.created_at.isoformat(),
            }
            for item in items
        ]
    }



def ensure_material_chunks(material: Material, db: Session) -> list[MaterialChunk]:
    chunks = (
        db.query(MaterialChunk)
        .filter(MaterialChunk.material_id == material.id)
        .order_by(MaterialChunk.chunk_index.asc())
        .all()
    )
    if chunks:
        return chunks

    records = material_chunk_records([("document", material.extracted_text)])
    chunks = [
        MaterialChunk(
            material_id=material.id,
            chunk_index=item["chunk_index"],
            source_ref=item["source_ref"],
            content=item["content"],
        )
        for item in records
    ]
    db.add_all(chunks)
    db.commit()
    for chunk in chunks:
        db.refresh(chunk)
    return chunks


def choose_material_coverage_groups(
    chunks: list[MaterialChunk],
    requested_count: int,
) -> list[str]:
    """Partition the complete indexed document into coverage bands."""
    if not chunks:
        return []

    target_groups = min(max(requested_count, 4), 8, len(chunks))
    groups = []
    band_size = max(1, math.ceil(len(chunks) / target_groups))

    for band_start in range(0, len(chunks), band_size):
        band = chunks[band_start:band_start + band_size]
        if not band:
            continue
        source = "\n\n".join(
            f"[{item.source_ref} | chunk {item.chunk_index + 1}]\n{item.content}"
            for item in band
        )
        # Keep each coverage request comfortably below the per-call material
        # context budget while still representing every chunk in the band.
        groups.append(source[:30000])

    return groups[:target_groups]


async def _generate_material_quiz(
    material: Material,
    req: QuizRequest,
    db: Session,
):
    provider = get_provider()
    if not provider.api_key:
        raise HTTPException(
            503,
            "Gemini is not configured on the server. Set GEMINI_API_KEY in Render/server environment settings.",
        )

    mastery = learner_mastery(req.learner_id, db)
    score = mastery.get(req.topic)
    difficulty = req.difficulty
    if difficulty == "adaptive":
        difficulty_plan = adaptive_difficulty_plan(score, req.count)
    else:
        difficulty_plan = [difficulty] * req.count

    previous = _previous_questions(req.learner_id, req.topic or material.title, db)
    chunks = ensure_material_chunks(material, db)
    groups = choose_material_coverage_groups(chunks, req.count)
    if not groups:
        raise HTTPException(400, "The material has no readable indexed content.")

    questions: list[dict] = []
    started = time.perf_counter()

    for group_index, source_group in enumerate(groups):
        if len(questions) >= req.count:
            break

        remaining = req.count - len(questions)
        planned_difficulty = (
            difficulty_plan[min(len(questions), len(difficulty_plan) - 1)]
            if difficulty_plan
            else "medium"
        )
        candidate_count = min(max(remaining, 2), 6)
        candidates = await provider.generate_material_questions(
            source_group,
            req.subject or req.topic or material.title,
            req.topic or material.title,
            planned_difficulty,
            candidate_count,
            previous + [q["question"] for q in questions],
            source_reference=f"coverage group {group_index + 1} of {len(groups)}",
        )

        selected = _unique_candidates(
            candidates,
            previous,
            questions,
            desired_difficulty=planned_difficulty if difficulty != "adaptive" else None,
        )
        for question in selected:
            if len(questions) >= req.count:
                break
            question["id"] = f"gemini-{uuid.uuid4().hex[:16]}"
            question["provider"] = "gemini"
            questions.append(shuffle_question_options(question))

    # A final gap-fill pass can use any remaining document chunks. This is still
    # source-grounded and globally deduplicated, and avoids returning fewer
    # questions simply because one coverage group produced unusable candidates.
    if len(questions) < req.count:
        remaining_text = "\n\n--- DOCUMENT COVERAGE CHUNK ---\n\n".join(
            item.content
            for item in chunks
            if item.chunk_index % max(1, len(chunks) // max(1, req.count)) == 0
        )
        candidates = await provider.generate_material_questions(
            remaining_text[:180000],
            req.subject or req.topic or material.title,
            req.topic or material.title,
            difficulty_plan[len(questions)] if len(questions) < len(difficulty_plan) else "medium",
            min(max((req.count - len(questions)) * 2, 4), 10),
            previous + [q["question"] for q in questions],
            source_reference="document-wide gap fill",
        )
        selected = _unique_candidates(
            candidates,
            previous,
            questions,
            desired_difficulty=None if difficulty == "adaptive" else difficulty,
        )
        for question in selected:
            if len(questions) >= req.count:
                break
            question["id"] = f"gemini-{uuid.uuid4().hex[:16]}"
            question["provider"] = "gemini"
            questions.append(shuffle_question_options(question))

    duration_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "material_quiz material=%s learner=%s chunks=%s requested=%s accepted=%s duration_ms=%s",
        material.id,
        req.learner_id,
        len(chunks),
        req.count,
        len(questions),
        duration_ms,
    )

    if len(questions) < req.count:
        raise safe_gemini_failure(
            provider,
            f"Gemini produced {len(questions)} validated new material questions instead of {req.count}.",
        )

    final_questions = questions[: req.count]
    quiz = store_questions(
        db,
        req.learner_id,
        final_questions,
        req.topic or material.title,
        "adaptive" if req.difficulty == "adaptive" else difficulty,
        material_id=material.id,
    )
    return {
        "id": quiz.id,
        "topic": req.topic or material.title,
        "difficulty": "adaptive" if req.difficulty == "adaptive" else difficulty,
        "questions": final_questions,
    }


@app.post("/api/materials/{material_id}/quiz", response_model=QuizOut)
async def material_quiz(
    material_id: int,
    req: QuizRequest,
    db: Session = Depends(get_db),
):
    material = db.get(Material, material_id)
    if not material or material.learner_id != req.learner_id:
        raise HTTPException(404, "Learning material not found")
    return await _generate_material_quiz(material, req, db)


def tutor_context(
    learner_id: int,
    topic: str,
    db: Session,
    material_id: int | None = None,
):
    learner = db.get(Learner, learner_id)
    if not learner:
        raise HTTPException(404, "Learner not found")

    mastery = learner_mastery(learner_id, db)
    recent = (
        db.query(Attempt)
        .filter(Attempt.learner_id == learner_id)
        .order_by(Attempt.created_at.desc())
        .limit(5)
        .all()
    )
    mistakes = (
        db.query(LearnerQuestionHistory)
        .filter(
            LearnerQuestionHistory.learner_id == learner_id,
            LearnerQuestionHistory.correct.is_(False),
        )
        .order_by(LearnerQuestionHistory.seen_at.desc())
        .limit(5)
        .all()
    )
    mistake_questions = [
        db.get(GeneratedQuestion, item.question_id) for item in mistakes
    ]

    context = {
        "learner_name": learner.name,
        "topic": topic,
        "mastery": mastery,
        "weak_topics": [
            name for name, value in mastery.items() if value < 70
        ],
        "strong_topics": [
            name for name, value in mastery.items() if value >= 80
        ],
        "recent_results": [
            {
                "topic": attempt.topic,
                "score": attempt.score,
                "difficulty": attempt.difficulty,
            }
            for attempt in recent
        ],
        "recent_mistakes": [
            question.subtopic
            for question in mistake_questions
            if question
        ],
    }

    if material_id:
        material = db.get(Material, material_id)
        if not material or material.learner_id != learner_id:
            raise HTTPException(404, "Learning material not found")
        context["material_title"] = material.title
        context["material_context"] = retrieve_material_context(
            material.extracted_text,
            topic or material.title,
        )

    return context


@app.get("/api/conversations/{learner_id}")
def list_conversations(learner_id: int, db: Session = Depends(get_db)):
    if not db.get(Learner, learner_id):
        raise HTTPException(404, "Learner not found")

    rows = (
        db.query(Conversation)
        .filter(Conversation.learner_id == learner_id)
        .order_by(
            Conversation.updated_at.desc(),
            Conversation.created_at.desc(),
        )
        .all()
    )

    result = []
    for conversation in rows:
        first = (
            db.query(Message)
            .filter(
                Message.conversation_id == conversation.id,
                Message.role == "user",
            )
            .order_by(Message.created_at.asc())
            .first()
        )
        last = (
            db.query(Message)
            .filter(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.desc())
            .first()
        )
        result.append(
            {
                "id": conversation.id,
                "topic": conversation.topic,
                "title": first.content[:70]
                if first
                else conversation.topic or "New conversation",
                "preview": last.content[:100] if last else "",
                "created_at": conversation.created_at.isoformat(),
                "updated_at": (
                    conversation.updated_at.isoformat()
                    if conversation.updated_at
                    else conversation.created_at.isoformat()
                ),
            }
        )
    return {"conversations": result}


@app.get("/api/conversations/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: int,
    learner_id: int,
    db: Session = Depends(get_db),
):
    conversation = db.get(Conversation, conversation_id)
    if not conversation or conversation.learner_id != learner_id:
        raise HTTPException(404, "Conversation not found")

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return {
        "conversation_id": conversation_id,
        "topic": conversation.topic,
        "messages": [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            for message in messages
        ],
    }


@app.post("/api/chat", response_model=ChatOut)
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    if not db.get(Learner, req.learner_id):
        raise HTTPException(404, "Learner not found")

    conversation = (
        db.get(Conversation, req.conversation_id)
        if req.conversation_id
        else None
    )
    if conversation and conversation.learner_id != req.learner_id:
        raise HTTPException(403, "Conversation does not belong to learner")

    if not conversation:
        conversation = Conversation(
            learner_id=req.learner_id,
            topic=req.topic,
        )
        db.add(conversation)
        db.flush()

    topic = req.topic or conversation.topic or "General"
    conversation.topic = topic

    db.add(
        Message(
            conversation_id=conversation.id,
            role="user",
            content=req.message,
        )
    )
    db.flush()

    context = tutor_context(
        req.learner_id,
        topic,
        db,
        req.material_id,
    )
    context["gemini_interaction_id"] = conversation.gemini_interaction_id or None
    history = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.desc())
        .limit(12)
        .all()
    )
    context["conversation"] = [
        {"role": message.role, "content": message.content}
        for message in reversed(history)
    ]

    lower = req.message.lower()
    quiz_intent = any(
        phrase in lower
        for phrase in (
            "test me",
            "quiz me",
            "mcq",
            "ask me",
            "give me questions",
            "create a quiz",
            "generate questions",
        )
    )

    quiz = None
    if quiz_intent:
        match = re.search(r"\b(\d+)\b", lower)
        count = min(max(int(match.group(1)) if match else 5, 3), 20)
        selected = infer_subject(req.message, topic)
        difficulty = (
            "hard"
            if "hard" in lower
            else "easy"
            if "easy" in lower
            else "medium"
            if "medium" in lower
            else "adaptive"
        )

        qreq = QuizRequest(
            learner_id=req.learner_id,
            topic=selected,
            subject=selected,
            difficulty=difficulty,
            count=count,
        )

        if req.material_id:
            material = db.get(Material, req.material_id)
            if not material or material.learner_id != req.learner_id:
                raise HTTPException(404, "Learning material not found")
            quiz = await _generate_material_quiz(material, qreq, db)
        else:
            quiz = await generate_quiz(qreq, db)

        reply = (
            f"I prepared {count} new {selected} questions at "
            f"{quiz['difficulty']} difficulty."
        )
    else:
        provider = get_provider()
        reply = await provider.chat(req.message, context)
        if not reply:
            raise safe_gemini_failure(
                provider,
                "Gemini did not return a usable assistant response.",
            )
        interaction_id = getattr(provider, "last_interaction_id", None)
        if interaction_id:
            conversation.gemini_interaction_id = interaction_id

    assistant = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=reply,
    )
    db.add(assistant)
    db.commit()
    db.refresh(assistant)

    return {
        "conversation_id": conversation.id,
        "message": {
            "id": assistant.id,
            "role": assistant.role,
            "content": assistant.content,
            "created_at": assistant.created_at.isoformat(),
        },
        "quiz": quiz,
        "provider": "gemini",
    }
