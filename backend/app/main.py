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
from .openai_responses import OpenAIResponsesProvider

Base.metadata.create_all(bind=engine)
app = FastAPI(title="AI LearnMate API", version="1.2.0")
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
    return {"name": "AI LearnMate", "status": "running", "version": "1.2.0"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-learnmate-api"}


@app.get("/health/ai")
def ai_health():
    provider = OpenAIResponsesProvider() if settings_for_openai() else get_provider()
    return {"provider": provider.name, "model": getattr(provider, "model", None), "key_loaded": bool(getattr(provider, "api_key", ""))}


def settings_for_openai():
    from .config import settings
    return bool(settings.llm_api_key and settings.llm_model)


@app.get("/api/igot")
def igot_info():
    return {
        "platform": "iGOT Karmayogi",
        "official_url": "https://igotkarmayogi.gov.in/",
        "public_content_url": "https://portal.igotkarmayogi.gov.in/",
        "integration_mode": "official public-resource import",
        "note": "Paste a public iGOT Karmayogi resource URL or upload the official learning material. Authenticated government APIs require official credentials and access not available to this demo."
    }


def learner_mastery(learner_id: int, db: Session):
    attempts = db.query(Attempt).filter(Attempt.learner_id == learner_id).all()
    scores = {}
    for attempt in attempts:
        scores.setdefault(attempt.topic, []).append(attempt.score)
    return {topic: round(sum(values) / len(values), 1) for topic, values in scores.items()}


def generate_quiz(req: QuizRequest, db: Session):
    ...
