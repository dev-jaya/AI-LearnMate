from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .db import Base


class Learner(Base):
    __tablename__ = "learners"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    attempts = relationship("Attempt", back_populates="learner", cascade="all, delete-orphan")
    quizzes = relationship("Quiz", back_populates="learner", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="learner", cascade="all, delete-orphan")
    generated_questions = relationship("GeneratedQuestion", back_populates="learner")
    question_history = relationship(
        "LearnerQuestionHistory",
        back_populates="learner",
        cascade="all, delete-orphan",
    )


class Subject(Base):
    __tablename__ = "subjects"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), unique=True, nullable=False)
    active = Column(Boolean, default=True, nullable=False)
    topics = relationship("Topic", back_populates="subject", cascade="all, delete-orphan")


class Topic(Base):
    __tablename__ = "topics"
    id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, ForeignKey("subjects.id"), nullable=False, index=True)
    name = Column(String(120), nullable=False)
    active = Column(Boolean, default=True, nullable=False)
    subject = relationship("Subject", back_populates="topics")


class Quiz(Base):
    __tablename__ = "quizzes"
    id = Column(Integer, primary_key=True)
    learner_id = Column(Integer, ForeignKey("learners.id"), nullable=False)
    material_id = Column(Integer, ForeignKey("learning_materials.id"), nullable=True, index=True)
    topic = Column(String(120), nullable=False)
    difficulty = Column(String(30), nullable=False)
    questions_json = Column(Text, nullable=False)
    submitted = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    learner = relationship("Learner", back_populates="quizzes")
    question_links = relationship(
        "QuizQuestion",
        back_populates="quiz",
        cascade="all, delete-orphan",
        order_by="QuizQuestion.position",
    )


class Attempt(Base):
    __tablename__ = "attempts"
    id = Column(Integer, primary_key=True)
    learner_id = Column(Integer, ForeignKey("learners.id"), nullable=False)
    topic = Column(String(120), nullable=False)
    difficulty = Column(String(30), nullable=False)
    score = Column(Float, nullable=False)
    total = Column(Integer, nullable=False)
    answers_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)
    learner = relationship("Learner", back_populates="attempts")


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    learner_id = Column(Integer, ForeignKey("learners.id"), nullable=False, index=True)
    gemini_interaction_id = Column(String(160), default="", nullable=False, index=True)
    topic = Column(String(120), default="", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    learner = relationship("Learner", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")


class GeneratedQuestion(Base):
    __tablename__ = "generated_questions"
    id = Column(Integer, primary_key=True)
    public_id = Column(String(80), unique=True, nullable=False, index=True)
    learner_id = Column(Integer, ForeignKey("learners.id"), nullable=True, index=True)
    material_id = Column(Integer, ForeignKey("learning_materials.id"), nullable=True, index=True)
    subject = Column(String(120), nullable=False, index=True)
    topic = Column(String(120), nullable=False, index=True)
    subtopic = Column(String(120), default="", nullable=False)
    difficulty = Column(String(30), nullable=False)
    question = Column(Text, nullable=False)
    normalized_question = Column(Text, nullable=False)
    options_json = Column(Text, nullable=False)
    answer = Column(Integer, nullable=False)
    explanation = Column(Text, nullable=False)
    fingerprint = Column(String(64), nullable=False, index=True)
    provider = Column(String(40), nullable=False, default="gemini")
    question_type = Column(String(40), nullable=False, default="conceptual")
    created_at = Column(DateTime, default=datetime.utcnow)
    learner = relationship("Learner", back_populates="generated_questions")


class QuestionFingerprint(Base):
    __tablename__ = "question_fingerprints"
    id = Column(Integer, primary_key=True)
    fingerprint = Column(String(64), nullable=False, index=True)
    question_id = Column(Integer, ForeignKey("generated_questions.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class QuizQuestion(Base):
    __tablename__ = "quiz_questions"
    id = Column(Integer, primary_key=True)
    quiz_id = Column(Integer, ForeignKey("quizzes.id"), nullable=False, index=True)
    question_id = Column(Integer, ForeignKey("generated_questions.id"), nullable=False, index=True)
    position = Column(Integer, nullable=False)
    quiz = relationship("Quiz", back_populates="question_links")


class MaterialChunk(Base):
    __tablename__ = "material_chunks"
    id = Column(Integer, primary_key=True)
    material_id = Column(Integer, ForeignKey("learning_materials.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    source_ref = Column(String(120), default="", nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class LearnerQuestionHistory(Base):
    __tablename__ = "learner_question_history"
    id = Column(Integer, primary_key=True)
    learner_id = Column(Integer, ForeignKey("learners.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("generated_questions.id"), nullable=False)
    fingerprint = Column(String(64), nullable=False, index=True)
    answered = Column(Boolean, default=False, nullable=False)
    correct = Column(Boolean, nullable=True)
    seen_at = Column(DateTime, default=datetime.utcnow)
    learner = relationship("Learner", back_populates="question_history")


Index(
    "ix_question_history_learner_fingerprint",
    LearnerQuestionHistory.learner_id,
    LearnerQuestionHistory.fingerprint,
)

