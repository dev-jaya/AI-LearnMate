from pydantic import BaseModel, Field
from typing import List

class LearnerCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)

class LearnerOut(BaseModel):
    id: int
    name: str

class QuizRequest(BaseModel):
    learner_id: int
    topic: str
    subject: str | None = None
    subtopic: str = ""
    difficulty: str = "adaptive"
    count: int = Field(default=5, ge=3, le=20)

class Question(BaseModel):
    id: str
    question: str
    options: List[str]
    answer: int
    correct_answer: int
    explanation: str
    difficulty: str
    topic: str
    subject: str = ""
    subtopic: str = ""
    fingerprint: str = ""
    provider: str = "fallback"
    question_type: str = "conceptual"

class QuizOut(BaseModel):
    id: int
    topic: str
    difficulty: str
    questions: List[Question]

class SubmitRequest(BaseModel):
    quiz_id: int | None = None
    learner_id: int
    topic: str
    difficulty: str
    total: int | None = Field(default=None, gt=0, le=50)
    correct: int | None = Field(default=None, ge=0, le=50)
    answers: List[int] = Field(default_factory=list, max_length=50)

class ChatRequest(BaseModel):
    learner_id: int
    message: str = Field(min_length=1, max_length=4000)
    topic: str = ""
    conversation_id: int | None = None

class ChatMessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: str

class ChatOut(BaseModel):
    conversation_id: int
    message: ChatMessageOut
    quiz: QuizOut | None = None
    provider: str = "fallback"

class AttemptOut(BaseModel):
    id: int
    topic: str
    score: float
    total: int
    difficulty: str
    created_at: str
