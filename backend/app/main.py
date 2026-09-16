from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
import json
import re
import uuid

from .db import Base, engine, get_db
from .models import Learner, Attempt, Quiz, Conversation, Message, GeneratedQuestion, QuestionFingerprint, LearnerQuestionHistory, Subject, Topic, QuizQuestion
from .materials import Material, extract_text, retrieve_material_context, import_igot_resource
from .schemas import *
from .ai_providers import SUBJECTS, get_provider, question_similarity

Base.metadata.create_all(bind=engine)
app = FastAPI(title="AI LearnMate API", version="1.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:5500", "http://127.0.0.1:5173", "http://127.0.0.1:5500", "https://ai-learnmate-frontend.onrender.com"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def migrate_sqlite_columns():
    if not str(engine.url).startswith("sqlite"): return
    columns = {item["name"] for item in inspect(engine).get_columns("generated_questions")}
    additions = {"normalized_question": "TEXT NOT NULL DEFAULT ''", "question_type": "VARCHAR(40) NOT NULL DEFAULT 'conceptual'"}
    with engine.begin() as connection:
        for name, definition in additions.items():
            if name not in columns: connection.execute(text(f"ALTER TABLE generated_questions ADD COLUMN {name} {definition}"))


migrate_sqlite_columns()


def seed_subject_catalog():
    with Session(bind=engine) as db:
        for subject_name in SUBJECTS:
            if not db.query(Subject).filter(Subject.name == subject_name).first(): db.add(Subject(name=subject_name))
        db.commit()


seed_subject_catalog()


@app.get("/")
def root(): return {"name": "AI LearnMate", "status": "running", "version": "1.3.0"}


@app.get("/health")
def health(): return {"status": "ok", "service": "ai-learnmate-api"}


@app.get("/health/ai")
def ai_health():
    provider = get_provider()
    return {"provider": provider.name, "model": provider.model, "configured": bool(provider.api_key)}


@app.get("/api/igot")
def igot_info():
    return {"platform": "iGOT Karmayogi", "official_url": "https://igotkarmayogi.gov.in/", "public_content_url": "https://portal.igotkarmayogi.gov.in/", "integration_mode": "official public-resource import", "note": "Paste a public iGOT Karmayogi resource URL or upload official learning material."}


@app.post("/api/learners", response_model=LearnerOut)
def create_learner(data: LearnerCreate, db: Session = Depends(get_db)):
    name = data.name.strip()
    if not name: raise HTTPException(400, "Name cannot be empty")
    learner = Learner(name=name); db.add(learner); db.commit(); db.refresh(learner); return learner


@app.get("/api/learners/{learner_id}", response_model=LearnerOut)
def get_learner(learner_id: int, db: Session = Depends(get_db)):
    learner = db.get(Learner, learner_id)
    if not learner: raise HTTPException(404, "Learner not found")
    return learner


@app.get("/api/topics")
def topics(db: Session = Depends(get_db)):
    records = db.query(Subject).filter(Subject.active.is_(True)).order_by(Subject.name).all()
    return {"topics": [r.name for r in records], "subjects": [{"name": r.name, "topics": [t.name for t in r.topics if t.active]} for r in records]}


def learner_mastery(learner_id: int, db: Session):
    attempts = db.query(Attempt).filter(Attempt.learner_id == learner_id).order_by(Attempt.created_at.desc()).all(); scores = {}
    for attempt in attempts: scores.setdefault(attempt.topic, []).append(attempt.score)
    return {topic: round(sum(values) / len(values), 1) for topic, values in scores.items()}


def store_questions(db: Session, learner_id: int, questions: list[dict], topic: str, difficulty: str) -> Quiz:
    quiz = Quiz(learner_id=learner_id, topic=topic, difficulty=difficulty, questions_json=json.dumps(questions)); db.add(quiz); db.flush()
    for position, question in enumerate(questions):
        stored = GeneratedQuestion(public_id=question["id"], learner_id=learner_id, subject=question["subject"], topic=question["topic"], subtopic=question["subtopic"], difficulty=question["difficulty"], question=question["question"], normalized_question=re.sub(r"[^a-z0-9]+", " ", question["question"].lower()).strip(), options_json=json.dumps(question["options"]), answer=question["answer"], explanation=question["explanation"], fingerprint=question["fingerprint"], provider="gemini", question_type=question.get("question_type", "conceptual"))
        db.add(stored); db.flush(); db.add(QuestionFingerprint(fingerprint=question["fingerprint"], question_id=stored.id)); db.add(LearnerQuestionHistory(learner_id=learner_id, question_id=stored.id, fingerprint=question["fingerprint"])); db.add(QuizQuestion(quiz_id=quiz.id, question_id=stored.id, position=position))
    db.commit(); db.refresh(quiz); return quiz


async def generate_quiz(req: QuizRequest, db: Session):
    if not db.get(Learner, req.learner_id): raise HTTPException(404, "Learner not found")
    mastery = learner_mastery(req.learner_id, db); difficulty = req.difficulty
    if difficulty == "adaptive":
        score = mastery.get(req.topic); difficulty = "hard" if score is not None and score >= 80 else "medium" if score is not None and score >= 50 else "easy"
    history = db.query(LearnerQuestionHistory).filter(LearnerQuestionHistory.learner_id == req.learner_id).all(); ids = [r.question_id for r in history]
    previous = [r.question for r in db.query(GeneratedQuestion).filter(GeneratedQuestion.id.in_(ids), GeneratedQuestion.topic == req.topic).order_by(GeneratedQuestion.created_at.desc()).limit(30).all()] if ids else []
    provider = get_provider()
    if not provider.api_key: raise HTTPException(503, "Gemini is not configured on the server.")
    context = {"mastery": mastery, "weak_topics": [n for n,s in mastery.items() if s < 70], "strong_topics": [n for n,s in mastery.items() if s >= 80], "excluded_questions": previous}
    questions = []
    for _ in range(3):
        remaining = req.count - len(questions)
        if remaining <= 0: break
        candidates = await provider.generate_questions(req.subject or req.topic, req.topic, req.subtopic, difficulty, max(remaining * 2, 6), context)
        for question in candidates or []:
            if any(question_similarity(question["question"], old) >= .9 for old in previous + [q["question"] for q in questions]): continue
            question["id"] = f"gemini-{uuid.uuid4().hex[:16]}"; question["provider"] = "gemini"; questions.append(question)
            if len(questions) == req.count: break
        context["excluded_questions"] = previous + [q["question"] for q in questions]
    if len(questions) < req.count: raise HTTPException(502, f"Gemini produced only {len(questions)} validated new questions. Please try again.")
    quiz = store_questions(db, req.learner_id, questions, req.topic, difficulty)
    return {"id": quiz.id, "topic": req.topic, "difficulty": difficulty, "questions": questions}


@app.post("/api/quiz", response_model=QuizOut)
async def create_quiz(req: QuizRequest, db: Session = Depends(get_db)): return await generate_quiz(req, db)


@app.post("/api/assessment/start", response_model=QuizOut)
async def start_assessment(req: QuizRequest, db: Session = Depends(get_db)): return await generate_quiz(req, db)


@app.post("/api/attempts", response_model=AttemptOut)
def submit(req: SubmitRequest, db: Session = Depends(get_db)):
    if not db.get(Learner, req.learner_id): raise HTTPException(404, "Learner not found")
    quiz = db.get(Quiz, req.quiz_id) if req.quiz_id else None
    if not quiz or quiz.learner_id != req.learner_id: raise HTTPException(404, "Quiz not found")
    if quiz.submitted: raise HTTPException(409, "Quiz already submitted")
    questions = json.loads(quiz.questions_json)
    if len(req.answers) != len(questions): raise HTTPException(400, "Submit one answer for every question")
    if any(a not in range(4) for a in req.answers): raise HTTPException(400, "Answers must use option indexes 0 to 3")
    correct = sum(a == q["answer"] for a,q in zip(req.answers, questions)); quiz.submitted = True
    for answer, question in zip(req.answers, questions):
        stored = db.query(GeneratedQuestion).filter(GeneratedQuestion.public_id == question["id"]).first()
        if stored:
            row = db.query(LearnerQuestionHistory).filter(LearnerQuestionHistory.learner_id == req.learner_id, LearnerQuestionHistory.question_id == stored.id).order_by(LearnerQuestionHistory.seen_at.desc()).first()
            if row: row.answered = True; row.correct = answer == question["answer"]
    score = round(correct * 100 / len(questions), 2); attempt = Attempt(learner_id=req.learner_id, topic=quiz.topic, difficulty=quiz.difficulty, score=score, total=len(questions), answers_json=json.dumps(req.answers)); db.add(attempt); db.commit(); db.refresh(attempt)
    return {"id": attempt.id, "topic": attempt.topic, "score": attempt.score, "total": attempt.total, "difficulty": attempt.difficulty, "created_at": attempt.created_at.isoformat()}


@app.post("/api/assessment/submit", response_model=AttemptOut)
def submit_assessment(req: SubmitRequest, db: Session = Depends(get_db)): return submit(req, db)


@app.get("/api/dashboard/{learner_id}")
def dashboard(learner_id: int, db: Session = Depends(get_db)):
    learner = db.get(Learner, learner_id)
    if not learner: raise HTTPException(404, "Learner not found")
    attempts = db.query(Attempt).filter(Attempt.learner_id == learner_id).order_by(Attempt.created_at.desc()).all(); mastery = learner_mastery(learner_id, db); weak = sorted(mastery.items(), key=lambda x:x[1])[:3]
    recommendations = [{"topic": t, "reason": f"Mastery is {s}%. Practice this topic next.", "priority": "high"} for t,s in weak if s < 70] or [{"topic": "Python", "reason": "Start a diagnostic assessment to build your learning profile.", "priority": "medium"}]
    names = [x.name for x in db.query(Subject).filter(Subject.active.is_(True)).order_by(Subject.name).all()]
    return {"learner": {"id": learner.id, "name": learner.name}, "attempts": len(attempts), "average_score": round(sum(a.score for a in attempts)/len(attempts),1) if attempts else 0, "mastery": mastery, "weak_topics": [t for t,s in mastery.items() if s < 70], "strong_topics": [t for t,s in mastery.items() if s >= 80], "learning_path": [{"topic": t, "mastery": mastery.get(t), "action": "Advanced practice" if mastery.get(t,0)>=80 else "Revision and guided practice" if mastery.get(t,0)<50 else "Focused practice"} for t in names], "recommendations": recommendations, "recent": [{"topic":a.topic,"score":a.score,"difficulty":a.difficulty,"date":a.created_at.isoformat()} for a in attempts[:8]]}


@app.get("/api/learners/{learner_id}/progress")
def progress(learner_id:int, db:Session=Depends(get_db)):
    data=dashboard(learner_id,db); return {"learner":data["learner"],"mastery":data["mastery"],"recent":data["recent"]}


@app.get("/api/learners/{learner_id}/analytics")
def analytics(learner_id:int, db:Session=Depends(get_db)):
    data=dashboard(learner_id,db); return {"attempts":data["attempts"],"average_score":data["average_score"],"mastery":data["mastery"]}


@app.get("/api/learners/{learner_id}/recommendations")
def recommendations(learner_id:int, db:Session=Depends(get_db)): return {"recommendations":dashboard(learner_id,db)["recommendations"]}


@app.post("/api/materials/upload")
async def upload_material(learner_id:int, file:UploadFile=File(...), db:Session=Depends(get_db)):
    if not db.get(Learner,learner_id): raise HTTPException(404,"Learner not found")
    raw=await file.read()
    try: content=extract_text(file.filename or "material.txt",raw,file.content_type or "")
    except ValueError as exc: raise HTTPException(400,str(exc))
    material=Material(learner_id=learner_id,title=(file.filename or "Learning material")[:200],filename=file.filename or "material",mime_type=file.content_type or "application/octet-stream",source_type="upload",extracted_text=content); db.add(material); db.commit(); db.refresh(material)
    return {"id":material.id,"title":material.title,"filename":material.filename,"source_type":material.source_type,"characters":len(content)}


@app.post("/api/materials/igot")
async def import_igot(learner_id:int,url:str,db:Session=Depends(get_db)):
    if not db.get(Learner,learner_id): raise HTTPException(404,"Learner not found")
    try: title,content,ctype=await import_igot_resource(url)
    except Exception as exc: raise HTTPException(400,f"Could not import the public iGOT resource: {exc}")
    material=Material(learner_id=learner_id,title=title,filename=title,mime_type=ctype,source_type="igot",source_url=url,extracted_text=content); db.add(material); db.commit(); db.refresh(material)
    return {"id":material.id,"title":material.title,"source_type":"igot","source_url":material.source_url,"characters":len(content)}


@app.get("/api/materials/{learner_id}")
def list_materials(learner_id:int,db:Session=Depends(get_db)):
    if not db.get(Learner,learner_id): raise HTTPException(404,"Learner not found")
    items=db.query(Material).filter(Material.learner_id==learner_id).order_by(Material.created_at.desc()).all()
    return {"materials":[{"id":x.id,"title":x.title,"filename":x.filename,"source_type":x.source_type,"source_url":x.source_url,"characters":len(x.extracted_text),"created_at":x.created_at.isoformat()} for x in items]}


async def _generate_material_quiz(material:Material,req:QuizRequest,db:Session):
    provider=get_provider()
    if not provider.api_key: raise HTTPException(503,"Gemini is not configured on the server.")
    mastery=learner_mastery(req.learner_id,db); difficulty=req.difficulty
    if difficulty=="adaptive":
        score=mastery.get(req.topic); difficulty="hard" if score is not None and score>=80 else "medium" if score is not None and score>=50 else "easy"
    previous=[r.question for r in db.query(GeneratedQuestion).filter(GeneratedQuestion.learner_id==req.learner_id,GeneratedQuestion.topic==req.topic).order_by(GeneratedQuestion.created_at.desc()).limit(30).all()]
    source=retrieve_material_context(material.extracted_text,req.topic or material.title)
    if not source: raise HTTPException(400,"The material has no readable content.")
    questions=[]
    for _ in range(3):
        remaining=req.count-len(questions)
        if remaining<=0: break
        candidates=await provider.generate_material_questions(source,req.subject or req.topic or material.title,req.topic or material.title,difficulty,max(remaining*2,6),previous+[q["question"] for q in questions])
        for q in candidates or []:
            if any(question_similarity(q["question"],old)>=.9 for old in previous+[x["question"] for x in questions]): continue
            q["id"]=f"gemini-{uuid.uuid4().hex[:16]}"; q["provider"]="gemini"; questions.append(q)
            if len(questions)==req.count: break
    if len(questions)<req.count: raise HTTPException(502,"Gemini could not produce enough new material-grounded questions. Try a shorter or clearer source document.")
    quiz=store_questions(db,req.learner_id,questions,req.topic or material.title,difficulty); return {"id":quiz.id,"topic":req.topic or material.title,"difficulty":difficulty,"questions":questions}


@app.post("/api/materials/{material_id}/quiz",response_model=QuizOut)
async def material_quiz(material_id:int,req:QuizRequest,db:Session=Depends(get_db)):
    material=db.get(Material,material_id)
    if not material or material.learner_id!=req.learner_id: raise HTTPException(404,"Learning material not found")
    return await _generate_material_quiz(material,req,db)


def tutor_context(learner_id:int,topic:str,db:Session,material_id:int|None=None):
    learner=db.get(Learner,learner_id)
    if not learner: raise HTTPException(404,"Learner not found")
    mastery=learner_mastery(learner_id,db); recent=db.query(Attempt).filter(Attempt.learner_id==learner_id).order_by(Attempt.created_at.desc()).limit(5).all(); mistakes=db.query(LearnerQuestionHistory).filter(LearnerQuestionHistory.learner_id==learner_id,LearnerQuestionHistory.correct.is_(False)).order_by(LearnerQuestionHistory.seen_at.desc()).limit(5).all(); mistake_questions=[db.get(GeneratedQuestion,x.question_id) for x in mistakes]
    context={"learner_name":learner.name,"topic":topic,"mastery":mastery,"weak_topics":[n for n,s in mastery.items() if s<70],"strong_topics":[n for n,s in mastery.items() if s>=80],"recent_results":[{"topic":a.topic,"score":a.score,"difficulty":a.difficulty} for a in recent],"recent_mistakes":[q.subtopic for q in mistake_questions if q]}
    if material_id:
        material=db.get(Material,material_id)
        if not material or material.learner_id!=learner_id: raise HTTPException(404,"Learning material not found")
        context["material_title"]=material.title; context["material_context"]=retrieve_material_context(material.extracted_text,topic or material.title)
    return context


@app.get("/api/conversations/{learner_id}")
def list_conversations(learner_id:int,db:Session=Depends(get_db)):
    if not db.get(Learner,learner_id): raise HTTPException(404,"Learner not found")
    rows=db.query(Conversation).filter(Conversation.learner_id==learner_id).order_by(Conversation.updated_at.desc(),Conversation.created_at.desc()).all(); result=[]
    for c in rows:
        first=db.query(Message).filter(Message.conversation_id==c.id,Message.role=="user").order_by(Message.created_at.asc()).first(); last=db.query(Message).filter(Message.conversation_id==c.id).order_by(Message.created_at.desc()).first(); result.append({"id":c.id,"topic":c.topic,"title":first.content[:70] if first else c.topic or "New conversation","preview":last.content[:100] if last else "","created_at":c.created_at.isoformat(),"updated_at":c.updated_at.isoformat() if c.updated_at else c.created_at.isoformat()})
    return {"conversations":result}


@app.get("/api/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id:int,learner_id:int,db:Session=Depends(get_db)):
    c=db.get(Conversation,conversation_id)
    if not c or c.learner_id!=learner_id: raise HTTPException(404,"Conversation not found")
    messages=db.query(Message).filter(Message.conversation_id==conversation_id).order_by(Message.created_at.asc()).all(); return {"conversation_id":conversation_id,"topic":c.topic,"messages":[{"id":m.id,"role":m.role,"content":m.content,"created_at":m.created_at.isoformat()} for m in messages]}


@app.post("/api/chat",response_model=ChatOut)
async def chat(req:ChatRequest,db:Session=Depends(get_db)):
    if not db.get(Learner,req.learner_id): raise HTTPException(404,"Learner not found")
    conversation=db.get(Conversation,req.conversation_id) if req.conversation_id else None
    if conversation and conversation.learner_id!=req.learner_id: raise HTTPException(403,"Conversation does not belong to learner")
    if not conversation: conversation=Conversation(learner_id=req.learner_id,topic=req.topic); db.add(conversation); db.flush()
    topic=req.topic or conversation.topic or "General"; conversation.topic=topic; db.add(Message(conversation_id=conversation.id,role="user",content=req.message)); db.flush()
    context=tutor_context(req.learner_id,topic,db,req.material_id); history=db.query(Message).filter(Message.conversation_id==conversation.id).order_by(Message.created_at.desc()).limit(12).all(); context["conversation"]=[{"role":m.role,"content":m.content} for m in reversed(history)]
    lower=req.message.lower(); quiz=None; quiz_intent=any(p in lower for p in ("test me","quiz me","mcq","ask me","give me questions","create a quiz","generate questions"))
    if quiz_intent:
        match=re.search(r"\b(\d+)\b",lower); count=min(max(int(match.group(1)) if match else 5,3),20); selected=next((s for s in SUBJECTS if s.lower() in lower),topic); difficulty="hard" if "hard" in lower else "easy" if "easy" in lower else "medium" if "medium" in lower else "adaptive"; qreq=QuizRequest(learner_id=req.learner_id,topic=selected,subject=selected,difficulty=difficulty,count=count)
        if req.material_id:
            material=db.get(Material,req.material_id)
            if not material or material.learner_id!=req.learner_id: raise HTTPException(404,"Learning material not found")
            quiz=await _generate_material_quiz(material,qreq,db)
        else: quiz=await generate_quiz(qreq,db)
        reply=f"I prepared {count} new {selected} questions at {quiz['difficulty']} difficulty."
    else:
        provider=get_provider(); reply=await provider.chat(req.message,context)
        if not reply: raise HTTPException(503,f"Gemini is unavailable right now: {provider.last_error or 'no response returned'}")
    assistant=Message(conversation_id=conversation.id,role="assistant",content=reply); db.add(assistant); db.commit(); db.refresh(assistant)
    return {"conversation_id":conversation.id,"message":{"id":assistant.id,"role":assistant.role,"content":assistant.content,"created_at":assistant.created_at.isoformat()},"quiz":quiz,"provider":"gemini"}
