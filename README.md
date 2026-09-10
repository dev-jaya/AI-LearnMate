# AI LearnMate — AI Learning Platform with Personalized Training and Automated MCQ Generation

A complete hackathon-ready MVP implementing the SIH proposal: learner profiling, diagnostic assessment, adaptive recommendations, topic-wise MCQ generation, scoring, progress analytics, and an optional LLM integration.

## Core workflow
1. Learner selects a subject/topic and takes a diagnostic quiz.
2. The adaptive engine estimates topic mastery.
3. Weak topics receive higher practice priority.
4. MCQs are generated from the selected topic and difficulty.
5. Quiz results update the learner model.
6. Dashboard shows mastery, weak areas, streak/progress, and recommendations.

## Technology
- Frontend: React + Vite
- Backend: Python + FastAPI
- Database: SQLite by default; PostgreSQL can be substituted later
- AI layer: Gemini-primary provider abstraction with optional Ollama/Hugging Face/OpenAI-compatible providers and validated local fallback generation
- Analytics: mastery scoring and recommendation engine
- Tutor: persistent multi-turn conversations with adaptive chat-to-quiz transitions

React's current documentation recommends modern React app setups rather than Create React App, and FastAPI provides automatic API documentation and a production-oriented Python API framework. See the official docs linked in `docs/references.md`.

## Run locally
### Backend
```bash
cd backend
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
API: http://localhost:8000
Swagger: http://localhost:8000/docs

### Frontend
Requires Node.js.
```bash
cd frontend
npm install
npm run dev
```
Open the URL shown by Vite, normally http://localhost:5173.

## Optional LLM
Copy `backend/.env.example` to `backend/.env` and set:
- `GEMINI_API_KEY` to make Gemini the primary generator and tutor
- `GEMINI_MODEL` (default `gemini-2.0-flash`)
- `GEMINI_BASE_URL` (default Google Gemini REST API URL)
- `AI_PROVIDER=fallback`, `ollama`, or `huggingface`
- `OLLAMA_BASE_URL` and `OLLAMA_MODEL` for local Ollama
- `HF_BASE_URL`, `HF_API_KEY`, and `HF_MODEL` for a Hugging Face-compatible endpoint
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`

When `GEMINI_API_KEY` is configured, Gemini is selected first for both MCQ generation and tutor chat. The backend validates Gemini output and retries once before using the fallback provider. If the key is missing or Gemini is unavailable, the application remains usable with its local generator. The subject catalog is stored in SQLite and currently covers C, C++, Java, Python, Data Structures, Algorithms, DBMS, Operating Systems, Computer Networks, Computer Organization, Software Engineering, Web Development, Artificial Intelligence, Machine Learning, Cybersecurity, and Cloud Computing.

## Dynamic questions and tutor
`GET /api/topics` returns the SQLite-backed subject catalog. `POST /api/quiz` and `POST /api/assessment/start` accept `subject`, `topic`, `subtopic`, `difficulty`, and `count`. Generated questions are validated, fingerprinted, stored, linked through `quiz_questions`, and excluded from later quizzes for the same learner. The service attempts a larger candidate pool before returning the requested count.

`POST /api/chat` accepts a learner ID, message, topic, and optional conversation ID. Messages and tutor replies are stored in SQLite. Requests containing `test me`, `MCQ`, or `ask me` transition into a persisted adaptive quiz.

The AI layer is RAG-ready through `backend/app/rag.py`, which defines ingestion, chunking, retrieval, and context-injection interfaces for future PDF or notes support.

## Demo login
The MVP intentionally has no production authentication. Enter any learner name on the landing screen. Add institutional SSO/OAuth before deployment.

## Important
This is an MVP/prototype implementation for a hackathon. It is not a production educational assessment system. AI-generated questions should be reviewed before high-stakes use.
