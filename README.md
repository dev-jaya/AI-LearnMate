# AI LearnMate — SIH26101 Smart Education

AI LearnMate is an AI-enabled learning platform for personalized training, competency-gap practice, automated assessment and conversational learning assistance. It is designed around the SIH26101 Smart Education problem statement: personalized learning, competency development, quizzes/MCQs from learning material, and an iGOT Karmayogi-oriented learning flow.

## Core workflow
1. Learner creates a lightweight demo profile.
2. Adaptive assessments estimate topic mastery and expose weak areas.
3. Learner uploads learning material or selects a public iGOT Karmayogi resource URL.
4. The backend extracts readable text and stores the material against the learner.
5. Google Gemini generates validated MCQs grounded in the selected material.
6. Question fingerprints and similarity checks prevent repeats for the learner.
7. Quiz results update mastery and recommendations.
8. The Gemini-powered tutor continues the conversation with learner context and selected material context when available.

## AI Assistant capabilities
- General questions and concept explanations
- Programming and engineering doubt solving
- Code explanation, debugging and corrected examples
- Exam-ready answers for 2-mark, 5-mark and 10-mark requests
- Telugu + English explanations when requested
- Practical learning, project, interview, viva, hackathon and study guidance
- Learning-progress and recent-activity review using application-provided data only
- Conversational history with persistent database storage
- Material-grounded assistance for uploaded or imported learning content

## SIH26101 feature coverage
- AI-enabled personalized learning platform
- Competency-gap detection from assessment history
- Personalized learning path and next-best-action recommendations
- Gemini-powered tutor conversation
- Dynamic, non-repeating MCQ generation
- Uploaded PDF/DOCX/PPTX/TXT/Markdown/CSV/JSON/HTML material ingestion
- Material-grounded MCQ generation from uploaded content
- Public iGOT Karmayogi resource import for demo/integration workflows
- Link to the official iGOT Karmayogi platform
- Persistent learner, quiz, question-history and conversation records

### iGOT integration note
The current demo implements a safe **public-resource connector**: users can paste a public iGOT Karmayogi resource URL and AI LearnMate imports readable public content for learning and assessment. Authenticated production API/SSO integration would require official credentials, API specifications and authorization from the iGOT/Karmayogi Bharat ecosystem.

## Technology
- Frontend: React + Vite
- Backend: Python + FastAPI
- Database: SQLite by default; PostgreSQL can be substituted later
- AI: Google Gemini API through the official `google-genai` SDK and Interactions API
- Current default AI model: `gemini-3.8-flash`
- Material extraction: pypdf plus built-in DOCX/PPTX/XML and text extraction with document-wide chunk indexing and coverage
- Analytics: mastery scoring, weak-topic detection and recommendations

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
```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally http://localhost:5173.

## Environment
Backend `.env` / Render environment:
```env
GEMINI_API_KEY=your_server_side_key
GEMINI_MODEL=gemini-3.8-flash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
DATABASE_URL=sqlite:///./ai_learnmate.db
FRONTEND_ORIGINS=http://localhost:5173,http://localhost:5500,http://127.0.0.1:5173,http://127.0.0.1:5500,https://ai-learnmate-frontend.onrender.com
```

The API key must remain server-side. Never put it in browser code or Vite environment variables.

Frontend production:
```env
VITE_API_URL=https://ai-learnmate-backend.onrender.com
```

## Material APIs
- `POST /api/materials/upload` — upload supported learning material
- `POST /api/materials/igot?learner_id=...&url=...` — import a public official iGOT URL
- `GET /api/materials/{learner_id}` — list learner materials
- `POST /api/materials/{material_id}/quiz` — generate material-grounded MCQs
- `GET /api/igot` — integration information and official links
- `POST /api/chat` — persistent conversational tutor; optional `material_id` grounds the answer in selected material

## Production note
Render free web services have ephemeral filesystems. SQLite data can be lost after restart, redeploy or spin-down. For a long-lived production deployment, use a persistent database such as managed PostgreSQL.

AI-generated assessment content should be reviewed before high-stakes use.
