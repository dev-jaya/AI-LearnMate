# AI LearnMate — SIH26101 Smart Education

AI LearnMate is an AI-enabled learning platform for personalized training, competency-gap practice and automated assessment. It is designed around the SIH26101 MoSPI Smart Education problem statement: personalized learning, competency development, quizzes/MCQs from learning material, and an iGOT Karmayogi-oriented learning flow.

## Core workflow
1. Learner creates a lightweight demo profile.
2. Adaptive assessments estimate topic mastery and expose weak areas.
3. Learner uploads a learning material or selects a public iGOT Karmayogi resource URL.
4. The backend extracts readable text and stores the material against the learner.
5. Gemini generates validated MCQs grounded in the selected material.
6. Question fingerprints and similarity checks prevent repeats for the learner.
7. Quiz results update mastery and recommendations.
8. The Gemini tutor can continue the conversation with learner context and selected material context.

## SIH26101 feature coverage
- AI-enabled personalized learning platform
- Competency-gap detection from assessment history
- Personalized learning path and next-best-action recommendations
- Gemini-powered normal tutor conversation
- Dynamic, non-repeating MCQ generation
- Uploaded PDF/DOCX/PPTX/TXT/Markdown/CSV/JSON/HTML material ingestion
- Material-grounded MCQ generation from uploaded content
- Public iGOT Karmayogi resource import for demo/integration workflows
- Link to the official iGOT Karmayogi platform
- Persistent learner, quiz, question-history and conversation records

### iGOT integration note
The current demo implements a safe **public-resource connector**: users can paste a public iGOT Karmayogi resource URL and AI LearnMate imports readable public content for learning/assessment. Official iGOT pages document authenticated government-user access and role-based portals; authenticated production API/SSO integration would require official credentials, API specifications and authorization from the iGOT/Karmayogi Bharat ecosystem. AI LearnMate does not claim an authenticated government API integration without those prerequisites.

## Technology
- Frontend: React + Vite
- Backend: Python + FastAPI
- Database: SQLite by default; PostgreSQL can be substituted later
- AI: Gemini-primary provider with validated local fallback
- Material extraction: pypdf plus built-in DOCX/PPTX/XML and text extraction
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
Backend `.env`:
- `GEMINI_API_KEY` — backend-only Gemini secret
- `GEMINI_MODEL=gemini-3.6-flash`
- `GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta`

Frontend production:
- `VITE_API_URL=https://ai-learnmate-backend.onrender.com`

Never put `GEMINI_API_KEY` in frontend code or Vite environment variables.

## Material APIs
- `POST /api/materials/upload` — upload supported learning material
- `POST /api/materials/igot?learner_id=...&url=...` — import a public official iGOT URL
- `GET /api/materials/{learner_id}` — list learner materials
- `POST /api/materials/{material_id}/quiz` — generate material-grounded MCQs
- `GET /api/igot` — integration information and official links
- `POST /api/chat` accepts optional `material_id` to ground tutor responses in the selected material

## Important production note
Render free web services have ephemeral filesystems. SQLite data on the service can be lost after restart, redeploy or spin-down. For a long-lived production deployment, use a persistent database such as managed PostgreSQL. The current SQLite setup is suitable for a hackathon/demo deployment but should not be presented as durable production storage.

AI-generated assessment content should be reviewed before high-stakes use.
