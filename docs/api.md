# API summary

- `GET /health` — service health
- `GET /health/ai` — current AI provider, model and server-side configuration state
- `POST /api/learners` — create learner profile
- `GET /api/learners/{learner_id}` — read learner profile
- `GET /api/topics` — available topics
- `POST /api/quiz` or `POST /api/assessment/start` — persist and generate an adaptive quiz
- `POST /api/attempts` or `POST /api/assessment/submit` — submit answers; persisted quizzes are scored by the server
- `GET /api/dashboard/{learner_id}` — mastery, weak/strong topics, learning path, recommendations and recent attempts
- `GET /api/learners/{learner_id}/progress` — mastery and recent progress
- `GET /api/learners/{learner_id}/analytics` — attempt and average-score analytics
- `GET /api/learners/{learner_id}/recommendations` — personalized next actions
- `POST /api/chat` — persistent tutor conversation; returns a quiz when the learner asks to be tested

Quiz submissions should use the persisted `quiz_id` and an `answers` array of option indexes. The backend rejects missing or invalid submissions and does not trust a client-provided score.

Question responses include `subject`, `topic`, `subtopic`, `difficulty`, `provider`, and a SHA-256 `fingerprint`. Learner question history is used to exclude previously shown questions before a new quiz is persisted.

The topic response includes both a flat `topics` list and database-backed `subjects` records with their active topic names. Quiz questions are also linked through the `quiz_questions` table for persistence and future question-level analytics.

## Gemini configuration

Use `backend/.env.example` as the template. The backend expects a server-side `GEMINI_API_KEY`; the model is selected with `GEMINI_MODEL` and the Google Generative Language endpoint with `GEMINI_BASE_URL`.

Gemini is the only AI provider used by the application. The frontend never receives the API key and calls the backend `/api/chat` and assessment routes instead.

Interactive OpenAPI docs are exposed by FastAPI at `/docs` while the backend is running.
