# API summary

- `GET /health` — service health
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

Quiz submissions should use the persisted `quiz_id` and an `answers` array of option indexes. The backend rejects missing, invalid, or duplicate submissions and does not trust a client-provided score.

Question responses include `subject`, `topic`, `subtopic`, `difficulty`, `provider`, and a SHA-256 `fingerprint`. The learner question history excludes previously shown fingerprints before a new quiz is persisted.

The topic response includes both a flat `topics` list and database-backed `subjects` records with their active topic names. Quiz questions are also linked through the `quiz_questions` table for persistence and future question-level analytics.

## Provider configuration

Use `backend/.env.example` as the template. Set `GEMINI_API_KEY` to make Gemini the primary provider for MCQs and chat. `GEMINI_MODEL` defaults to `gemini-2.0-flash`, and `GEMINI_BASE_URL` defaults to Google's Generative Language REST API. Gemini failures are retried once, then the local provider is used. `AI_PROVIDER=fallback` remains a no-key mode; existing Ollama, Hugging Face, and OpenAI-compatible settings remain available as optional providers when Gemini is not configured.

Interactive OpenAPI docs are exposed by FastAPI at `/docs` while the backend is running.
