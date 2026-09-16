# Architecture

```text
                 ┌──────────────────────────┐
                 │       React Frontend     │
                 │ Dashboard / Quiz / Chat  │
                 └────────────┬─────────────┘
                              │ REST/JSON
                 ┌────────────▼─────────────┐
                 │        FastAPI API       │
                 │ Learners / Quiz / Chat   │
                 └───────┬───────────┬──────┘
                         │           │
                 ┌───────▼───┐   ┌──▼────────────────┐
                 │ SQLite DB │   │ LLM Provider       │
                 │ learners  │   │ abstraction        │
                 │ attempts  │   └─────────┬──────────┘
                 │ chat data │             │
                 └───────────┘     ┌───────▼──────────┐
                                   │ Google Gemini API │
                                   │ server-side key   │
                                   └───────────────────┘
```

## AI request flow

The browser sends chat and assessment requests only to the FastAPI backend. The backend builds the Gemini request from the current user message, recent conversation history, learner progress data and optional selected learning material context. The Gemini API key remains on the server.

## Adaptive loop

Diagnostic → mastery estimate → weak-topic ranking → difficulty selection → Gemini MCQ generation → answer validation → score → learner-model update → next recommendation.

## Conversational context

The application persists conversations and messages in the database. Recent messages are sent to Gemini for genuine follow-ups, while learner progress and material context are supplied separately as supporting context. New requests take priority over stale topic selections.

## Production extension

- PostgreSQL for institutional scale
- JWT/OAuth/college SSO
- teacher/admin portal
- content repository and richer RAG/embeddings
- question quality classifier and duplicate detector
- audit logs and consent/privacy controls
- multilingual content
- deployment with HTTPS and managed secrets
