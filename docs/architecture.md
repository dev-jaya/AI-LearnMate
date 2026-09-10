# Architecture

```text
                 ┌──────────────────────────┐
                 │       React Frontend     │
                 │ Dashboard / Quiz / UX    │
                 └────────────┬─────────────┘
                              │ REST/JSON
                 ┌────────────▼─────────────┐
                 │        FastAPI API       │
                 │ Learners / Quiz / Scores │
                 └───────┬───────────┬──────┘
                         │           │
                 ┌───────▼───┐   ┌──▼─────────────┐
                 │ SQLite DB │   │ Adaptive/AI     │
                 │ learners  │   │ Engine          │
                 │ attempts  │   │ local + LLM     │
                 └───────────┘   └─────────────────┘
```

## Adaptive loop
Diagnostic → mastery estimate → weak-topic ranking → difficulty selection → MCQ generation → answer validation → score → learner-model update → next recommendation.

## Production extension
- PostgreSQL for institutional scale
- JWT/OAuth/college SSO
- teacher/admin portal
- content repository and embeddings/RAG
- question quality classifier and duplicate detector
- audit logs and consent/privacy controls
- multilingual content
- deployment with HTTPS and managed secrets
