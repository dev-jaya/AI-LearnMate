from uuid import uuid4

from fastapi.testclient import TestClient

from app import main


class FakeProvider:
    name = "openai"
    model = "test-model"
    api_key = "test-key"
    last_error = ""

    async def chat(self, message, context):
        return f"Echo: {message}"

    async def generate_material_questions(self, material_text, subject, topic, difficulty, count, excluded_questions):
        questions = []
        for index in range(count):
            questions.append({
                "id": f"test-{index}",
                "subject": subject,
                "topic": topic,
                "subtopic": "Material-based",
                "difficulty": difficulty,
                "question": f"What does the material explain in question {index}?",
                "options": ["The first concept", "The second concept", "The third concept", "The fourth concept"],
                "answer": 0,
                "explanation": "The answer is stated in the supplied material.",
                "fingerprint": f"test-fingerprint-{index}-{uuid4().hex}",
                "question_type": "conceptual",
                "provider": "openai",
            })
        return questions


def test_chat_history_is_saved_and_reloadable(monkeypatch):
    client = TestClient(main.app)
    learner = client.post("/api/learners", json={"name": f"CI-{uuid4().hex[:8]}"}).json()
    monkeypatch.setattr(main, "_openai_configured", lambda: True)
    monkeypatch.setattr(main, "OpenAIResponsesProvider", FakeProvider)

    first = client.post("/api/chat", json={
        "learner_id": learner["id"],
        "message": "Hello from automated tests",
        "topic": "Java",
    })
    assert first.status_code == 200, first.text
    conversation_id = first.json()["conversation_id"]

    history = client.get(f"/api/conversations/{learner['id']}")
    assert history.status_code == 200, history.text
    assert any(item["id"] == conversation_id for item in history.json()["conversations"])

    messages = client.get(f"/api/conversations/{conversation_id}/messages", params={"learner_id": learner["id"]})
    assert messages.status_code == 200, messages.text
    contents = [item["content"] for item in messages.json()["messages"]]
    assert "Hello from automated tests" in contents
    assert "Echo: Hello from automated tests" in contents


def test_material_quiz_uses_configured_provider_without_gemini(monkeypatch):
    client = TestClient(main.app)
    learner = client.post("/api/learners", json={"name": f"CI-Material-{uuid4().hex[:8]}"}).json()
    material = client.post(
        "/api/materials/upload",
        params={"learner_id": learner["id"]},
        files={"file": ("lesson.txt", ("Binary search works on sorted data by repeatedly checking the middle element and discarding the impossible half. " * 3).encode(), "text/plain")},
    )
    assert material.status_code == 200, material.text

    monkeypatch.setattr(main, "_openai_configured", lambda: True)
    monkeypatch.setattr(main, "OpenAIResponsesProvider", FakeProvider)

    response = client.post(
        f"/api/materials/{material.json()['id']}/quiz",
        json={"learner_id": learner["id"], "topic": "Algorithms", "difficulty": "easy", "count": 3},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["questions"]) == 3
    assert all(question["provider"] == "openai" for question in payload["questions"])
